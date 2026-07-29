#!/usr/bin/env python3
"""
test_probe_prefill.py — prove the opportunity sizer can tell the two cases apart.

probe_prefill.py is about to decide whether two weeks of work is worth doing. If
its verdict logic is wrong, it will send you down the wrong path with total
confidence. So it is first pointed at two mock servers whose behaviour is known
by construction:

  NO_REUSE mock    — first-token latency proportional to TOTAL prompt tokens
  REUSE mock       — first-token latency proportional to NEW tokens only

The probe must return NO_REUSE for the first and REUSE_WORKS for the second.

Run:  python3 tools/test_probe_prefill.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe_prefill.py")

MS_PER_TOKEN = 0.25          # simulated prefill cost
FAILURES: list[str] = []


def make_handler(reuse: bool):
    """Build a mock whose prefill cost depends on the caching hypothesis."""
    state = {"seen": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/health"):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            chars = sum(len(m.get("content", "")) for m in payload.get("messages", []))
            total_tokens = max(1, chars // 4)

            with lock:
                prev = state["seen"]
                state["seen"] = total_tokens
            new_tokens = max(1, total_tokens - prev)

            # THE hypothesis under test.
            billed = new_tokens if reuse else total_tokens
            prefill_s = billed * MS_PER_TOKEN / 1000.0

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def chunk(text, usage=None):
                obj = {"choices": [{"delta": {"content": text}, "index": 0}]}
                if usage:
                    obj["usage"] = usage
                return f"data: {json.dumps(obj)}\n\n".encode()

            time.sleep(prefill_s)
            try:
                self.wfile.write(chunk("{", {"prompt_tokens": total_tokens}))
                self.wfile.flush()
                for i in range(7):
                    time.sleep(0.002)
                    self.wfile.write(chunk(f"t{i}"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def run_case(label: str, reuse: bool, expect: str) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reuse))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        proc = subprocess.run(
            [sys.executable, PROBE, "--url", f"http://127.0.0.1:{port}",
             "--turns", "6", "--max-tokens", "8", "--label", label],
            capture_output=True, text=True, encoding="utf-8", timeout=300)
        out = proc.stdout
        got = "UNKNOWN"
        for token in ("NO_REUSE", "REUSE_WORKS", "PARTIAL_REUSE", "INCONCLUSIVE"):
            if f"VERDICT: {token}" in out:
                got = token
                break
        ok = got == expect
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<28} -> {got} (expected {expect})")
        if not ok:
            FAILURES.append(f"{label}: got {got}, expected {expect}")
            print(out)
    finally:
        server.shutdown()


def main() -> int:
    print("\nOpportunity sizer — can it tell caching from no caching?\n")
    run_case("server re-prefills everything", reuse=False, expect="NO_REUSE")
    run_case("server reuses the prefix", reuse=True, expect="REUSE_WORKS")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        return 1
    print("Discriminator works: the probe's verdict can be trusted.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
