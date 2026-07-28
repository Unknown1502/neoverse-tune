#!/usr/bin/env python3
"""
test_loadgen.py — validate the load generator against a server with known timing.

A latency measurement tool that is itself wrong produces confident nonsense. So
before loadgen.py is pointed at llama-server, it is pointed at a mock that
deliberately stalls a known number of milliseconds before the first token and a
known number between subsequent tokens. If loadgen is measuring what it claims,
the reported TTFT and TPOT must land on those planted values.

The mock is a plain http.server speaking the same SSE shape llama-server's
OpenAI-compatible endpoint emits. No Arm hardware, no model, no network.

Run:  python3 tools/test_loadgen.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
LOADGEN = os.path.join(HERE, "loadgen.py")

# Ground truth planted in the mock.
PREFILL_MS = 120.0    # stall before the first token  -> should show up as TTFT
DECODE_MS = 20.0      # stall between later tokens    -> should show up as TPOT
N_TOKENS = 12


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep the test output readable
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
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(text):
            payload = {"choices": [{"delta": {"content": text}, "index": 0}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        time.sleep(PREFILL_MS / 1000.0)          # prefill stall
        try:
            self.wfile.write(chunk("{"))
            self.wfile.flush()
            for i in range(N_TOKENS - 1):
                time.sleep(DECODE_MS / 1000.0)   # per-token decode stall
                self.wfile.write(chunk(f"t{i}"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    print("\nLoad generator — validated against planted timings")
    print(f"  ground truth: TTFT={PREFILL_MS:.0f}ms  TPOT={DECODE_MS:.0f}ms  "
          f"tokens={N_TOKENS}\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failures: list[str] = []
    tmp = tempfile.mkdtemp(prefix="specarm-loadgen-")
    out_path = os.path.join(tmp, "serve.json")

    try:
        proc = subprocess.run(
            [sys.executable, LOADGEN,
             "--url", f"http://127.0.0.1:{port}",
             "--concurrency", "4", "--requests", "12",
             "--warmup", "2", "--max-tokens", str(N_TOKENS),
             "--label", "mock", "--out", out_path],
            capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            print("FAIL  loadgen exited nonzero")
            return 1

        report = json.load(open(out_path))

        def check(name: str, ok: bool, detail: str = "") -> None:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}"
                  f"{('  [' + detail + ']') if detail else ''}")
            if not ok:
                failures.append(name)

        check("all requests succeeded",
              report["requests_failed"] == 0 and report["requests_ok"] == 12,
              f"ok={report['requests_ok']} failed={report['requests_failed']}")

        # TTFT must recover the prefill stall. Generous upper bound: the mock is
        # threaded and concurrency 4 on a shared CPU adds scheduling jitter.
        ttft_p50 = report["ttft_ms"]["p50"]
        check("TTFT recovers the planted prefill stall",
              PREFILL_MS * 0.75 <= ttft_p50 <= PREFILL_MS * 2.5,
              f"p50={ttft_p50:.1f}ms vs planted {PREFILL_MS:.0f}ms")

        # TPOT must recover the inter-token stall, and critically must NOT be
        # contaminated by the much larger prefill stall.
        tpot_p50 = report["tpot_ms"]["p50"]
        check("TPOT recovers the planted decode stall",
              DECODE_MS * 0.5 <= tpot_p50 <= DECODE_MS * 3.0,
              f"p50={tpot_p50:.1f}ms vs planted {DECODE_MS:.0f}ms")

        check("TPOT is not contaminated by prefill",
              tpot_p50 < PREFILL_MS * 0.75,
              f"tpot p50={tpot_p50:.1f}ms must stay well under TTFT {PREFILL_MS:.0f}ms")

        check("token accounting is right",
              report["total_output_tokens"] == 12 * N_TOKENS,
              f"got {report['total_output_tokens']}, expected {12 * N_TOKENS}")

        check("throughput reported", report["output_tokens_per_sec"] > 0)

        # Percentiles must be ordered; an unordered percentile function is a
        # silent correctness bug that only shows up in the final table.
        t = report["total_ms"]
        check("percentiles are monotonic",
              t["p50"] <= t["p90"] <= t["p95"] <= t["p99"] <= t["max"],
              f"p50={t['p50']} p90={t['p90']} p95={t['p95']} p99={t['p99']}")

    finally:
        server.shutdown()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Load generator checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
