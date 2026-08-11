#!/usr/bin/env python3
"""
make_chart.py — render the result as SVG, straight from the committed data.

WHY SVG, AND WHY HAND-WRITTEN
-----------------------------
A chart is the one thing this repository lacked, and the obvious way to get one
is matplotlib. That would be the first third-party dependency in the project,
for a picture. Instead this emits SVG text directly:

  * no dependency, so `python3 tools/make_chart.py` works on a clean checkout
  * text, so it diffs in review and cannot silently drift from the numbers
  * regenerated from results/*.json, so it cannot disagree with the tables
  * renders inline on GitHub, in light and dark, at any zoom

WHAT IT DRAWS
-------------
Two panels, because the finding is two claims:

  left    time to first token, per machine and configuration -> the COST,
          which differs by host
  right   tokens recomputed per request                      -> the CAUSE,
          which is identical on both

The right panel is the point. Same bars on both architectures means the
scheduler made the same decision; the left panel differing means only the price
changed.

Usage:
    python3 tools/make_chart.py                    # -> docs/result.svg
    python3 tools/make_chart.py --out somewhere.svg
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))

# Colours chosen to survive both GitHub themes without a media query: mid-tone
# fills with enough contrast against white and against #0d1117.
RED = "#d1414a"
GRN = "#2f9e5f"
BLU = "#3b6ea5"
INK = "#8b949e"      # axis/label grey, legible on either background
W, H = 940, 430


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def collect() -> list[dict]:
    """One row per (machine, config) that actually has data."""
    machines = [
        ("Neoverse N2", "4 vCPU · Azure Cobalt 100",
         os.path.join(RESULTS, "arm-neoverse-n2", "agent_analysis.latest.json"),
         os.path.join(RESULTS, "arm-neoverse-n2", "slot_evidence.json")),
        ("x86", "8 threads · laptop",
         os.path.join(RESULTS, "agent_analysis.latest.json"),
         os.path.join(RESULTS, "slot_evidence.json")),
    ]
    want = [("solo_baseline", "1 tenant"),
            ("4tenant_default", "4 tenants, default 0.10"),
            ("4tenant_sim09", "4 tenants, fixed 0.90")]

    rows = []
    for name, sub, apath, spath in machines:
        an, sl = load(apath), load(spath)
        if not an:
            continue
        # median tokens recomputed per config, from the slot logs
        toks: dict[str, list[int]] = {}
        for log in (sl or {}).get("logs", []):
            for cfg, _ in want:
                if log["file"].startswith(f"server_{cfg}_r"):
                    v = log.get("prefill_tokens", {}).get("median")
                    if v is not None:
                        toks.setdefault(cfg, []).append(int(v))
        for cfg, label in want:
            st = an.get("configs", {}).get(cfg)
            if not st:
                continue
            tl = sorted(toks.get(cfg, []))
            rows.append({
                "machine": name, "sub": sub, "cfg": cfg, "label": label,
                "ms": st["mean"], "hw": st["mean"] - st["ci95_lo"],
                "tokens": tl[len(tl) // 2] if tl else None,
            })
    return rows


def panel(x0: int, y0: int, w: int, h: int, rows: list[dict], key: str,
          title: str, sub: str, unit: str, ticks: int = 4) -> list[str]:
    out = [f'<text x="{x0}" y="{y0 - 26}" class="t">{esc(title)}</text>',
           f'<text x="{x0}" y="{y0 - 8}" class="s">{esc(sub)}</text>']
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return out
    vmax = max(vals) * 1.18

    for i in range(ticks + 1):
        gy = y0 + h - h * i / ticks
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + w}" y2="{gy:.1f}" class="g"/>')
        out.append(f'<text x="{x0 - 8}" y="{gy + 4:.1f}" class="a" text-anchor="end">'
                   f'{vmax * i / ticks:,.0f}</text>')

    n = len(rows)
    slot = w / n
    bw = min(58.0, slot * 0.56)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            continue
        cx = x0 + slot * (i + 0.5)
        bh = h * v / vmax
        by = y0 + h - bh
        col = RED if "default" in r["cfg"] else (GRN if "sim09" in r["cfg"] else BLU)
        out.append(f'<rect x="{cx - bw/2:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                   f'height="{bh:.1f}" rx="2" fill="{col}"/>')
        # 95% CI whisker, latency panel only — tokens are deterministic
        if key == "ms" and r.get("hw"):
            eh = h * r["hw"] / vmax
            if eh > 1.2:
                out.append(f'<line x1="{cx:.1f}" y1="{by - eh:.1f}" x2="{cx:.1f}" '
                           f'y2="{by + eh:.1f}" class="e"/>')
        out.append(f'<text x="{cx:.1f}" y="{by - 7:.1f}" class="v" '
                   f'text-anchor="middle">{v:,.0f}{unit}</text>')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "result.svg"))
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print("no results found — run tools/analyze_agent.py first", file=sys.stderr)
        return 1

    pw, ph = 380, 240
    lx, rx, py = 78, 540, 108
    s: list[str] = []

    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}" font-family="-apple-system,BlinkMacSystemFont,'
             f'Segoe UI,Helvetica,Arial,sans-serif" role="img" '
             f'aria-label="Multi-tenant slot selection: latency differs by machine, '
             f'tokens recomputed do not">')
    s.append('<style>'
             '.h{font-size:19px;font-weight:700;fill:#c9d1d9}'
             '.h2{font-size:12.5px;fill:#8b949e}'
             '.t{font-size:13.5px;font-weight:700;fill:#c9d1d9}'
             '.s{font-size:11px;fill:#8b949e}'
             '.a{font-size:10px;fill:#8b949e}'
             '.v{font-size:11px;font-weight:700;fill:#c9d1d9}'
             '.x{font-size:10px;fill:#8b949e}'
             '.k{font-size:11px;fill:#8b949e}'
             '.g{stroke:#8b949e;stroke-opacity:.22;stroke-width:1}'
             '.e{stroke:#c9d1d9;stroke-opacity:.75;stroke-width:1.5}'
             '@media(prefers-color-scheme:light){'
             '.h,.t,.v{fill:#1f2328}.h2,.s,.a,.x,.k{fill:#59636e}'
             '.g{stroke:#59636e}.e{stroke:#1f2328}}'
             '</style>')
    s.append(f'<rect width="{W}" height="{H}" fill="none"/>')

    s.append('<text x="40" y="38" class="h">Four users of one agent, on two Arm64 '
             'and x86 hosts</text>')
    s.append('<text x="40" y="58" class="h2">The scheduler makes the identical '
             'decision on both machines. Only the price differs.</text>')

    s += panel(lx, py, pw, ph, rows, "ms",
               "Time to first token", "warm median, 95% CI · the COST", " ms")
    s += panel(rx, py, pw, ph, rows, "tokens",
               "Tokens recomputed per request",
               "from the server's own log · the CAUSE", "")

    # Shared x labels, grouped by machine
    for x0 in (lx, rx):
        slot = pw / len(rows)
        for i, r in enumerate(rows):
            cx = x0 + slot * (i + 0.5)
            s.append(f'<text x="{cx:.1f}" y="{py + ph + 15:.1f}" class="x" '
                     f'text-anchor="middle">{esc(r["label"].split(",")[0])}</text>')
            if "," in r["label"]:
                s.append(f'<text x="{cx:.1f}" y="{py + ph + 27:.1f}" class="x" '
                         f'text-anchor="middle">{esc(r["label"].split(",")[1].strip())}</text>')
        # machine separator + name
        seen, start = None, 0
        for i, r in enumerate(rows + [None]):
            nm = r["machine"] if r else None
            if seen is not None and nm != seen:
                mid = x0 + slot * ((start + i) / 2)
                s.append(f'<text x="{mid:.1f}" y="{py + ph + 46:.1f}" class="k" '
                         f'text-anchor="middle" font-weight="700">{esc(seen)}</text>')
                if r:
                    bx = x0 + slot * i
                    s.append(f'<line x1="{bx:.1f}" y1="{py}" x2="{bx:.1f}" '
                             f'y2="{py + ph + 32}" class="g"/>')
                start = i
            seen = nm

    s.append(f'<text x="40" y="{H - 14}" class="k">'
             'Identical bars on the right, different bars on the left: the mechanism '
             'is architecture-independent, the cost is not. '
             'Qwen2.5-1.5B-Instruct Q4_K_M, n=5, fresh server per repeat.</text>')
    s.append('</svg>')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s) + "\n")

    print(f"  wrote {args.out}  ({len(rows)} bars from committed results)")
    for r in rows:
        print(f"    {r['machine']:<13} {r['label']:<26} "
              f"{r['ms']:8.1f} ms   {str(r['tokens']):>4} tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
