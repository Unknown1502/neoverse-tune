#!/usr/bin/env python3
"""
test_gen_variants.py — validate the variant-table generator.

The generator decides what the C++ probe links against. If it silently emits an
empty or malformed list, the probe fails with a linker error that says nothing
useful. So it is tested against synthetic checkouts, including the awkward cases:
a moved directory layout, interface headers that must be skipped, and a checkout
with no i8mm variants at all.

Run:  python3 tools/test_gen_variants.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gen_variants",
                                              os.path.join(HERE, "gen_variants.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

FAILURES: list[str] = []

REAL_NAMES = [
    "clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod",
    "clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod",
    "clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm",
    "clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm",
]


def make_checkout(root: str, names: list[str], nested: bool = True,
                  extra: list[str] | None = None) -> None:
    d = os.path.join(root, "kai", "ukernels", "matmul", gen.FAMILY) if nested else root
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, f"kai_matmul_{n}.h"), "w").close()
    for e in extra or []:
        open(os.path.join(d, e), "w").close()


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  [' + detail + ']') if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("\nVariant generator — synthetic KleidiAI checkouts\n")

    # 1. Standard layout.
    tmp = tempfile.mkdtemp(prefix="kai-std-")
    try:
        make_checkout(tmp, REAL_NAMES)
        found = gen.discover(tmp)
        check("standard layout finds all variants", len(found) == 4, f"got {len(found)}")
        isas = {isa for _n, isa in found}
        check("ISA classification", isas == {"dotprod", "i8mm"}, str(sorted(isas)))

        out = os.path.join(tmp, "out", "kai_variants.inc")
        rc = gen.main.__globals__["discover"] and 0   # keep linters quiet
        sys.argv = ["gen_variants.py", tmp, out]
        rc = gen.main()
        check("generator exits 0", rc == 0, f"rc={rc}")

        text = open(out).read()
        check("emits includes", text.count("#include") == 4,
              f"{text.count('#include')} includes")
        check("emits macro list", "#define KAI_VARIANT_LIST" in text)
        check("every variant in list", all(f"KAI_V({n}," in text for n in REAL_NAMES))

        # Line continuations must be correct or the macro silently truncates.
        body = text.split("#define KAI_VARIANT_LIST")[1].strip().splitlines()
        kai_v = [l for l in body if "KAI_V(" in l]
        conts = [l.rstrip().endswith("\\") for l in kai_v]
        check("continuations on all but last",
              all(conts[:-1]) and not conts[-1],
              f"{sum(conts)} of {len(conts)} continue")
        check("define line continues", text.split("\n")[
            [i for i, l in enumerate(text.split("\n"))
             if l.startswith("#define KAI_VARIANT_LIST")][0]].rstrip().endswith("\\"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 2. Layout moved: generator must still find headers by walking.
    tmp = tempfile.mkdtemp(prefix="kai-flat-")
    try:
        make_checkout(tmp, REAL_NAMES, nested=False)
        check("flat/moved layout still discovered", len(gen.discover(tmp)) == 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 3. Interface headers are not runnable kernels and must be skipped.
    tmp = tempfile.mkdtemp(prefix="kai-iface-")
    try:
        make_checkout(tmp, REAL_NAMES, extra=[
            "kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h",
            "kai_lhs_quant_pack_qai8dxp_f32.h",
            "README.md",
        ])
        found = gen.discover(tmp)
        check("interface/pack headers skipped", len(found) == 4, f"got {len(found)}")
        check("no interface entry", not any("interface" in n for n, _ in found))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 4. No i8mm anywhere: must still succeed but warn.
    tmp = tempfile.mkdtemp(prefix="kai-noi8mm-")
    try:
        make_checkout(tmp, [n for n in REAL_NAMES if "i8mm" not in n])
        found = gen.discover(tmp)
        check("dotprod-only checkout still works", len(found) == 2, f"got {len(found)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 5. Empty checkout must fail loudly, not emit an empty macro.
    tmp = tempfile.mkdtemp(prefix="kai-empty-")
    try:
        sys.argv = ["gen_variants.py", tmp, os.path.join(tmp, "out.inc")]
        check("empty checkout exits non-zero", gen.main() != 0)
        check("no output file written on failure",
              not os.path.exists(os.path.join(tmp, "out.inc")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("Variant generator checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
