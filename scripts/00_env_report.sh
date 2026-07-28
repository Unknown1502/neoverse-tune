#!/usr/bin/env bash
#
# 00_env_report.sh — capture exactly which Arm core we are standing on.
#
# This is the gate for the whole project. SpecArm's thesis is that KleidiAI's
# i8mm (SMMLA) matmul microkernels sit idle during batch=1 decode. On a core
# with no i8mm at all (Neoverse N1 / AWS Graviton2) that experiment is not
# "negative", it is *meaningless* — there is no i8mm kernel to wake up. Such a
# host is only useful as a control group.
#
# Usage:
#   scripts/00_env_report.sh                 # report, always exit 0
#   scripts/00_env_report.sh --require-i8mm  # exit 3 if this core lacks i8mm
#
# Writes: results/env_<hostname>_<timestamp>.json  (and results/env.latest.json)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

REQUIRE_I8MM=0
[ "${1:-}" = "--require-i8mm" ] && REQUIRE_I8MM=1

require_aarch64

# ---------------------------------------------------------------- core identity
cpuinfo_field() {
  grep -m1 "^$1" /proc/cpuinfo 2>/dev/null | sed 's/.*: *//' || true
}

IMPLEMENTER="$(cpuinfo_field 'CPU implementer')"
PART="$(cpuinfo_field 'CPU part')"

# Arm Ltd (0x41) Neoverse part numbers. Anything unrecognised is reported as
# unknown with its raw ID rather than guessed at.
CORE_NAME="unknown"
if [ "$IMPLEMENTER" = "0x41" ]; then
  case "$PART" in
    0xd0c) CORE_NAME="Neoverse-N1" ;;
    0xd40) CORE_NAME="Neoverse-V1" ;;
    0xd49) CORE_NAME="Neoverse-N2" ;;
    0xd4f) CORE_NAME="Neoverse-V2" ;;
    *)     CORE_NAME="Arm-unknown($PART)" ;;
  esac
elif [ -n "$PART" ]; then
  CORE_NAME="impl${IMPLEMENTER}-part${PART}"
fi

# ---------------------------------------------------------------- ISA features
# These are the features that decide which KleidiAI microkernel can be selected.
FEATURES=(asimd asimddp i8mm bf16 sve sve2 sme sme2 fphp asimdhp)
feat_lines=()
for f in "${FEATURES[@]}"; do
  if cpu_has "$f"; then v=true; else v=false; fi
  feat_lines+=("    $(json_escape "$f"): $v")
done

HAS_I8MM=false;  cpu_has i8mm  && HAS_I8MM=true
HAS_SVE2=false;  cpu_has sve2  && HAS_SVE2=true
HAS_BF16=false;  cpu_has bf16  && HAS_BF16=true
HAS_DOTP=false;  cpu_has asimddp && HAS_DOTP=true

# SVE vector length in bits, if SVE is present. On Neoverse N2 this is 128 —
# the same width as NEON — so any SVE2 win must come from predication, not
# width. Recording it stops us overclaiming later.
SVE_BITS="null"
if [ "$HAS_SVE2" = true ] || cpu_has sve; then
  if have python3; then
    v="$(python3 - <<'PY' 2>/dev/null || true
import ctypes, os
try:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SVE_GET_VL = 51
    r = libc.prctl(PR_SVE_GET_VL, 0, 0, 0, 0)
    print((r & 0xffff) * 8 if r >= 0 else "")
except Exception:
    print("")
PY
)"
    [ -n "$v" ] && SVE_BITS="$v"
  fi
fi

# ---------------------------------------------------------------- environment
NPROC="$(nproc 2>/dev/null || echo 0)"
KERNEL="$(uname -sr)"
DISTRO="$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || echo unknown)"
HOSTN="$(hostname 2>/dev/null || echo unknown)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CC_VER="$( (cc --version 2>/dev/null || gcc --version 2>/dev/null) | head -n1 || echo unknown)"
CMAKE_VER="$(cmake --version 2>/dev/null | head -n1 || echo 'not installed')"

# perf availability decides whether kernel-symbol attribution (03) can run.
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)"
PERF_OK=false
have perf && [ "$PARANOID" != unavailable ] && [ "$PARANOID" -le 2 ] 2>/dev/null && PERF_OK=true

# Whichever llama.cpp we actually built against — the real reproducibility key.
LLAMA_SHA="null"
if [ -d "$VENDOR_DIR/llama.cpp/.git" ]; then
  LLAMA_SHA="$(json_escape "$(git -C "$VENDOR_DIR/llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)")"
fi

# Burstable instances (AWS t4g) silently throttle mid-benchmark and quietly
# corrupt results. Flag the shape so the Skeptic can reject runs from one.
VIRT="$(systemd-detect-virt 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------- crux verdict
if [ "$HAS_I8MM" = true ]; then
  CRUX_ROLE="subject"       # i8mm exists -> the wake-up experiment is meaningful
else
  CRUX_ROLE="control"       # no i8mm    -> useful only as a contrast row
fi

OUT="$RESULTS_DIR/env_${HOSTN}_${STAMP}.json"
{
  echo "{"
  echo "  \"schema\": \"specarm.env/1\","
  echo "  \"timestamp_utc\": $(json_escape "$TS"),"
  echo "  \"hostname\": $(json_escape "$HOSTN"),"
  echo "  \"core\": $(json_escape "$CORE_NAME"),"
  echo "  \"cpu_implementer\": $(json_escape "$IMPLEMENTER"),"
  echo "  \"cpu_part\": $(json_escape "$PART"),"
  echo "  \"nproc\": $NPROC,"
  echo "  \"sve_vector_bits\": $SVE_BITS,"
  echo "  \"virtualization\": $(json_escape "$VIRT"),"
  echo "  \"kernel\": $(json_escape "$KERNEL"),"
  echo "  \"distro\": $(json_escape "$DISTRO"),"
  echo "  \"compiler\": $(json_escape "$CC_VER"),"
  echo "  \"cmake\": $(json_escape "$CMAKE_VER"),"
  echo "  \"perf_event_paranoid\": $(json_escape "$PARANOID"),"
  echo "  \"perf_usable\": $PERF_OK,"
  echo "  \"llama_cpp_sha\": $LLAMA_SHA,"
  echo "  \"crux_role\": $(json_escape "$CRUX_ROLE"),"
  echo "  \"features\": {"
  # Join with commas: every line but the last gets a trailing comma.
  for i in "${!feat_lines[@]}"; do
    if [ "$i" -lt $(( ${#feat_lines[@]} - 1 )) ]; then
      printf '%s,\n' "${feat_lines[$i]}"
    else
      printf '%s\n'  "${feat_lines[$i]}"
    fi
  done
  echo "  }"
  echo "}"
} > "$OUT"

cp "$OUT" "$RESULTS_DIR/env.latest.json"

# ---------------------------------------------------------------- human summary
echo
echo "  core ................ $CORE_NAME  (${NPROC} threads, virt=$VIRT)"
echo "  i8mm (SMMLA) ........ $HAS_I8MM"
echo "  bf16 ................ $HAS_BF16"
echo "  sve2 ................ $HAS_SVE2  (vector bits: $SVE_BITS)"
echo "  dotprod (SDOT) ...... $HAS_DOTP"
echo "  perf usable ......... $PERF_OK  (paranoid=$PARANOID)"
echo "  report .............. $OUT"
echo

if [ "$CRUX_ROLE" = "subject" ]; then
  log "CRUX ROLE: SUBJECT — this core has i8mm. The wake-up experiment is valid here."
else
  warn "CRUX ROLE: CONTROL — no i8mm on this core."
  warn "There is no SMMLA kernel to wake up, so a null result here proves nothing"
  warn "about the thesis. Use this host only as the N1-class contrast row."
  if [ "$REQUIRE_I8MM" = 1 ]; then
    die "--require-i8mm was set and this core lacks i8mm. Refusing to produce misleading data."
  fi
fi

if [ "$VIRT" != "none" ] && [ "$VIRT" != "unknown" ]; then
  warn "Virtualized host ($VIRT): PMU counters are typically restricted, and"
  warn "burstable instance types throttle mid-run. Treat wall-clock as primary."
fi
