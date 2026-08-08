#!/usr/bin/env bash
#
# 00_env_report.sh — identify the Arm core this run is standing on.
#
# WHY THE CORE IDENTITY MATTERS TO THIS PROJECT
# ---------------------------------------------
# The finding in this repository is about slot *selection*: llama-server routes
# a request to a slot by longest-common-prefix similarity, and against the 0.10
# default every tenant of an agent matches every slot. A mis-routed request pays
# to re-prefill ~220 tokens it had already processed.
#
# That mechanism is scheduler behaviour and is architecture-independent. What is
# NOT architecture-independent is what those 220 tokens *cost*. Prefill is
# matrix multiplication, and which int8 matmul path llama.cpp and KleidiAI can
# select depends on the core: Neoverse N1 offers dotprod (SDOT) only, while N2
# and V2 add i8mm (SMMLA). A report that quotes a millisecond figure without
# naming the core it came from is not reproducible.
#
# Whether that difference is large enough to change the *optimal threshold* per
# core is an open question. This script produces the input to that comparison.
# It does not answer it, and nothing here should be read as claiming an answer.
#
# HISTORICAL NOTE
# ---------------
# An earlier hypothesis held that KleidiAI's i8mm microkernels sat idle during
# batch-1 decode, and this script classified hosts as "subject" or "control" for
# that experiment. The hypothesis was wrong — KleidiAI ships an mr=1 dotprod
# kernel precisely for batch-1 decode — and the classification died with it. See
# docs/methodology.md §1. The detection below is unchanged and was never the
# faulty part.
#
# Usage:
#   scripts/00_env_report.sh                  # report, exit 0
#   scripts/00_env_report.sh --require i8mm   # exit 3 unless the core has i8mm
#
# Writes: results/env_<hostname>_<timestamp>.json  (and results/env.latest.json)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

REQUIRE_FEAT=""
case "${1:-}" in
  "")           ;;
  --require)    REQUIRE_FEAT="${2:?--require needs a feature name, e.g. --require i8mm}" ;;
  --require=*)  REQUIRE_FEAT="${1#--require=}" ;;
  *)            die "unknown argument '$1' (expected: --require <feature>)" ;;
esac

require_aarch64

# ---------------------------------------------------------------- core identity
cpuinfo_field() {
  grep -m1 "^$1" /proc/cpuinfo 2>/dev/null | sed 's/.*: *//' || true
}

IMPLEMENTER="$(cpuinfo_field 'CPU implementer')"
PART="$(cpuinfo_field 'CPU part')"

# Arm Ltd (0x41) Neoverse part numbers. Anything unrecognised is reported as
# unknown with its raw ID rather than guessed at — a wrong core name would
# silently mislabel every number in the run.
CORE_NAME="unknown"
case "$IMPLEMENTER" in
  0x41)
    case "$PART" in
      0xd0c) CORE_NAME="Neoverse-N1" ;;
      0xd40) CORE_NAME="Neoverse-V1" ;;
      0xd49) CORE_NAME="Neoverse-N2" ;;
      0xd4f) CORE_NAME="Neoverse-V2" ;;
      *)     CORE_NAME="Arm-unknown($PART)" ;;
    esac
    ;;
  *)
    [ -n "$PART" ] && CORE_NAME="impl${IMPLEMENTER}-part${PART}"
    ;;
esac

# ---------------------------------------------------------------- ISA features
# These decide which GGML / KleidiAI microkernel can be selected, and therefore
# how fast a re-prefill actually is on this host.
FEATURES=(asimd asimddp i8mm bf16 sve sve2 sme sme2 fphp asimdhp)
feat_lines=()
for f in "${FEATURES[@]}"; do
  if cpu_has "$f"; then v=true; else v=false; fi
  feat_lines+=("    $(json_escape "$f"): $v")
done

HAS_I8MM=false;  cpu_has i8mm    && HAS_I8MM=true
HAS_SVE2=false;  cpu_has sve2    && HAS_SVE2=true
HAS_BF16=false;  cpu_has bf16    && HAS_BF16=true
HAS_DOTP=false;  cpu_has asimddp && HAS_DOTP=true

# The best int8 matmul path this core can offer. This is a statement about the
# HARDWARE, not about which kernel llama.cpp actually selected for a given
# tensor — that depends on the build, the quantization format and the batch
# shape. Reported so a per-core prefill comparison has a factual axis.
if   [ "$HAS_I8MM" = true ]; then INT8_PATH="i8mm"
elif [ "$HAS_DOTP" = true ]; then INT8_PATH="dotprod"
else                             INT8_PATH="none"
fi

# SVE vector length in bits, if SVE is present. On Neoverse N2 this is 128 —
# the same width as NEON — so any SVE2 win must come from predication, not
# width. Recording it stops us overclaiming later.
SVE_BITS="null"
if [ "$HAS_SVE2" = true ] || cpu_has sve; then
  if have python3; then
    v="$(python3 - <<'PY' 2>/dev/null || true
import ctypes
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

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)"
PERF_OK=false
have perf && [ "$PARANOID" != unavailable ] && [ "$PARANOID" -le 2 ] 2>/dev/null && PERF_OK=true

# Whichever llama.cpp we actually built against — the real reproducibility key.
LLAMA_SHA="null"
if [ -d "$VENDOR_DIR/llama.cpp/.git" ]; then
  LLAMA_SHA="$(json_escape "$(git -C "$VENDOR_DIR/llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)")"
fi

# Burstable instances (AWS t4g) silently throttle mid-benchmark and quietly
# corrupt results. Flag the shape so a noisy run can be explained rather than
# rationalised. MAX_RSD_PCT in the adjudicator is the backstop.
VIRT="$(systemd-detect-virt 2>/dev/null || echo unknown)"

OUT="$RESULTS_DIR/env_${HOSTN}_${STAMP}.json"
{
  echo "{"
  echo "  \"schema\": \"specarm.env/2\","
  echo "  \"timestamp_utc\": $(json_escape "$TS"),"
  echo "  \"hostname\": $(json_escape "$HOSTN"),"
  echo "  \"core\": $(json_escape "$CORE_NAME"),"
  echo "  \"cpu_implementer\": $(json_escape "$IMPLEMENTER"),"
  echo "  \"cpu_part\": $(json_escape "$PART"),"
  echo "  \"nproc\": $NPROC,"
  echo "  \"int8_matmul_path\": $(json_escape "$INT8_PATH"),"
  echo "  \"sve_vector_bits\": $SVE_BITS,"
  echo "  \"virtualization\": $(json_escape "$VIRT"),"
  echo "  \"kernel\": $(json_escape "$KERNEL"),"
  echo "  \"distro\": $(json_escape "$DISTRO"),"
  echo "  \"compiler\": $(json_escape "$CC_VER"),"
  echo "  \"cmake\": $(json_escape "$CMAKE_VER"),"
  echo "  \"perf_event_paranoid\": $(json_escape "$PARANOID"),"
  echo "  \"perf_usable\": $PERF_OK,"
  echo "  \"llama_cpp_sha\": $LLAMA_SHA,"
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
echo "  int8 matmul path .... $INT8_PATH"
echo "  i8mm (SMMLA) ........ $HAS_I8MM"
echo "  dotprod (SDOT) ...... $HAS_DOTP"
echo "  bf16 ................ $HAS_BF16"
echo "  sve2 ................ $HAS_SVE2  (vector bits: $SVE_BITS)"
echo "  perf usable ......... $PERF_OK  (paranoid=$PARANOID)"
echo "  report .............. $OUT"
echo

case "$INT8_PATH" in
  i8mm)
    log "This core can reach i8mm (SMMLA) kernels."
    log "Pair it with a dotprod-only core (Neoverse N1 / Graviton2) to compare"
    log "what a mis-routed slot costs on each. That comparison is not yet made."
    ;;
  dotprod)
    log "This core offers dotprod (SDOT) but NOT i8mm — this is the N1 class."
    log "Useful as the contrast host: same scheduler bug, different prefill cost."
    ;;
  none)
    warn "No int8 dot-product or matmul extension detected. Prefill will use a"
    warn "generic path, and timings here will not be comparable to a Neoverse host."
    ;;
esac

if [ "$VIRT" != "none" ] && [ "$VIRT" != "unknown" ]; then
  warn "Virtualized host ($VIRT): PMU counters are typically restricted, and"
  warn "burstable instance types throttle mid-run. Treat wall-clock as primary."
fi

if [ -n "$REQUIRE_FEAT" ]; then
  if cpu_has "$REQUIRE_FEAT"; then
    log "required feature '$REQUIRE_FEAT': present"
  else
    printf '\033[1;31m[specarm:error]\033[0m %s\n' \
      "--require $REQUIRE_FEAT: this core does not advertise it. Refusing to continue." >&2
    exit 3
  fi
fi
