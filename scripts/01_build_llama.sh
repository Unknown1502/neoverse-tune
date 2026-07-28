#!/usr/bin/env bash
#
# 01_build_llama.sh — build llama.cpp twice: with and without KleidiAI.
#
# The A/B pair is the entire point. "KleidiAI is fast" is unfalsifiable without
# the same source, same compiler, same flags, differing only in
# -DGGML_CPU_KLEIDIAI. Anything else and the Skeptic will (correctly) reject it.
#
# BUILD_SHARED_LIBS=OFF is deliberate: a single static binary makes KleidiAI's
# kai_* microkernel symbols resolve cleanly under perf in 03_kernel_attrib.sh.
#
# Usage: scripts/01_build_llama.sh [--jobs N]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

JOBS="$(nproc 2>/dev/null || echo 2)"
[ "${1:-}" = "--jobs" ] && JOBS="${2:?--jobs needs a value}"

for tool in git cmake; do
  have "$tool" || die "missing '$tool' — install build deps first (see .github/workflows/crux.yml)"
done

mkdir -p "$VENDOR_DIR"
SRC="$VENDOR_DIR/llama.cpp"

if [ ! -d "$SRC/.git" ]; then
  log "cloning llama.cpp ($LLAMA_REF) into $SRC"
  git clone --filter=blob:none "$LLAMA_REPO" "$SRC"
  git -C "$SRC" checkout "$LLAMA_REF"
else
  log "reusing existing checkout at $SRC"
fi

SHA="$(git -C "$SRC" rev-parse HEAD)"
log "llama.cpp resolved to $SHA"

build_variant() {
  local name="$1" kleidi="$2"
  local dir="$SRC/build-$name"

  log "configuring '$name' (GGML_CPU_KLEIDIAI=$kleidi)"
  cmake -S "$SRC" -B "$dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_KLEIDIAI="$kleidi" \
    -DGGML_NATIVE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    > "$RESULTS_DIR/build_${name}_configure.log" 2>&1 \
    || { tail -n 40 "$RESULTS_DIR/build_${name}_configure.log" >&2
         die "cmake configure failed for '$name' (full log: $RESULTS_DIR/build_${name}_configure.log)"; }

  log "building '$name' with $JOBS jobs"
  cmake --build "$dir" --target llama-bench -j "$JOBS" \
    > "$RESULTS_DIR/build_${name}.log" 2>&1 \
    || { tail -n 40 "$RESULTS_DIR/build_${name}.log" >&2
         die "build failed for '$name' (full log: $RESULTS_DIR/build_${name}.log)"; }

  local bin
  bin="$(find "$dir" -name llama-bench -type f -perm -u+x 2>/dev/null | head -n1)"
  [ -n "$bin" ] || die "built '$name' but cannot locate the llama-bench binary under $dir"
  echo "$bin"
}

BIN_KLEIDI="$(build_variant kleidi ON)"
BIN_BASE="$(build_variant base OFF)"

# Confirm the KleidiAI build genuinely linked KleidiAI in. A build that silently
# ignored the flag would otherwise produce a fake "no difference" result.
KAI_SYMS=0
if have nm; then
  KAI_SYMS="$(nm -C "$BIN_KLEIDI" 2>/dev/null | grep -c ' kai_' || true)"
fi

cat > "$RESULTS_DIR/build.latest.json" <<EOF
{
  "schema": "specarm.build/1",
  "llama_cpp_sha": $(json_escape "$SHA"),
  "llama_cpp_ref": $(json_escape "$LLAMA_REF"),
  "bin_kleidi": $(json_escape "$BIN_KLEIDI"),
  "bin_base": $(json_escape "$BIN_BASE"),
  "kai_symbols_in_kleidi_build": ${KAI_SYMS:-0},
  "jobs": $JOBS
}
EOF

log "kleidi build: $BIN_KLEIDI"
log "base   build: $BIN_BASE"
log "kai_* symbols found in kleidi binary: ${KAI_SYMS:-0}"

if [ "${KAI_SYMS:-0}" -eq 0 ]; then
  warn "No kai_* symbols in the KleidiAI build."
  warn "Either this toolchain silently dropped KleidiAI, or symbols were stripped."
  warn "Do NOT report an A/B result from this build until that is explained."
fi
