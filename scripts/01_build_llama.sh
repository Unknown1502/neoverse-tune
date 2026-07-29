#!/usr/bin/env bash
#
# 01_build_llama.sh — build llama-server from a recorded upstream revision.
#
# The finding this repo documents is about llama-server's slot-selection
# behaviour, so the exact revision matters: a future release may change the
# default threshold or the selection heuristic, at which point these numbers
# describe history rather than the present. The resolved SHA is recorded in the
# build manifest, which is what actually makes a run reproducible — more than a
# tag we cannot verify would.
#
# Usage: scripts/01_build_llama.sh [--jobs N]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

JOBS="$(nproc 2>/dev/null || echo 2)"
[ "${1:-}" = "--jobs" ] && JOBS="${2:?--jobs needs a value}"

for tool in git cmake; do
  have "$tool" || die "missing '$tool'"
done

mkdir -p "$VENDOR_DIR"
SRC="$VENDOR_DIR/llama.cpp"

if [ ! -d "$SRC/.git" ]; then
  log "cloning llama.cpp ($LLAMA_REF)"
  git clone --filter=blob:none "$LLAMA_REPO" "$SRC"
  git -C "$SRC" checkout "$LLAMA_REF"
else
  log "reusing checkout at $SRC"
fi

SHA="$(git -C "$SRC" rev-parse HEAD)"
log "llama.cpp resolved to $SHA"

BUILD="$SRC/build"
log "configuring"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  > "$RESULTS_DIR/build_configure.log" 2>&1 \
  || { tail -n 40 "$RESULTS_DIR/build_configure.log" >&2
       die "cmake configure failed (log: $RESULTS_DIR/build_configure.log)"; }

log "building llama-server with $JOBS jobs"
cmake --build "$BUILD" --target llama-server -j "$JOBS" \
  > "$RESULTS_DIR/build.log" 2>&1 \
  || { tail -n 40 "$RESULTS_DIR/build.log" >&2
       die "build failed (log: $RESULTS_DIR/build.log)"; }

SERVER="$(find "$BUILD" -name llama-server -type f -perm -u+x 2>/dev/null | head -n1)"
[ -n "$SERVER" ] || die "built, but cannot locate llama-server under $BUILD"

cat > "$RESULTS_DIR/build.latest.json" <<EOF
{
  "schema": "specarm.build/2",
  "llama_cpp_sha": $(json_escape "$SHA"),
  "llama_cpp_ref": $(json_escape "$LLAMA_REF"),
  "server": $(json_escape "$SERVER"),
  "jobs": $JOBS
}
EOF

log "server: $SERVER"
echo "$SERVER"
