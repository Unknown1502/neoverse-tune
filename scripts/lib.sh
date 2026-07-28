#!/usr/bin/env bash
# Shared helpers for SpecArm scripts. Source this, don't execute it.

set -euo pipefail

SPECARM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPECARM_ROOT

RESULTS_DIR="${SPECARM_RESULTS:-$SPECARM_ROOT/results}"
VENDOR_DIR="${SPECARM_VENDOR:-$SPECARM_ROOT/vendor}"
MODELS_DIR="${SPECARM_MODELS:-$SPECARM_ROOT/models}"
export RESULTS_DIR VENDOR_DIR MODELS_DIR

# Upstream llama.cpp. We do NOT hard-pin a tag we cannot verify; instead we
# record the exact resolved SHA into the environment report, which is what
# actually makes a run reproducible.
LLAMA_REPO="${SPECARM_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${SPECARM_LLAMA_REF:-master}"
export LLAMA_REPO LLAMA_REF

log()  { printf '\033[1;36m[specarm]\033[0m %s\n'       "$*" >&2; }
warn() { printf '\033[1;33m[specarm:warn]\033[0m %s\n'  "$*" >&2; }
die()  { printf '\033[1;31m[specarm:error]\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Emit a JSON string literal with the characters JSON actually forbids escaped.
json_escape() {
  local s=${1-}
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\t'/\\t}
  s=${s//$'\r'/\\r}
  s=${s//$'\n'/\\n}
  printf '"%s"' "$s"
}

# Does this CPU advertise a given feature in /proc/cpuinfo's Features line?
# Word-boundary matched: plain `i8mm` must not match `svei8mm`.
cpu_has() {
  local feat="$1"
  grep -m1 '^Features' /proc/cpuinfo 2>/dev/null \
    | grep -qw -- "$feat"
}

require_aarch64() {
  local m
  m="$(uname -m)"
  [ "$m" = "aarch64" ] || [ "$m" = "arm64" ] \
    || die "SpecArm measures Arm kernel selection; this host is '$m'. Run on an arm64 Linux target."
}

mkdir -p "$RESULTS_DIR"
