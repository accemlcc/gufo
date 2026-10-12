#!/usr/bin/env bash
# Builds tools/kolibri/dequant_diff against gufo's core library and a llama.cpp
# tree (used only for ggml's reference dequantizers).
#
#   LLAMA_DIR=~/kolibri-xcheck/llama.cpp tools/kolibri/build_dequant_diff.sh [build-dir]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
llama="${LLAMA_DIR:?set LLAMA_DIR to a llama.cpp build to compare against}"
tree="${1:-$root/build/gpu-test}"
for needed in "$llama/include/llama.h" "$llama/ggml/src/ggml-quants.h" \
              "$tree/libgufo_core.a"; do
  [[ -e "$needed" ]] || { echo "missing $needed" >&2; exit 1; }
done
mkdir -p "$root/artifacts/kolibri"
c++ -std=c++20 -O2 -I"$root" -I"$llama/include" -I"$llama/ggml/include" \
  -I"$llama/ggml/src" \
  "$root/tools/kolibri/dequant_diff.cpp" "$tree/libgufo_core.a" \
  -L"$llama/build/bin" -Wl,-rpath,"$llama/build/bin" -lggml -lggml-base \
  -o "$root/artifacts/kolibri/dequant_diff" -pthread
echo "$root/artifacts/kolibri/dequant_diff"
