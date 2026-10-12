#!/usr/bin/env bash
# Builds tools/kolibri/llama_logits against a patched llama.cpp.
#
# Stock llama.cpp cannot read a kolibri1 artifact - the architecture, the
# sigmoid_logit_add gating and the `kolibri1` pre-tokenizer only exist in the
# Hob-forge patch (https://huggingface.co/Hob-forge/Kolibri-1-GGUF, file
# kolibri1-llama.cpp.patch, applied to llama.cpp 836d571). Point LLAMA_DIR at a
# tree built with that patch; CPU-only is enough and avoids the resident engine.
#
#   LLAMA_DIR=~/kolibri-xcheck/llama.cpp tools/kolibri/build_llama_logits.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
llama="${LLAMA_DIR:?set LLAMA_DIR to a llama.cpp tree built with the kolibri1 patch}"
for header in "$llama/include/llama.h"; do
  [[ -f "$header" ]] || { echo "missing $header" >&2; exit 1; }
done
lib="$llama/build/bin"
[[ -d "$lib" ]] || lib="$llama/build/lib"
[[ -f "$lib/libllama.so" ]] || { echo "no libllama.so under $llama/build - build it first" >&2; exit 1; }
mkdir -p "$root/artifacts/kolibri"
c++ -std=c++17 -O2 -o "$root/artifacts/kolibri/llama_logits" \
  "$root/tools/kolibri/llama_logits.cpp" \
  -I"$llama/include" -I"$llama/ggml/include" \
  -L"$lib" -Wl,-rpath,"$lib" -lllama -lggml -lggml-base
echo "$root/artifacts/kolibri/llama_logits"
