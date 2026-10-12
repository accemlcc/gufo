#!/usr/bin/env bash
# Kolibri 1 graph parity: scalar reference vs patched llama.cpp.
#
# Runs both engines over the same tiny artifacts and the same token ids and
# compares full-vocabulary logits for every position. Covers the cases a kernel
# port must not get wrong: positions past the window (clipping), the global
# layer at first/middle/last positions, single-token prompts (where a rope bug
# is invisible) and multi-token prompts (where it is not).
#
#   LLAMA_DIR=~/kolibri-xcheck/llama.cpp tools/kolibri/run_parity.sh
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
max_abs="${MAX_ABS:-1e-2}"
max_kl="${MAX_KL:-1e-5}"
status=0
one() { # NAME PROMPTS GENERATOR-ARGS...
  local name="$1" prompts="$2"; shift 2
  python3 "$root/tools/kolibri/make_tiny_artifact.py" "/tmp/parity_$name.gguf" "$@" >/dev/null || {
    echo "$name: generator failed"; status=1; return; }
  printf '%s\n' "$prompts" > "/tmp/parity_$name.txt"
  "$root/artifacts/kolibri/reference_logits" "/tmp/parity_$name.gguf" \
    "/tmp/parity_ref_$name.json" "/tmp/parity_$name.txt" 2>/dev/null || {
    echo "$name: reference failed"; status=1; return; }
  "$root/artifacts/kolibri/llama_logits" "/tmp/parity_$name.gguf" \
    "/tmp/parity_llm_$name.json" "/tmp/parity_$name.txt" 2>/dev/null || {
    echo "$name: llama.cpp failed"; status=1; return; }
  echo "== $name [$*] '$prompts'"
  python3 "$root/tools/kolibri/compare_logits.py" --max-abs "$max_abs" --kl "$max_kl" \
    "/tmp/parity_ref_$name.json" "/tmp/parity_llm_$name.json" || status=1
}
one single            "5"                        --mode mixed
one two_tokens        "5 12"                     --mode mixed
one four_tokens       "5 12 33 7"                --mode mixed
one global_first      "5 12 33 7"                --mode mixed --layers 2 --global-layer 0
one global_middle     "5 12 33 7"                --mode mixed --layers 3 --global-layer 1
one all_sliding       "5 12 33 7"                --mode sliding
one window_clips      "5 12 33 7 9 21 4 8"       --mode sliding --window 3
one window_clips_mixed "5 12 33 7 9 21 4 8"      --mode mixed --layers 3 --window 2
exit $status
