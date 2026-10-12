#!/usr/bin/env bash
# Builds the Kolibri 1 reference logits dumper against an existing cmake tree.
#
# The reference library is EXCLUDE_FROM_ALL (it is a test oracle, never part of
# a shipped binary), so this links against the built archives directly instead
# of adding a production target. Build the tree first:
#   cmake -B build/gpu-test . && cmake --build build/gpu-test --target gufo_kolibri1_reference
# Output: artifacts/kolibri/reference_logits
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
tree="${1:-$root/build/gpu-test}"
for archive in \
  "$tree/src/models/kolibri1/libgufo_kolibri1_reference.a" \
  "$tree/src/models/kolibri1/libgufo_kolibri1.a" \
  "$tree/libgufo_core.a"; do
  [[ -f "$archive" ]] || { echo "missing $archive - build the tree first" >&2; exit 1; }
done
mkdir -p "$root/artifacts/kolibri"
c++ -std=c++20 -O2 -I"$root" \
  "$root/tools/kolibri/reference_logits.cpp" \
  "$tree/src/models/kolibri1/libgufo_kolibri1_reference.a" \
  "$tree/src/models/kolibri1/libgufo_kolibri1.a" \
  "$tree/libgufo_core.a" \
  -o "$root/artifacts/kolibri/reference_logits" -pthread
echo "$root/artifacts/kolibri/reference_logits"
