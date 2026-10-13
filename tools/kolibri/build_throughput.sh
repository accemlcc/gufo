#!/usr/bin/env bash
# Builds the Kolibri 1 throughput driver against an existing cmake tree.
#
# Same shape as build_reference_logits.sh: the engine's own archive is enough,
# so no production target is added for a measurement tool. Build the tree first:
#   cmake -B build/gpu-test -G Ninja -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=ON
#   cmake --build build/gpu-test --target gufo_kolibri1
# Output: artifacts/kolibri/throughput
#
# Run it where the render node is, on the host that also sees the artifact:
#   LD_LIBRARY_PATH=/opt/rocm/lib artifacts/kolibri/throughput <model.gguf>
#
# __HIP_PLATFORM_AMD__ is implied inside a HIP translation unit; this is a host
# one that reads the runtime's headers and links its library, so it says so.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
tree="${1:-$root/build/gpu-test}"
rocm="${ROCM_PATH:-/opt/rocm}"
for archive in \
  "$tree/src/models/kolibri1/libgufo_kolibri1.a" \
  "$tree/libgufo_core.a"; do
  [[ -f "$archive" ]] || { echo "missing $archive - build the tree first" >&2; exit 1; }
done
mkdir -p "$root/artifacts/kolibri"
c++ -std=c++20 -O2 -D__HIP_PLATFORM_AMD__ -I"$root" -I"$rocm/include" \
  "$root/tools/kolibri/throughput.cpp" \
  "$tree/src/models/kolibri1/libgufo_kolibri1.a" \
  "$tree/libgufo_core.a" \
  -L"$rocm/lib" -lamdhip64 -lcrypto -licui18n -licuuc -pthread \
  -o "$root/artifacts/kolibri/throughput"
echo "$root/artifacts/kolibri/throughput"
