#!/usr/bin/env bash
# Builds the qwen35moe parity driver against the configured cpu-test tree.
# Usage: tools/qwen35moe/build_parity.sh [build-dir]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${1:-$root/build/cpu-test}"
c++ -std=c++20 -O2 -fopenmp -I"$root" \
  "$root/tools/qwen35moe/parity.cpp" \
  "$build/src/models/qwen35moe/libgufo_qwen35moe_reference.a" \
  "$build/src/models/qwen35moe/libgufo_qwen35moe.a" \
  "$build/libgufo_core.a" \
  -licui18n -licuuc -licudata -lcurl -lssl -lcrypto -lz -lpng -ljpeg -lwebp \
  -o "$build/qwen35moe_parity"
echo "built $build/qwen35moe_parity"
