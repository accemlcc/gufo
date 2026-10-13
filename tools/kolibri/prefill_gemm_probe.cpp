// Compares Kolibri 1's two projection paths at the shapes of the released
// artifact, without the artifact.
//
// The engine runs a dense projection through the batch-invariant GEMV in steps
// of 16 rows, reading a weight matrix once per step; above that width it can
// hand the block to the shared tiled GEMM instead (`MatrixCoreProjection` in
// src/models/kolibri1/kernels/rocm/executor.cpp). Which of the two wins is a
// question of weight traffic against the GEMV's own ceiling, and that is
// answerable per shape with random weights: both decode the same bytes, so
// the traffic and the arithmetic are the model's and only the values are not.
//
// It answers two things per shape and width:
//   - time, and with it the projection cost of a forward per token;
//   - how far the tiled path's int8 activation quantization moves a result,
//     measured against the GEMV on the same input.
// A 47 GiB artifact is not needed for either, and a few hundred MiB of random
// world fits beside a running server. What it does not answer is what the whole
// model does: experts, attention and output head are not in this measurement.
//
// Build it against an existing tree, the way tools/kolibri/build_throughput.sh
// builds its driver, with ENGINE_ENABLE_HIP defined by hand because the shared
// kernel headers hide behind it, and run it where the matrix cores are:
//   c++ -std=c++20 -O2 -D__HIP_PLATFORM_AMD__ -DENGINE_ENABLE_HIP=1 \
//       -I<root> -I/opt/rocm/include tools/kolibri/prefill_gemm_probe.cpp \
//       <tree>/src/models/kolibri1/libgufo_kolibri1.a <tree>/libgufo_core.a \
//       -L/opt/rocm/lib -lamdhip64 -lcrypto -licui18n -licuuc -pthread \
//       -o artifacts/kolibri/prefill_gemm_probe
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace {

namespace kr = gufo::models::kolibri1::rocm;
using gufo::core::GgmlType;

/// The seven projections of one layer: the attention's four and the three of the
/// ungated shared expert. The routed experts are grouped and stay on the GEMV;
/// the output head is asked for one row at a time, so neither is here.
struct Shape {
  const char* name;
  std::uint32_t m;
  std::uint32_t k;
  GgmlType type;
  kr::WeightFormat format;
};

constexpr Shape kShapes[] = {
    {"attn_q", 6144, 2560, GgmlType::kQ4_K, kr::WeightFormat::kQ4_K},
    {"attn_k", 512, 2560, GgmlType::kQ4_K, kr::WeightFormat::kQ4_K},
    {"attn_v", 512, 2560, GgmlType::kQ6_K, kr::WeightFormat::kQ6_K},
    {"attn_output", 2560, 6144, GgmlType::kQ6_K, kr::WeightFormat::kQ6_K},
    {"shared_gate", 512, 2560, GgmlType::kQ4_K, kr::WeightFormat::kQ4_K},
    {"shared_up", 512, 2560, GgmlType::kQ4_K, kr::WeightFormat::kQ4_K},
    {"shared_down", 2560, 512, GgmlType::kQ6_K, kr::WeightFormat::kQ6_K},
};

constexpr std::uint32_t kProjectRows = 16;
constexpr std::uint32_t kMaxRows = 256;
constexpr std::uint32_t kWidestK = 6144;
constexpr std::uint32_t kLayers = 50;

#define CHECK_HIP(call)                                                    \
  do {                                                                     \
    const hipError_t err_ = (call);                                        \
    if (err_ != hipSuccess) {                                              \
      std::fprintf(stderr, "FAIL %s: %s\n", #call,                         \
                   hipGetErrorString(err_));                               \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

/// The single-group tables of 1..kProjectRows rows, as the executor holds them:
/// one group, expert 0, and the rows as its slots.
[[nodiscard]] std::vector<std::int32_t> BuildTables() {
  std::vector<std::int32_t> tables(kProjectRows * kr::GroupTableInts(1), 0);
  for (std::uint32_t n = 1; n <= kProjectRows; ++n) {
    std::int32_t* t = tables.data() + (n - 1) * kr::GroupTableInts(1);
    t[0] = 1;
    t[1] = 0;
    t[2] = static_cast<std::int32_t>(n);
    for (std::uint32_t s = 0; s < n; ++s) {
      t[3 + s] = static_cast<std::int32_t>(s);
    }
  }
  return tables;
}

/// The engine's current path: `rows` through the GEMV in steps of 16.
void RunGemv(const kr::GroupedGemvArgs& base, std::uint32_t rows,
             const std::int32_t* tables, hipStream_t stream) {
  for (std::uint32_t offset = 0; offset < rows; offset += kProjectRows) {
    const std::uint32_t n = std::min(kProjectRows, rows - offset);
    kr::GroupedGemvArgs args = base;
    args.groups = tables + (n - 1) * kr::GroupTableInts(1);
    args.rows = n;
    args.x = base.x + static_cast<std::size_t>(offset) * base.k;
    args.y = base.y + static_cast<std::size_t>(offset) * base.m;
    if (!kr::GroupedGemv(args, stream)) {
      std::fprintf(stderr, "FAIL the GEMV refused this shape at %u rows\n", n);
      std::exit(1);
    }
  }
}

/// The path being tried: the rows as Q8_1 tiles, then one tiled launch.
void RunTiled(GgmlType type, const void* w, const float* x, void* tiles,
              float* y, std::uint32_t rows, std::uint32_t m, std::uint32_t k,
              hipStream_t stream) {
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x, tiles, rows, k, stream);
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(type, w, tiles, y, rows, m, k,
                                                stream);
}

[[nodiscard]] double Milliseconds(std::size_t iterations, hipStream_t stream,
                                  const std::function<void()>& body) {
  hipEvent_t first{}, last{};
  CHECK_HIP(hipEventCreate(&first));
  CHECK_HIP(hipEventCreate(&last));
  for (std::size_t i = 0; i < 3; ++i) {
    body();
  }
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipEventRecord(first, stream));
  for (std::size_t i = 0; i < iterations; ++i) {
    body();
  }
  CHECK_HIP(hipEventRecord(last, stream));
  CHECK_HIP(hipEventSynchronize(last));
  float ms = 0.0F;
  CHECK_HIP(hipEventElapsedTime(&ms, first, last));
  CHECK_HIP(hipEventDestroy(first));
  CHECK_HIP(hipEventDestroy(last));
  return ms / static_cast<double>(iterations);
}

/// Largest |a-b| over both rows, relative to the largest |a|. Random weights
/// make the values themselves meaningless; their agreement is not.
[[nodiscard]] double RelativeDifference(const std::vector<float>& a,
                                        const std::vector<float>& b) {
  double scale = 0.0;
  for (const float v : a) {
    scale = std::max(scale, static_cast<double>(std::fabs(v)));
  }
  if (scale == 0.0) {
    return 0.0;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(a[i] - b[i]) / scale);
  }
  return worst;
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::fprintf(stderr, "this probe measures the matrix cores of gfx1151\n");
    return 77;
  }

  std::vector<std::int32_t> host_tables = BuildTables();
  std::int32_t* tables = nullptr;
  CHECK_HIP(hipMalloc(&tables, host_tables.size() * sizeof(std::int32_t)));
  CHECK_HIP(hipMemcpy(tables, host_tables.data(),
                      host_tables.size() * sizeof(std::int32_t),
                      hipMemcpyHostToDevice));

  std::mt19937 rng(20261013);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
  std::vector<float> host_x(static_cast<std::size_t>(kMaxRows) * kWidestK);
  for (float& v : host_x) {
    v = unit(rng);
  }
  float* x = nullptr;
  CHECK_HIP(hipMalloc(&x, host_x.size() * sizeof(float)));
  CHECK_HIP(hipMemcpy(x, host_x.data(), host_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const std::size_t row_bytes =
      static_cast<std::size_t>(kMaxRows) * kWidestK * sizeof(float);
  float* gemv_y = nullptr;
  float* tiled_y = nullptr;
  CHECK_HIP(hipMalloc(&gemv_y, row_bytes));
  CHECK_HIP(hipMalloc(&tiled_y, row_bytes));

  const std::uint32_t widths[] = {16, 32, 64, 128, 256};
  std::vector<double> layer_gemv(5, 0.0), layer_tiled(5, 0.0);

  for (const Shape& s : kShapes) {
    const std::size_t w_bytes = gufo::quant::EncodedSizeBytes(
        s.type, static_cast<std::size_t>(s.m) * s.k);
    if (w_bytes == 0) {
      std::fprintf(stderr, "FAIL %s is not a quantized shape\n", s.name);
      return 1;
    }
    std::vector<std::uint8_t> host_w(w_bytes);
    for (std::uint8_t& byte : host_w) {
      byte = static_cast<std::uint8_t>(rng() & 0xFFU);
    }
    void* w = nullptr;
    CHECK_HIP(hipMalloc(&w, w_bytes));
    CHECK_HIP(hipMemcpy(w, host_w.data(), w_bytes, hipMemcpyHostToDevice));

    void* tiles = nullptr;
    CHECK_HIP(hipMalloc(&tiles,
                       gufo::hip::QuantizedActivationBytes(kMaxRows, s.k)));

    std::printf("%s  %u x %u  %.2f MiB\n", s.name, s.m, s.k,
                w_bytes / (1024.0 * 1024.0));
    for (std::size_t i = 0; i < 5; ++i) {
      const std::uint32_t rows = widths[i];
      const std::uint32_t steps = (rows + kProjectRows - 1) / kProjectRows;
      const kr::GroupedGemvArgs base{s.format, w, tables, 1, x, gemv_y,
                                     1,        1, s.m,    s.k};
      const double gemv_ms =
          Milliseconds(20, 0, [&] { RunGemv(base, rows, tables, 0); });
      const double tiled_ms = Milliseconds(20, 0, [&] {
        RunTiled(s.type, w, x, tiles, tiled_y, rows, s.m, s.k, 0);
      });

      // One number per shape, where both paths have a full block to work with.
      double rel = 0.0;
      if (rows == 128) {
        std::vector<float> ya(static_cast<std::size_t>(rows) * s.m),
            yb(static_cast<std::size_t>(rows) * s.m);
        CHECK_HIP(hipMemcpy(ya.data(), gemv_y, ya.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(yb.data(), tiled_y, yb.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        rel = RelativeDifference(ya, yb);
      }

      // The GEMV reads the whole matrix per step of 16 rows; the tiled path
      // reads it once for the block, plus the activation twice.
      const double gemv_gbs = (w_bytes * steps) / (gemv_ms * 1e6);
      const double tiled_gbs =
          (w_bytes + 2.0 * gufo::hip::QuantizedActivationBytes(rows, s.k)) /
          (tiled_ms * 1e6);
      std::printf(
          "  %4u rows  gemv %7.3f ms (%6.0f GB/s)  tiled %7.3f ms (%6.0f GB/s)"
          "  %5.2fx  int8 dev %.2e\n",
          rows, gemv_ms, gemv_gbs, tiled_ms, tiled_gbs, gemv_ms / tiled_ms, rel);
      layer_gemv[i] += gemv_ms;
      layer_tiled[i] += tiled_ms;
    }
    CHECK_HIP(hipFree(tiles));
    CHECK_HIP(hipFree(w));
  }

  std::printf("\nseven projections of one layer, and the forward they imply "
              "(%u layers, nothing else counted):\n",
              kLayers);
  for (std::size_t i = 0; i < 5; ++i) {
    const double rows = widths[i];
    const double token_gemv = layer_gemv[i] / rows * kLayers;
    const double token_tiled = layer_tiled[i] / rows * kLayers;
    std::printf("  %4u rows  gemv %7.2f ms/layer -> %6.0f tok/s   tiled "
                "%7.2f ms/layer -> %6.0f tok/s   %5.2fx\n",
                widths[i], layer_gemv[i], 1000.0 / token_gemv, layer_tiled[i],
                1000.0 / token_tiled, token_gemv / token_tiled);
  }
  CHECK_HIP(hipFree(gemv_y));
  CHECK_HIP(hipFree(tiled_y));
  CHECK_HIP(hipFree(x));
  CHECK_HIP(hipFree(tables));
  return 0;
}
