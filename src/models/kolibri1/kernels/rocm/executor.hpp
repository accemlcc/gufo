#ifndef GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/kolibri1/kernels/rocm/device_model.hpp"

namespace gufo::models::kolibri1::rocm {

/// Rows of one projection launch. A grouped launch reads the weights of every
/// expert it touches, so it wants as many rows as it can get; a dense
/// projection has no such trade-off and 16 rows is the widest shape the
/// batch-invariant GEMV is written for. Prefill therefore walks its rows
/// through the projections in steps of this size, which costs a re-read of the
/// projection weights per step and nothing else.
inline constexpr std::uint32_t kProjectRows = 16;

/// Rows a projection block needs before the tiled GEMM is the faster of the
/// two. Measured per shape on gfx1151 with the artifact's own widths
/// (`tools/kolibri/prefill_gemm_probe.cpp`): over the seven projections of one
/// layer, the tiled path is 0.76 of the GEMV at 16 rows, 1.46 at 32, 2.95 at
/// 64, 3.51 at 128 and 4.97 at 256. The narrow projections of a layer (the key,
/// value and shared gate/up, 512 rows out) are the ones that lose at 16 rows -
/// their tile is mostly empty - and they are also the reason the crossover is
/// where it is rather than one row above `kProjectRows`.
inline constexpr std::uint32_t kMatrixCoreRows = 32;

/// One session's attention state. The ten layers without positional encoding
/// keep every position of the context; the sliding layers keep a ring, so a
/// long context costs the ten full caches and nothing else.
struct KvCache {
  ~KvCache();
  void* allocation{nullptr};
  std::size_t bytes{0};
  std::uint32_t max_context{0};
  /// Slots of the sliding layers' rings.
  std::uint32_t ring{0};
  /// Per layer, binary16, one row per position holding every key or value head.
  std::vector<std::uint16_t*> k;
  std::vector<std::uint16_t*> v;
};

/// Runs the Kolibri 1 graph on device-resident weights. Scratch is shared by
/// every session of one executor and calls are serialized on one stream.
class Executor {
public:
  /// `max_rows` bounds one forward (the prefill chunk); `max_logit_rows`
  /// bounds how many rows of one forward ask for logits.
  Executor(const DeviceModel& model, std::uint32_t max_rows,
           std::uint32_t max_logit_rows, std::uint32_t max_context);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Cache bytes for one session: the full-context layers at `max_context`
  /// positions, the sliding ones at `RingSlots(...)`.
  [[nodiscard]] static std::size_t CacheBytes(const Config& config,
                                              std::uint32_t max_context,
                                              std::uint32_t ring) noexcept;
  /// Ring slots a sliding layer needs for a forward of `max_rows`: its own
  /// window plus the rows written before any of them is read.
  [[nodiscard]] static std::uint32_t RingSlots(const Config& config,
                                               std::uint32_t max_rows) noexcept;

  [[nodiscard]] std::unique_ptr<KvCache> CreateCache(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;

  /// Evaluates `tokens` at positions [first_position, +tokens.size()) and
  /// writes the logits of `logit_rows` (indices into `tokens`, in order) into
  /// the device logits buffer; read them back with CopyLogits. Every earlier
  /// position must already be in `cache`.
  void Forward(KvCache& cache, std::span<const std::int32_t> tokens,
               std::uint32_t first_position,
               std::span<const std::uint32_t> logit_rows);

  /// The logits of the last Forward, [rows][vocab].
  void CopyLogits(std::size_t rows, std::vector<float>* out) const;

  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::uint32_t ring() const noexcept { return ring_; }
  [[nodiscard]] const DeviceModel& model() const noexcept { return model_; }

private:
  /// y[rows][w.rows] = x[rows][w.cols] . W^T. Up to `kProjectRows` this is the
  /// batch-invariant GEMV with its prebuilt single-group table; a wider block
  /// goes to the shared tiled GEMM, which reads the weights once for all of it.
  /// See `MatrixCoreProjection` in the definition for what decides.
  void Project(const DeviceTensor& w, const float* x, std::uint32_t rows,
               float* y);
  /// The routed experts of one layer over the rows of `h`, writing
  /// count * experts_used rows of expert output into `routed` and their routing
  /// weights into `weights`.
  void Experts(const DeviceLayer& l, const float* h, std::uint32_t count,
               float* routed, float* weights);
  /// The one ungated shared expert over `rows` rows of `h`.
  void SharedExpert(const DeviceLayer& l, const float* h, std::uint32_t rows,
                    float* out);

  const DeviceModel& model_;
  std::uint32_t max_rows_;
  std::uint32_t max_logit_rows_;
  std::uint32_t max_context_;
  std::uint32_t ring_;
  hipStream_t stream_{nullptr};

  /// Off unless GUFO_KOLIBRI_ROCM_PREFILL_STAGE_PROFILE is set, which is the
  /// same shape as the deepseek runtime's stage-profile switches. When on,
  /// `Forward` brackets the embedding, five stages of every layer, the output
  /// norm and the logits with events and prints their per-token cost to stderr.
  /// It exists because the cost of a stage inside a live forward is not
  /// something an isolated kernel probe can say - one such derivation was off
  /// by a factor of three (docs/models/kolibri-1/BENCHMARKS.md).
  bool stage_profile_{false};
  /// Four fixed markers plus five per layer, created only while profiling.
  std::vector<hipEvent_t> stage_events_;

  void* scratch_{nullptr};
  std::int32_t* tokens_{nullptr};
  float* x_{nullptr};
  float* h_{nullptr};
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* attn_{nullptr};
  float* o_{nullptr};
  float* dense_gate_{nullptr};
  float* dense_up_{nullptr};
  float* dense_act_{nullptr};
  float* shared_{nullptr};
  float* moe_gate_{nullptr};
  float* moe_up_{nullptr};
  float* moe_act_{nullptr};
  float* moe_out_{nullptr};
  float* moe_weights_{nullptr};
  std::int32_t* moe_ids_{nullptr};
  float* moe_logits_{nullptr};
  std::uint32_t* moe_counts_{nullptr};
  std::int32_t* moe_groups_{nullptr};
  /// Single-group tables for one to kProjectRows rows, built once.
  std::int32_t* dense_groups_{nullptr};
  float* logits_{nullptr};
  /// Q8_1 tiles of one projection's input rows, for the path that runs a wide
  /// projection on the matrix cores. Sized for the widest input any projection
  /// reads, so one buffer serves every layer of a forward.
  void* act_q8_{nullptr};
  std::size_t act_q8_bytes_{0};
  float* logit_x_{nullptr};
  float* partials_{nullptr};
};

}  // namespace gufo::models::kolibri1::rocm

#endif  // GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_EXECUTOR_HPP_
