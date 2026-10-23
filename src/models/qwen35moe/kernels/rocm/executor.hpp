#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"

namespace gufo::models::qwen35moe::rocm {

/// Rows of one forward (the prefill chunk). Projections below this width run
/// the batch-invariant grouped GEMV; the MoE table covers the whole forward.
inline constexpr std::uint32_t kProjectRows = 16;

/// One session's attention state: every full-attention layer keeps every
/// position of the context in F32. Gated DeltaNet layers keep their own
/// recurrent state in SsmState below; this struct is the K/V part only.
struct KvCache {
  ~KvCache();
  void* allocation{nullptr};
  std::size_t bytes{0};
  std::uint32_t max_context{0};
  /// Per full-attention layer in trunk order, then K and V: each
  /// [kv_heads][max_context][head_dim] F32.
  std::vector<float*> k;
  std::vector<float*> v;
};

/// One session's Gated DeltaNet state: per linear layer the depthwise-conv
/// history and the recurrent matrix, both F32, sized by the model dims.
struct SsmState {
  ~SsmState();
  void* allocation{nullptr};
  std::size_t bytes{0};
  /// Per linear layer in trunk order: [qkv_dim][conv_kernel] history.
  std::vector<float*> conv;
  /// Per linear layer: [value_heads][state_size][value_head_dim] matrix.
  std::vector<float*> recurrent;
};

/// Runs the Qwen3.5-MoE graph on device-resident weights. Scratch is shared
/// by every session of one executor and calls are serialized on one stream.
class Executor {
public:
  /// `max_rows` bounds one forward (the prefill chunk), `max_logit_rows`
  /// bounds how many rows of one forward ask for logits.
  Executor(const DeviceModel& model, std::uint32_t max_rows,
           std::uint32_t max_logit_rows, std::uint32_t max_context);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Cache bytes for one session: the full layers' K/V at `max_context`
  /// positions plus the SSM conv history and recurrent state.
  [[nodiscard]] static std::size_t CacheBytes(
      const Config& config, std::uint32_t max_context) noexcept;

  [[nodiscard]] std::unique_ptr<KvCache> CreateCache(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;
  [[nodiscard]] std::unique_ptr<SsmState> CreateSsmState(
      std::string* error_msg = nullptr) const;

  /// Evaluates `tokens` at positions [first_position, +tokens.size()) and
  /// writes the logits of `logit_rows` (indices into `tokens`, in order) into
  /// the device logits buffer; read them back with CopyLogits. Every earlier
  /// position must already be in `cache` and `ssm`.
  void Forward(KvCache& cache, SsmState& ssm,
               std::span<const std::int32_t> tokens,
               std::uint32_t first_position,
               std::span<const std::uint32_t> logit_rows);

  /// The logits of the last Forward, [rows][vocab].
  void CopyLogits(std::size_t rows, std::vector<float>* out) const;

  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] const DeviceModel& model() const noexcept { return model_; }

private:
  void ForwardLayer(KvCache& cache, SsmState& ssm, const DeviceLayer& l,
                    std::uint32_t layer_idx, const float* x,
                    std::uint32_t count, std::uint32_t first_position,
                    float* mixed);
  /// One dense projection through the grouped kernel.
  void Project(const DeviceTensor& w, const float* x, std::uint32_t rows,
               float* y);
  /// Stages a dense single-group table for `rows` rows into dense_groups_.
  void StageDenseGroups(std::uint32_t rows);
  /// The routed experts of one layer over the rows of `h`.
  void Experts(const DeviceLayer& l, const float* h, std::uint32_t count,
               float* routed, float* weights);
  /// The one sigmoid-gated shared expert over `rows` rows of `h`.
  void SharedExpert(const DeviceLayer& l, const float* h, std::uint32_t rows,
                    float* out, float* gated);

  const DeviceModel& model_;
  std::uint32_t max_rows_;
  std::uint32_t max_logit_rows_;
  std::uint32_t max_context_;
  hipStream_t stream_{nullptr};

  void* scratch_{nullptr};
  std::int32_t* tokens_{nullptr};
  float* x_{nullptr};
  float* h_{nullptr};
  float* qkv_{nullptr};
  float* conv_{nullptr};
  float* gate_{nullptr};
  float* alpha_{nullptr};
  float* beta_{nullptr};
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* attn_{nullptr};
  float* o_{nullptr};
  float* moe_gate_{nullptr};
  float* moe_up_{nullptr};
  float* moe_act_{nullptr};
  float* moe_out_{nullptr};
  float* moe_weights_{nullptr};
  std::int32_t* moe_ids_{nullptr};
  float* moe_logits_{nullptr};
  std::uint32_t* moe_counts_{nullptr};
  std::int32_t* moe_groups_{nullptr};
  std::int32_t* dense_groups_{nullptr};
  float* dense_gate_{nullptr};
  float* dense_up_{nullptr};
  float* shared_{nullptr};
  float* tmp_mix_{nullptr};
  float* logits_{nullptr};
  void* act_q8_{nullptr};
  std::size_t act_q8_bytes_{0};
  float* partials_{nullptr};
};

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_
