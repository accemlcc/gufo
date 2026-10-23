#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/config.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace gufo::models::qwen35moe::rocm {

/// One weight resident in device memory, still in its GGUF encoding. Nothing
/// is dequantized on the way up: this machine has one pool of memory shared by
/// the processor and the matrix cores, so a second copy in a wider format
/// would cost bandwidth twice - once to write it and once to read it - to save
/// arithmetic that the memory time already pays for.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};  ///< contiguous reduction dimension
  std::uint32_t rows{1};  ///< output dimension
  std::uint32_t experts{1};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
  /// The format a projection kernel must decode. Anything but the types the
  /// released artifacts carry throws: a silent conversion here would show up
  /// as a model that answers slightly wrong rather than as an artifact the
  /// engine does not read.
  [[nodiscard]] WeightFormat quant() const;
};

/// One trunk block. Linear layers carry the Gated DeltaNet set, full-attention
/// layers the GQA set; every layer carries the same mixture.
struct DeviceLayer {
  bool is_full_attention{false};
  DeviceTensor attn_norm;
  // Full attention.
  DeviceTensor attn_q;
  DeviceTensor attn_k;
  DeviceTensor attn_v;
  DeviceTensor attn_output;
  DeviceTensor attn_q_norm;
  DeviceTensor attn_k_norm;
  // Gated DeltaNet.
  DeviceTensor attn_qkv;
  DeviceTensor attn_gate;
  DeviceTensor ssm_a;
  DeviceTensor ssm_conv1d;
  DeviceTensor ssm_dt;
  DeviceTensor ssm_alpha;
  DeviceTensor ssm_beta;
  DeviceTensor ssm_norm;
  DeviceTensor ssm_out;
  // Mixture.
  DeviceTensor post_attention_norm;
  DeviceTensor ffn_gate_inp;
  DeviceTensor ffn_gate_inp_shexp;
  DeviceTensor ffn_gate_shexp;
  DeviceTensor ffn_up_shexp;
  DeviceTensor ffn_down_shexp;
  DeviceTensor ffn_gate_exps;
  DeviceTensor ffn_up_exps;
  DeviceTensor ffn_down_exps;
};

/// The whole model in device memory, uploaded once and read by every forward.
class DeviceModel {
 public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams every tensor from the mapped payload into its own allocation.
  /// On failure returns null and leaves `error_msg` the reason.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] std::uint32_t vocab_size() const noexcept { return vocab_; }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Pair d of a head turns by position * theta_scale^d. A double: at the
  /// trained context length the float form of the angle is worth less than the
  /// cosine needs.
  [[nodiscard]] double theta_scale() const noexcept { return theta_scale_; }

 private:
  DeviceModel() = default;

  Config config_;
  std::uint32_t vocab_{0};
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  double theta_scale_{0.0};
};

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_
