#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::qwen35moe::rocm {
namespace {

// The kernels read a few elements past the end of a row when a dimension does
// not divide the tile they work on. The payload is a read-only mapping and
// cannot be overread safely, so every tensor gets this much slack of its own.
constexpr std::size_t kTailMargin = 4096;

struct Uploader {
  gufo::hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  void* Allocate(std::size_t size, std::string_view what) {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(what) + " (" +
           std::to_string(size) + " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    return ptr;
  }

  // Copies the tensor after checking the shape the binder recorded against
  // the layout the kernels read: [cols][rows][experts] with cols contiguous.
  DeviceTensor Copy(const TensorRef& t, std::uint64_t cols, std::uint64_t rows,
                    std::uint64_t experts) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.cols != cols || t.rows != rows || t.experts != experts) {
      Fail("tensor " + std::string(t.name) + " has shape [" +
           std::to_string(t.cols) + ", " + std::to_string(t.rows) + ", " +
           std::to_string(t.experts) + "], expected [" + std::to_string(cols) +
           ", " + std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return d;
    }
    const std::size_t size = t.SizeBytes();
    if (size == 0) {
      Fail("tensor " + std::string(t.name) + " has no encoded size");
      return d;
    }
    void* ptr = Allocate(size, t.name);
    if (ptr == nullptr) {
      return d;
    }
    if (!stager.Copy(t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.name = t.name;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(rows);
    d.experts = static_cast<std::uint32_t>(experts);
    return d;
  }
};

}  // namespace

WeightFormat DeviceTensor::quant() const {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return WeightFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return WeightFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return WeightFormat::kQ6_K;
    case core::GgmlType::kQ8_0:
      return WeightFormat::kQ8_0;
    case core::GgmlType::kIQ4_XS:
      return WeightFormat::kIQ4_XS;
    default:
      throw std::invalid_argument(
          "qwen35moe projections need Q4_K, Q5_K, Q6_K, Q8_0 or IQ4_XS "
          "weights");
  }
}

DeviceModel::~DeviceModel() {
  for (void* ptr : allocations_) {
    (void)hipFree(ptr);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& weights,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  const Config& c = weights.config;
  m->config_ = c;
  const std::uint64_t h = c.hidden_size;
  m->vocab_ = weights.token_embd.empty()
                  ? 0
                  : static_cast<std::uint32_t>(weights.token_embd.rows);
  m->theta_scale_ = std::pow(static_cast<double>(c.rope_theta),
                             -2.0 / static_cast<double>(c.rotary_dim));

  const auto regions = reader.GetMappedRegions();
  auto stager = gufo::hip::WeightUpload::Create(
      std::span<const core::GgufMappedRegion>(regions.begin(), regions.end()),
      error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, error_msg};
  m->token_embd_ = up.Copy(weights.token_embd, h, m->vocab_, 1);
  // A tied head is the same bytes: one upload, two views of it.
  const bool tied =
      weights.output.data == weights.token_embd.data &&
      weights.output.num_elements == weights.token_embd.num_elements;
  m->output_ = tied ? m->token_embd_ : up.Copy(weights.output, h, m->vocab_, 1);
  m->output_norm_ = up.Copy(weights.output_norm, h, 1, 1);

  const std::uint64_t qd = c.AttnOutDim();
  const std::uint64_t kvd = c.KvDim();
  const std::uint64_t qkv = c.SsmQkvDim();
  const std::uint64_t vd = c.SsmValueDim();
  const std::uint64_t rk = c.SsmValueHeads();
  const std::uint64_t hd = c.head_dim;
  const std::uint64_t hv = c.SsmValueHeadDim();
  const std::uint64_t e = c.num_experts;
  const std::uint64_t f = c.expert_ffn_size;
  const std::uint64_t s = c.shared_ffn_size;
  m->layers_.reserve(weights.layers.size());
  for (const LayerWeights& l : weights.layers) {
    DeviceLayer d;
    d.is_full_attention = l.is_full_attention;
    d.attn_norm = up.Copy(l.attn_norm, h, 1, 1);
    if (l.is_full_attention) {
      d.attn_q = up.Copy(l.attn_q, h, 2 * qd, 1);
      d.attn_k = up.Copy(l.attn_k, h, kvd, 1);
      d.attn_v = up.Copy(l.attn_v, h, kvd, 1);
      d.attn_output = up.Copy(l.attn_output, qd, h, 1);
      d.attn_q_norm = up.Copy(l.attn_q_norm, hd, 1, 1);
      d.attn_k_norm = up.Copy(l.attn_k_norm, hd, 1, 1);
    } else {
      d.attn_qkv = up.Copy(l.attn_qkv, h, qkv, 1);
      d.attn_gate = up.Copy(l.attn_gate, h, vd, 1);
      d.ssm_a = up.Copy(l.ssm_a, rk, 1, 1);
      d.ssm_conv1d = up.Copy(l.ssm_conv1d, c.ssm_conv_kernel, qkv, 1);
      d.ssm_dt = up.Copy(l.ssm_dt, rk, 1, 1);
      d.ssm_alpha = up.Copy(l.ssm_alpha, h, rk, 1);
      d.ssm_beta = up.Copy(l.ssm_beta, h, rk, 1);
      d.ssm_norm = up.Copy(l.ssm_norm, hv, 1, 1);
      d.ssm_out = up.Copy(l.ssm_out, vd, h, 1);
    }
    d.post_attention_norm = up.Copy(l.post_attention_norm, h, 1, 1);
    d.ffn_gate_inp = up.Copy(l.ffn_gate_inp, h, e, 1);
    d.ffn_gate_inp_shexp = up.Copy(l.ffn_gate_inp_shexp, h, 1, 1);
    d.ffn_gate_shexp = up.Copy(l.ffn_gate_shexp, h, s, 1);
    d.ffn_up_shexp = up.Copy(l.ffn_up_shexp, h, s, 1);
    d.ffn_down_shexp = up.Copy(l.ffn_down_shexp, s, h, 1);
    d.ffn_gate_exps = up.Copy(l.ffn_gate_exps, h, f, e);
    d.ffn_up_exps = up.Copy(l.ffn_up_exps, h, f, e);
    d.ffn_down_exps = up.Copy(l.ffn_down_exps, f, h, e);
    m->layers_.push_back(d);
  }
  if (!stager->Finish(error_msg)) {
    up.Fail("weight upload did not finish");
  }
  if (!up.ok) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::qwen35moe::rocm
