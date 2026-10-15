#include "src/models/kolibri1/kernels/rocm/device_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::kolibri1::rocm {
namespace {

/// The kernels read a few elements past the end of a row when a dimension does
/// not divide the tile they work on. The payload is a read-only mapping and
/// cannot be overread safely, so every tensor gets this much slack of its own.
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

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
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
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    return d;
  }
};

}  // namespace

WeightFormat DeviceTensor::quant() const {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return WeightFormat::kQ4_K;
    case core::GgmlType::kQ6_K:
      return WeightFormat::kQ6_K;
    default:
      throw std::invalid_argument(
          "kolibri1 projects only Q4_K and Q6_K weights");
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
  m->config_ = weights.config;
  m->vocab_ = weights.VocabSize();
  // Every pair of a head turns: the released file carries no rope dimension
  // count, so the divisor is the head dimension and not a rotary subset.
  m->theta_scale_ =
      std::pow(static_cast<double>(weights.config.rope_theta),
               -2.0 / static_cast<double>(weights.config.head_dim));

  const auto regions = reader.GetMappedRegions();
  auto stager = gufo::hip::WeightUpload::Create(
      std::span<const core::GgufMappedRegion>(regions.begin(), regions.end()),
      error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, error_msg};
  m->token_embd_ = up.Copy(weights.token_embd);
  // A tied head is the same bytes: one upload, two views of it.
  m->output_ = weights.TiedOutput() ? m->token_embd_ : up.Copy(weights.output);
  m->output_norm_ = up.Copy(weights.output_norm);
  m->layers_.reserve(weights.layers.size());
  for (const LayerWeights& l : weights.layers) {
    DeviceLayer d;
    d.attn_norm = up.Copy(l.attn_norm);
    d.attn_q = up.Copy(l.attn_q);
    d.attn_k = up.Copy(l.attn_k);
    d.attn_v = up.Copy(l.attn_v);
    d.attn_q_norm = up.Copy(l.attn_q_norm);
    d.attn_k_norm = up.Copy(l.attn_k_norm);
    d.attn_output = up.Copy(l.attn_output);
    d.post_attn_norm = up.Copy(l.post_attn_norm);
    d.ffn_norm = up.Copy(l.ffn_norm);
    d.router = up.Copy(l.router);
    d.router_bias = up.Copy(l.router_bias);
    d.gate_exps = up.Copy(l.gate_exps);
    d.up_exps = up.Copy(l.up_exps);
    d.down_exps = up.Copy(l.down_exps);
    d.gate_shexp = up.Copy(l.gate_shexp);
    d.up_shexp = up.Copy(l.up_shexp);
    d.down_shexp = up.Copy(l.down_shexp);
    d.post_ffn_norm = up.Copy(l.post_ffn_norm);
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

}  // namespace gufo::models::kolibri1::rocm
