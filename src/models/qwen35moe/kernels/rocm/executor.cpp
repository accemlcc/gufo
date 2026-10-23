// Qwen3.5-MoE's device forward: embed, one trunk layer after the other, the
// output norm and the head. Linear layers run the gated delta net path,
// every fourth layer the full GQA path; every layer ends in the same
// softmax top-k mixture with its sigmoid-gated shared expert.
#include "src/models/qwen35moe/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/kernels/rocm/attention.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/kernels/rocm/qkv_post.hpp"
#include "src/models/qwen35moe/kernels/rocm/ssm.hpp"

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/kernels/rocm/attention.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/kernels/rocm/qkv_post.hpp"
#include "src/models/qwen35moe/kernels/rocm/ssm.hpp"

namespace gufo::models::qwen35moe::rocm {
namespace {

constexpr std::size_t kAlign = 16;

std::size_t Pad(std::size_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

struct Arena {
  char* at{nullptr};

  template<class T>
  T* Take(std::size_t count) {
    auto* p = reinterpret_cast<T*>(at);
    at += Pad(count * sizeof(T));
    return p;
  }
};

/// Group table for one dense projection over `rows` rows.
void DenseGroups(std::int32_t* groups, std::uint32_t rows) {
  groups[0] = 1;
  groups[1] = 0;
  groups[2] = static_cast<std::int32_t>(rows);
  for (std::uint32_t r = 0; r < rows; ++r) {
    groups[3 + r] = static_cast<std::int32_t>(r);
  }
}

}  // namespace

KvCache::~KvCache() {
  if (allocation != nullptr) {
    (void)hipFree(allocation);
  }
}

SsmState::~SsmState() {
  if (allocation != nullptr) {
    (void)hipFree(allocation);
  }
}

std::size_t Executor::CacheBytes(const Config& config,
                                 std::uint32_t max_context) noexcept {
  std::size_t bytes = 0;
  const std::size_t kv_row =
      static_cast<std::size_t>(config.num_kv_heads) * config.head_dim;
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    if (!config.IsLinear(l)) {
      bytes += 2 * Pad(max_context * kv_row * sizeof(float));
    } else {
      bytes += Pad(config.SsmQkvDim() * 4 * sizeof(float));
      bytes += Pad(static_cast<std::size_t>(config.ssm_time_step_rank) *
                   128 * 128 * sizeof(float));
    }
  }
  return bytes;
}

Executor::Executor(const DeviceModel& model, std::uint32_t max_rows,
                   std::uint32_t max_logit_rows, std::uint32_t max_context)
    : model_(model),
      max_rows_(max_rows),
      max_logit_rows_(max_logit_rows),
      max_context_(max_context) {
  HIP_CHECK(hipStreamCreate(&stream_));
  const Config& c = model.config();
  const std::size_t d = c.hidden_size;
  const std::size_t v = model.vocab_size();
  const std::size_t qd = c.AttnOutDim();
  const std::size_t kvd = c.KvDim();
  const std::size_t qkv = c.SsmQkvDim();
  const std::size_t vd = c.SsmValueDim();
  const std::size_t rk = c.SsmValueHeads();
  const std::size_t f = c.expert_ffn_size;
  const std::size_t s = c.shared_ffn_size;
  const std::size_t u = c.experts_used;
  const std::size_t e = c.num_experts;
  const std::size_t qkv_max = std::max<std::size_t>(2 * qd, qkv);
  const std::size_t mix_max = std::max(qd, vd);
  std::size_t arena = 0;
  const auto need = [&](std::size_t count, std::size_t element) {
    arena += Pad(count * element);
  };
  need(max_rows, sizeof(std::int32_t));                 // tokens_
  need(max_rows * d, sizeof(float));                    // x_
  need(max_rows * d, sizeof(float));                    // h_
  need(max_rows * qkv_max, sizeof(float));              // qkv_
  need(max_rows * qkv_max, sizeof(float));              // conv_
  need(max_rows * kvd, sizeof(float));                  // k_
  need(max_rows * kvd, sizeof(float));                  // v_
  need(max_rows * qd, sizeof(float));                   // attn_
  need(max_rows * mix_max, sizeof(float));              // o_ (gated)
  need(max_rows * vd, sizeof(float));                   // gate_ (ssm z)
  need(max_rows * rk, sizeof(float));                   // alpha_
  need(max_rows * rk, sizeof(float));                   // beta_
  need(max_rows * u, sizeof(float));                    // moe_weights_
  need(max_rows * u, sizeof(std::int32_t));             // moe_ids_
  need(e, sizeof(std::uint32_t));                       // moe_counts_
  need(GroupTableInts(static_cast<std::uint32_t>(e)),
       sizeof(std::int32_t));                           // moe_groups_
  need(max_rows * u * f, sizeof(float));                // moe_gate_
  need(max_rows * u * f, sizeof(float));                // moe_up_
  need(max_rows * u * f, sizeof(float));                // moe_act_
  need(max_rows * u * d, sizeof(float));                // moe_out_
  need(max_rows * s, sizeof(float));                    // dense_gate_
  need(max_rows * s, sizeof(float));                    // dense_up_
  need(max_rows * d, sizeof(float));                    // shared_
  need(max_rows * d, sizeof(float));                    // tmp_mix_
  need(GroupTableInts(kMaxGroupSlots), sizeof(std::int32_t));
  need(max_logit_rows * v, sizeof(float));              // logits_
  HIP_CHECK(hipMalloc(&scratch_, arena));
  Arena a{static_cast<char*>(scratch_)};
  tokens_ = a.Take<std::int32_t>(max_rows_);
  x_ = a.Take<float>(max_rows_ * d);
  h_ = a.Take<float>(max_rows_ * d);
  qkv_ = a.Take<float>(max_rows_ * qkv_max);
  conv_ = a.Take<float>(max_rows_ * qkv_max);
  k_ = a.Take<float>(max_rows_ * kvd);
  v_ = a.Take<float>(max_rows_ * kvd);
  attn_ = a.Take<float>(max_rows_ * qd);
  o_ = a.Take<float>(max_rows_ * mix_max);
  gate_ = a.Take<float>(max_rows_ * vd);
  alpha_ = a.Take<float>(max_rows_ * rk);
  beta_ = a.Take<float>(max_rows_ * rk);
  moe_weights_ = a.Take<float>(max_rows_ * u);
  moe_ids_ = a.Take<std::int32_t>(max_rows_ * u);
  moe_counts_ = a.Take<std::uint32_t>(e);
  moe_groups_ = a.Take<std::int32_t>(
      GroupTableInts(static_cast<std::uint32_t>(e)));
  moe_gate_ = a.Take<float>(max_rows_ * u * f);
  moe_up_ = a.Take<float>(max_rows_ * u * f);
  moe_act_ = a.Take<float>(max_rows_ * u * f);
  moe_out_ = a.Take<float>(max_rows_ * u * d);
  dense_gate_ = a.Take<float>(max_rows_ * s);
  dense_up_ = a.Take<float>(max_rows_ * s);
  shared_ = a.Take<float>(max_rows_ * d);
  tmp_mix_ = a.Take<float>(max_rows_ * d);
  dense_groups_ = a.Take<std::int32_t>(GroupTableInts(kMaxGroupSlots));
  logits_ = a.Take<float>(max_logit_rows_ * v);
}

Executor::~Executor() {
  if (scratch_ != nullptr) {
    (void)hipFree(scratch_);
  }
  (void)hipStreamDestroy(stream_);
}

std::unique_ptr<KvCache> Executor::CreateCache(
    std::uint32_t max_context, std::string* error_msg) const {
  const Config& c = model_.config();
  auto cache = std::unique_ptr<KvCache>(new KvCache());
  cache->max_context = max_context;
  std::size_t bytes = 0;
  const std::size_t kv_row =
      static_cast<std::size_t>(c.num_kv_heads) * c.head_dim;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (!c.IsLinear(l)) {
      bytes += 2 * Pad(max_context * kv_row * sizeof(float));
    }
  }
  if (hipMalloc(&cache->allocation, bytes + 4096) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "KvCache allocation failed";
    }
    return nullptr;
  }
  cache->bytes = bytes;
  char* at = static_cast<char*>(cache->allocation);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.IsLinear(l)) {
      cache->k.push_back(nullptr);
      cache->v.push_back(nullptr);
      continue;
    }
    const std::size_t plane = Pad(max_context * kv_row * sizeof(float));
    cache->k.push_back(reinterpret_cast<float*>(at));
    at += plane;
    cache->v.push_back(reinterpret_cast<float*>(at));
    at += plane;
  }
  HIP_CHECK(hipMemsetAsync(cache->allocation, 0, bytes, stream_));
  return cache;
}

std::unique_ptr<SsmState> Executor::CreateSsmState(
    std::string* error_msg) const {
  const Config& c = model_.config();
  auto st = std::unique_ptr<SsmState>(new SsmState());
  std::size_t bytes = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (!c.IsLinear(l)) {
      continue;
    }
    bytes += Pad(c.SsmQkvDim() * 4 * sizeof(float));
    bytes += Pad(static_cast<std::size_t>(c.ssm_time_step_rank) * 128 * 128 *
                 sizeof(float));
  }
  if (hipMalloc(&st->allocation, bytes + 4096) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "SsmState allocation failed";
    }
    return nullptr;
  }
  char* at = static_cast<char*>(st->allocation);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (!c.IsLinear(l)) {
      st->conv.push_back(nullptr);
      st->recurrent.push_back(nullptr);
      continue;
    }
    const std::size_t ch = Pad(c.SsmQkvDim() * 4 * sizeof(float));
    st->conv.push_back(reinterpret_cast<float*>(at));
    at += ch;
    const std::size_t rh = Pad(static_cast<std::size_t>(
                                   c.ssm_time_step_rank) *
                               128 * 128 * sizeof(float));
    st->recurrent.push_back(reinterpret_cast<float*>(at));
    at += rh;
  }
  HIP_CHECK(hipMemsetAsync(st->allocation, 0, bytes, stream_));
  return st;
}

void Executor::Forward(KvCache& cache, SsmState& ssm,
                       std::span<const std::int32_t> tokens,
                       std::uint32_t first_position,
                       std::span<const std::uint32_t> logit_rows) {
  const Config& c = model_.config();
  const std::size_t d = c.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > max_rows_) {
    throw std::invalid_argument("forward of empty or oversize chunk");
  }
  if (first_position + n > max_context_) {
    throw std::invalid_argument("forward runs past the context bound");
  }
  if (logit_rows.size() > max_logit_rows_) {
    throw std::invalid_argument("forward asks for too many logit rows");
  }
  HIP_CHECK(hipMemcpyAsync(tokens_, tokens.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
  EmbedRows({model_.token_embd().data, model_.token_embd().quant(), tokens_,
             x_, n, static_cast<std::uint32_t>(d)},
            stream_);
  // DIAGNOSTIC ONLY: with GUFO_QWEN35MOE_DUMP_DEV_LAYERS set to a directory,
  // writes the embed stream, the post-residual stream after every layer and
  // the final normed stream. Filenames carry a per-Forward call index so the
  // test's prefill plus two decodes do not overwrite each other. Remove
  // before finishing: it synchronizes the stream on every layer.
  const char* dev_dump = std::getenv("GUFO_QWEN35MOE_DUMP_DEV_LAYERS");
  static int dev_call = 0;
  const int this_call = dev_call;
  dump_call_ = this_call;
  auto dump_stream = [&](const char* tag, const float* ptr) {
    DumpStage(tag, ptr, n, static_cast<std::uint32_t>(d), this_call);
  };
  {
    char tag[32];
    std::snprintf(tag, sizeof(tag), "embed");
    dump_stream(tag, x_);
  }
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    ForwardLayer(cache, ssm, model_.layers()[l], l, x_, n, first_position, h_);
    if (dev_dump != nullptr) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "layer%02u_mix", l);
      DumpStage(tag, h_, n, static_cast<std::uint32_t>(d), this_call);
    }
    AddRows({x_, h_, static_cast<std::size_t>(n) * d}, stream_);
    const DeviceLayer& w = model_.layers()[l];
    RmsNormRows({x_, w.post_attention_norm.f32(), h_, n,
                 static_cast<std::uint32_t>(d), c.rms_eps},
                stream_);
    Experts(w, h_, n, moe_out_, moe_weights_);
    SharedExpert(w, h_, n, shared_, shared_);
    MoeCombineArgs margs{};
    margs.out = tmp_mix_;
    margs.routed = moe_out_;
    margs.shared = shared_;
    margs.weights = moe_weights_;
    margs.rows = n;
    margs.hidden = static_cast<std::uint32_t>(d);
    margs.used = c.experts_used;
    MoeCombine(margs, stream_);
    AddRows({x_, tmp_mix_, static_cast<std::size_t>(n) * d}, stream_);
    if (dev_dump != nullptr) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "layer%02u", l);
      dump_stream(tag, x_);
      std::snprintf(tag, sizeof(tag), "layer%02u_pnorm", l);
      DumpStage(tag, h_, n, static_cast<std::uint32_t>(d), this_call);
      std::snprintf(tag, sizeof(tag), "layer%02u_moe", l);
      DumpStage(tag, tmp_mix_, n, static_cast<std::uint32_t>(d), this_call);
    }
  }
  RmsNormRows({x_, model_.output_norm().f32(), h_, n,
               static_cast<std::uint32_t>(d), c.rms_eps},
              stream_);
  dump_stream("final", h_);
  ++dev_call;
  for (std::size_t k = 0; k < logit_rows.size(); ++k) {
    const std::uint32_t r = logit_rows[k];
    if (r >= n) {
      throw std::invalid_argument("logit row past the chunk");
    }
    StageDenseGroups(1);
    if (!GroupedGemv({model_.output().quant(), model_.output().data,
                       dense_groups_, 1, h_ + static_cast<std::size_t>(r) * d,
                       logits_ + k * model_.vocab_size(), 1, 1,
                       model_.vocab_size(), static_cast<std::uint32_t>(d)},
                      stream_)) {
      throw std::invalid_argument("no grouped kernel for the output head");
    }
  }
}

void Executor::ForwardLayer(KvCache& cache, SsmState& ssm,
                            const DeviceLayer& w, std::uint32_t layer,
                            const float* x, std::uint32_t count,
                            std::uint32_t first_position, float* mixed) {
  const Config& c = model_.config();
  const std::size_t d = c.hidden_size;
  const std::uint32_t n = count;
  RmsNormRows({x, w.attn_norm.f32(), mixed, n, static_cast<std::uint32_t>(d),
               c.rms_eps},
              stream_);
  if (dump_call_ >= 0) {
    char tag[32];
    std::snprintf(tag, sizeof(tag), "layer%02u_anorm", layer);
    DumpStage(tag, mixed, n, static_cast<std::uint32_t>(d), dump_call_);
  }
  if (!w.is_full_attention) {
    const std::uint32_t qkv = c.SsmQkvDim();
    const std::uint32_t vd = c.SsmValueDim();
    const std::uint32_t rk = c.SsmValueHeads();
    Project(w.attn_qkv, mixed, n, qkv_);
    Project(w.attn_gate, mixed, n, gate_);
    Project(w.ssm_alpha, mixed, n, alpha_);
    Project(w.ssm_beta, mixed, n, beta_);
    SsmArgs sa{};
    sa.qkv = qkv_;
    sa.conv_w = static_cast<const float*>(w.ssm_conv1d.data);
    sa.conv_state = ssm.conv[layer];
    sa.conv_out = conv_;
    sa.recurrent = ssm.recurrent[layer];
    sa.alpha = alpha_;
    sa.beta = beta_;
    sa.ssm_a = w.ssm_a.f32();
    sa.ssm_dt = w.ssm_dt.f32();
    sa.ssm_norm = w.ssm_norm.f32();
    sa.gate = gate_;
    sa.out = o_;
    sa.rows = n;
    sa.qkv_dim = qkv;
    sa.num_key_heads = c.ssm_group_count;
    sa.num_heads = rk;
    sa.key_dim = c.ssm_state_size;
    sa.value_dim = vd;
    sa.head_dim = c.SsmValueHeadDim();
    LaunchSsm(sa, stream_);
    Project(w.ssm_out, o_, n, mixed);
    return;
  }
  const std::uint32_t qd = c.AttnOutDim();
  const std::uint32_t kvd = c.KvDim();
  Project(w.attn_q, mixed, n, qkv_);
  Project(w.attn_k, mixed, n, k_);
  Project(w.attn_v, mixed, n, v_);
  QkvPostArgs qargs{};
  qargs.qg = qkv_;
  qargs.k = k_;
  qargs.v = v_;
  qargs.q_norm = w.attn_q_norm.f32();
  qargs.k_norm = w.attn_k_norm.f32();
  qargs.q_out = attn_;
  qargs.gate_out = o_;
  qargs.k_cache = cache.k[layer];
  qargs.v_cache = cache.v[layer];
  qargs.rows = n;
  qargs.first_position = first_position;
  qargs.max_context = max_context_;
  qargs.heads = c.num_heads;
  qargs.kv_heads = c.num_kv_heads;
  qargs.dim = c.head_dim;
  qargs.rotary_dim = c.rotary_dim;
  qargs.theta_scale = model_.theta_scale();
  qargs.eps = c.rms_eps;
  QkvPost(qargs, stream_);
  AttentionArgs aargs{};
  aargs.q = attn_;
  aargs.k_cache = cache.k[layer];
  aargs.v_cache = cache.v[layer];
  aargs.out = attn_;
  aargs.rows = n;
  aargs.first_position = first_position;
  aargs.max_context = max_context_;
  aargs.heads = c.num_heads;
  aargs.kv_heads = c.num_kv_heads;
  aargs.dim = c.head_dim;
  aargs.scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));
  Attention(aargs, stream_);
  SigmoidGateAct({attn_, o_, attn_, static_cast<std::size_t>(n) * qd},
                 stream_);
  Project(w.attn_output, attn_, n, mixed);
}

void Executor::Project(const DeviceTensor& w, const float* x,
                       std::uint32_t rows, float* y) {
  StageDenseGroups(rows);
  if (!GroupedGemv({w.quant(), w.data, dense_groups_, 1, x, y, rows, 1,
                     w.rows, w.cols},
                    stream_)) {
    throw std::invalid_argument("no grouped kernel for " +
                                std::string(w.name));
  }
}

void Executor::Experts(const DeviceLayer& l, const float* h,
                       std::uint32_t count, float* routed, float* weights) {
  const Config& c = model_.config();
  const std::uint32_t u = c.experts_used;
  const std::uint32_t e = c.num_experts;
  const std::size_t f = c.expert_ffn_size;
  HIP_CHECK(hipMemsetAsync(moe_counts_, 0, e * sizeof(std::uint32_t), stream_));
  MoeRouteArgs rargs{};
  rargs.x = h;
  rargs.router = l.ffn_gate_inp.f32();
  rargs.logits = moe_logits_;
  rargs.ids = moe_ids_;
  rargs.weights = weights;
  rargs.counts = moe_counts_;
  rargs.groups = moe_groups_;
  rargs.rows = count;
  rargs.hidden = static_cast<std::uint32_t>(c.hidden_size);
  rargs.experts = e;
  rargs.used = u;
  MoeRoute(rargs, stream_);
  auto routed_proj = [&](const DeviceTensor& w, const float* in, float* y,
                         std::uint32_t m, std::uint32_t k,
                         std::uint32_t spr) {
    if (!GroupedGemv({w.quant(), w.data, moe_groups_,
                       std::min(e, count * u), in, y, count, spr, m, k},
                      stream_)) {
      throw std::invalid_argument("no grouped kernel for " +
                                  std::string(w.name));
    }
  };
  routed_proj(l.ffn_gate_exps, h, moe_gate_, static_cast<std::uint32_t>(f),
              static_cast<std::uint32_t>(c.hidden_size), u);
  routed_proj(l.ffn_up_exps, h, moe_up_, static_cast<std::uint32_t>(f),
              static_cast<std::uint32_t>(c.hidden_size), u);
  GateUpAct({moe_gate_, moe_up_, moe_act_,
             static_cast<std::size_t>(count) * u * f},
            stream_);
  routed_proj(l.ffn_down_exps, moe_act_, routed,
              static_cast<std::uint32_t>(c.hidden_size),
              static_cast<std::uint32_t>(f), 1);
}

void Executor::SharedExpert(const DeviceLayer& l, const float* h,
                            std::uint32_t rows, float* out, float* gated) {
  const Config& c = model_.config();
  const std::uint32_t s = c.shared_ffn_size;
  const std::uint32_t d = c.hidden_size;
  StageDenseGroups(rows);
  auto dense = [&](const DeviceTensor& w, const float* in, float* y,
                   std::uint32_t m, std::uint32_t k) {
    if (!GroupedGemv({w.quant(), w.data, dense_groups_, 1, in, y, rows, 1, m,
                       k},
                      stream_)) {
      throw std::invalid_argument("no grouped kernel for " +
                                  std::string(w.name));
    }
  };
  dense(l.ffn_gate_shexp, h, dense_gate_, s, d);
  dense(l.ffn_up_shexp, h, dense_up_, s, d);
  GateUpAct({dense_gate_, dense_up_, dense_up_, static_cast<std::size_t>(rows) * s}, stream_);
  dense(l.ffn_down_shexp, dense_up_, out, d, s);
  GateSharedArgs gargs{};
  gargs.shared = out;
  gargs.gate = l.ffn_gate_inp_shexp.f32();
  gargs.h = h;
  gargs.rows = rows;
  gargs.hidden = d;
  GateShared(gargs, stream_);
  if (gated != out) {
    HIP_CHECK(hipMemcpyAsync(gated, out,
                             static_cast<std::size_t>(rows) * d * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
  }
}

void Executor::StageDenseGroups(std::uint32_t rows) {
  std::vector<std::int32_t> host(3 + rows);
  DenseGroups(host.data(), rows);
  HIP_CHECK(hipMemcpyAsync(dense_groups_, host.data(),
                           (3 + rows) * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
}

void Executor::CopyLogits(std::size_t rows,
                          std::vector<float>* out) const {
  out->assign(rows * model_.vocab_size(), 0.0F);
  HIP_CHECK(hipMemcpyAsync(out->data(), logits_,
                           rows * model_.vocab_size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

// DIAGNOSTIC ONLY. Remove before finishing.
void Executor::DumpStage(const char* tag, const float* ptr, std::uint32_t n,
                         std::uint32_t d, int call) {
  const char* dir = std::getenv("GUFO_QWEN35MOE_DUMP_DEV_LAYERS");
  if (dir == nullptr) {
    return;
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  std::vector<float> host(static_cast<std::size_t>(n) * d);
  HIP_CHECK(hipMemcpy(host.data(), ptr, host.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  char path[320];
  std::snprintf(path, sizeof(path), "%s/dev_c%d_%s.bin", dir, call, tag);
  FILE* f = std::fopen(path, "wb");
  if (f != nullptr) {
    std::fwrite(host.data(), sizeof(float), host.size(), f);
    std::fclose(f);
  }
}

}  // namespace gufo::models::qwen35moe::rocm
