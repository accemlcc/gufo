#include "src/models/kolibri1/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/kolibri1/kernels/rocm/attention.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"
#include "src/models/kolibri1/kernels/rocm/qkv_post.hpp"

namespace gufo::models::kolibri1::rocm {
namespace {

/// Every scratch buffer starts on this boundary so a kernel may read a vector
/// of floats or halves as one value.
constexpr std::size_t kAlign = 256;

[[nodiscard]] constexpr std::size_t Pad(std::size_t n) noexcept {
  return (n + kAlign - 1) & ~static_cast<std::size_t>(kAlign - 1);
}

/// Hands out the buffers of one allocation in declaration order. A forward is
/// a long chain of kernels and every one of them wants a named piece of memory;
/// carving them from a single allocation keeps the launch path free of the
/// allocator and makes the model's working set one number.
struct Arena {
  char* at{nullptr};

  template<class T>
  T* Take(std::size_t count) {
    auto* p = reinterpret_cast<T*>(at);
    at += Pad(count * sizeof(T));
    return p;
  }
};

[[nodiscard]] constexpr std::size_t ArenaBytes(std::size_t count,
                                               std::size_t element) noexcept {
  return Pad(count * element);
}

}  // namespace

KvCache::~KvCache() {
  if (allocation != nullptr) {
    (void)hipFree(allocation);
  }
}

std::uint32_t Executor::RingSlots(const Config& config,
                                  std::uint32_t max_rows) noexcept {
  // A sliding layer may read its window of the past plus the rows of the
  // forward it is part of; a ring at least that wide never overwrites a key a
  // live row still reads, which is what makes a rollback a frontier move.
  const std::uint32_t want = config.sliding_window + max_rows;
  std::uint32_t ring = 1;
  while (ring < want) {
    ring <<= 1;
  }
  return ring;
}

std::size_t Executor::CacheBytes(const Config& config,
                                 std::uint32_t max_context,
                                 std::uint32_t ring) noexcept {
  std::size_t bytes = 0;
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const std::size_t capacity = config.IsSliding(l) ? ring : max_context;
    const std::size_t row =
        static_cast<std::size_t>(config.kv_heads[l]) * config.head_dim * 2;
    bytes += 2 * ArenaBytes(capacity * row, 2);
  }
  return bytes;
}

Executor::Executor(const DeviceModel& model, std::uint32_t max_rows,
                   std::uint32_t max_logit_rows, std::uint32_t max_context)
    : model_(model),
      max_rows_(max_rows),
      max_logit_rows_(max_logit_rows),
      max_context_(max_context),
      ring_(RingSlots(model.config(), max_rows)) {
  const Config& c = model.config();
  std::uint32_t kv_dim = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    kv_dim = std::max(kv_dim, c.KvDim(l));
  }
  const std::uint32_t d = c.hidden_size;
  const std::uint32_t q_dim = c.QDim();
  const std::uint32_t used = c.experts_used;
  const std::uint32_t experts = c.num_experts;
  const std::uint32_t width = c.expert_ffn_size;
  const std::size_t vocab = model.vocab_size();

  // Sizes in bytes, then the arena hands them out in the same order. Keeping
  // the two lists together is the only thing that keeps them honest.
  std::size_t bytes = 0;
  const auto Need = [&](std::size_t floats) {
    bytes += ArenaBytes(floats, 4);
  };
  Need(static_cast<std::size_t>(max_rows_) * d);       // x_
  Need(static_cast<std::size_t>(max_rows_) * d);       // h_
  Need(static_cast<std::size_t>(max_rows_) * q_dim);   // q_
  Need(static_cast<std::size_t>(max_rows_) * kv_dim);  // k_
  Need(static_cast<std::size_t>(max_rows_) * kv_dim);  // v_
  Need(static_cast<std::size_t>(max_rows_) * q_dim);   // attn_
  Need(static_cast<std::size_t>(max_rows_) * d);       // o_
  Need(static_cast<std::size_t>(max_rows_) * width);   // dense_gate_
  Need(static_cast<std::size_t>(max_rows_) * width);   // dense_up_
  Need(static_cast<std::size_t>(max_rows_) * width);   // dense_act_
  Need(static_cast<std::size_t>(max_rows_) * d);       // shared_
  Need(static_cast<std::size_t>(max_rows_) * used * width);    // moe_gate_
  Need(static_cast<std::size_t>(max_rows_) * used * width);    // moe_up_
  Need(static_cast<std::size_t>(max_rows_) * used * width);    // moe_act_
  Need(static_cast<std::size_t>(max_rows_) * used * d);        // moe_out_
  Need(static_cast<std::size_t>(max_rows_) * used);            // moe_weights_
  Need(static_cast<std::size_t>(max_rows_) * used);            // moe_ids_
  Need(experts);                                               // moe_counts_
  Need(GroupTableInts(experts));                               // moe_groups_
  Need(kProjectRows * GroupTableInts(1));                      // dense_groups_
  Need(static_cast<std::size_t>(max_rows));                    // tokens_
  Need(static_cast<std::size_t>(max_logit_rows_) * d);         // logit_x_
  Need(static_cast<std::size_t>(max_logit_rows_) * vocab);     // logits_
  Need(AttentionPartialFloats(1, c.num_heads, c.head_dim, max_context));
  HIP_CHECK(hipMalloc(&scratch_, bytes));
  Arena arena{static_cast<char*>(scratch_)};
  x_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * d);
  h_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * d);
  q_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * q_dim);
  k_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * kv_dim);
  v_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * kv_dim);
  attn_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * q_dim);
  o_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * d);
  dense_gate_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * width);
  dense_up_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * width);
  dense_act_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * width);
  shared_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * d);
  moe_gate_ =
      arena.Take<float>(static_cast<std::size_t>(max_rows_) * used * width);
  moe_up_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * used * width);
  moe_act_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * used * width);
  moe_out_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * used * d);
  moe_weights_ = arena.Take<float>(static_cast<std::size_t>(max_rows_) * used);
  moe_ids_ = arena.Take<std::int32_t>(static_cast<std::size_t>(max_rows_) * used);
  moe_counts_ = arena.Take<std::uint32_t>(experts);
  moe_groups_ = arena.Take<std::int32_t>(GroupTableInts(experts));
  dense_groups_ =
      arena.Take<std::int32_t>(kProjectRows * GroupTableInts(1));
  tokens_ = arena.Take<std::int32_t>(max_rows);
  logit_x_ = arena.Take<float>(static_cast<std::size_t>(max_logit_rows_) * d);
  logits_ = arena.Take<float>(static_cast<std::size_t>(max_logit_rows_) * vocab);
  partials_ =
      arena.Take<float>(AttentionPartialFloats(1, c.num_heads, c.head_dim,
                                               max_context));

  // The single-group table of a dense projection says: one expert, these rows
  // as its slots. Every width the executor steps through gets its own table
  // built once here, so a projection launch carries no host-to-device copy of
  // its own - at six projections and one routing per layer, one small copy per
  // launch would otherwise be a millisecond of a decode step.
  std::int32_t tables[kProjectRows * GroupTableInts(1)];
  for (std::uint32_t n = 1; n <= kProjectRows; ++n) {
    std::int32_t* t = tables + (n - 1) * GroupTableInts(1);
    t[0] = 1;  // one group
    t[1] = 0;  // its expert
    t[2] = static_cast<std::int32_t>(n);
    for (std::uint32_t s = 0; s < n; ++s) {
      t[3 + s] = static_cast<std::int32_t>(s);
    }
  }
  HIP_CHECK(hipMemcpy(dense_groups_, tables, sizeof(tables),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
}

Executor::~Executor() {
  if (scratch_ != nullptr) {
    (void)hipFree(scratch_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<KvCache> Executor::CreateCache(std::uint32_t max_context,
                                               std::string* error_msg) const {
  const Config& c = model_.config();
  const std::size_t bytes = CacheBytes(c, max_context, ring_);
  auto cache = std::make_unique<KvCache>();
  void* allocation = nullptr;
  if (hipMalloc(&allocation, bytes) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "cannot allocate a " + std::to_string(bytes) +
                   " byte cache for " + std::to_string(max_context) +
                   " positions";
    }
    return nullptr;
  }
  cache->allocation = allocation;
  cache->bytes = bytes;
  cache->max_context = max_context;
  cache->ring = ring_;
  char* at = static_cast<char*>(allocation);
  cache->k.resize(c.num_layers);
  cache->v.resize(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t capacity = c.IsSliding(l) ? ring_ : max_context;
    const std::size_t row =
        static_cast<std::size_t>(c.kv_heads[l]) * c.head_dim * 2;
    cache->k[l] = reinterpret_cast<std::uint16_t*>(at);
    at += ArenaBytes(capacity * row, 2);
    cache->v[l] = reinterpret_cast<std::uint16_t*>(at);
    at += ArenaBytes(capacity * row, 2);
  }
  return cache;
}

void Executor::Project(const DeviceTensor& w, const float* x,
                       std::uint32_t rows, float* y) {
  const std::uint32_t m = w.rows;
  const std::uint32_t k = w.cols;
  for (std::uint32_t base = 0; base < rows; base += kProjectRows) {
    const std::uint32_t n = std::min(kProjectRows, rows - base);
    const std::int32_t* table = dense_groups_ + (n - 1) * GroupTableInts(1);
    const GroupedGemvArgs args{w.quant(), w.data,    table,
                               1,        x + base * k, y + base * m,
                               n,        1,           m,
                               k};
    if (!GroupedGemv(args, stream_)) {
      throw std::invalid_argument(
          "kolibri1 has no projection kernel for this shape");
    }
  }
}

void Executor::SharedExpert(const DeviceLayer& l, const float* h,
                            std::uint32_t rows, float* out) {
  const std::size_t count =
      static_cast<std::size_t>(rows) * model_.config().shared_expert_ffn_size;
  Project(l.gate_shexp, h, rows, dense_gate_);
  Project(l.up_shexp, h, rows, dense_up_);
  GateUpAct({dense_gate_, dense_up_, dense_act_, count}, stream_);
  Project(l.down_shexp, dense_act_, rows, out);
}

void Executor::Experts(const DeviceLayer& l, const float* h,
                       std::uint32_t count, float* routed, float* weights) {
  const Config& c = model_.config();
  const std::uint32_t used = c.experts_used;
  const std::uint32_t experts = c.num_experts;
  const std::uint32_t d = c.hidden_size;
  const std::uint32_t width = c.expert_ffn_size;
  // The routing table holds kProjectRows rows' slots, so the mixture runs in
  // the same steps as the projections. Both read every expert weight they
  // touch, which is what makes a large prefill cost weight traffic per step
  // instead of per token - the routed half of this model is 224 MB per layer
  // and a step touches at most 96 of its 384 experts.
  for (std::uint32_t base = 0; base < count; base += kProjectRows) {
    const std::uint32_t n = std::min(kProjectRows, count - base);
    const std::uint32_t slots = n * used;
    const float* x = h + static_cast<std::size_t>(base) * d;
    float* w = weights + static_cast<std::size_t>(base) * used;
    HIP_CHECK(hipMemsetAsync(moe_counts_, 0, experts * sizeof(std::uint32_t),
                             stream_));
    MoeRoute({x, l.router.f32(), l.router_bias.f32(), nullptr,
              moe_ids_ + static_cast<std::size_t>(base) * used, w,
              moe_counts_, moe_groups_, n, d, experts, used},
             stream_);
    // The launch bound is the number of groups that can hold a slot at all,
    // not the number of experts: a grid over all 384 makes a decode step read
    // the count of 378 empty groups, which measured two to three times the
    // cost of the projection itself. That bound is only safe while the table
    // lists a group per expert that was chosen - see moe.hpp - because blocks
    // past the table's count return without writing their slots. A grouped
    // launch that is not written for a shape declines it, and running on
    // without its output would mix stale activations into the residual.
    // `slots_per_row` says how many slots read the same row of `in`. The gate
    // and the up half read their row's hidden state, so the six slots of a row
    // share one input row; the down half reads the activation its own slot has
    // just produced, so its input holds one row per slot. Handing `used` to the
    // down half makes every slot of a row multiply the first slot's activation,
    // which leaves only slot 0 of each row correct.
    const auto Routed = [&](const DeviceTensor& w, const float* in, float* out,
                            std::uint32_t m, std::uint32_t k,
                            std::uint32_t shared_rows) {
      if (!GroupedGemv({w.quant(), w.data, moe_groups_, slots, in, out, n,
                        shared_rows, m, k},
                       stream_)) {
        throw std::invalid_argument("kolibri1 has no grouped kernel for " +
                                    std::string(w.name));
      }
    };
    Routed(l.gate_exps, x,
           moe_gate_ + static_cast<std::size_t>(base) * used * width, width, d,
           used);
    Routed(l.up_exps, x,
           moe_up_ + static_cast<std::size_t>(base) * used * width, width, d,
           used);
    GateUpAct({moe_gate_ + static_cast<std::size_t>(base) * used * width,
               moe_up_ + static_cast<std::size_t>(base) * used * width,
               moe_act_ + static_cast<std::size_t>(base) * used * width,
               static_cast<std::size_t>(slots) * width},
              stream_);
    Routed(l.down_exps,
           moe_act_ + static_cast<std::size_t>(base) * used * width,
           routed + static_cast<std::size_t>(base) * used * d, d, width, 1);
  }
}

void Executor::Forward(KvCache& cache, std::span<const std::int32_t> tokens,
                       std::uint32_t first_position,
                       std::span<const std::uint32_t> logit_rows) {
  const Config& c = model_.config();
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > max_rows_) {
    throw std::invalid_argument("kolibri1 forward of " + std::to_string(n) +
                                " rows outside the executor's bound");
  }
  if (first_position + n > cache.max_context) {
    throw std::invalid_argument("kolibri1 forward past the cache's context");
  }
  const std::uint32_t d = c.hidden_size;
  HIP_CHECK(hipMemcpyAsync(tokens_, tokens.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
  EmbedRows({model_.token_embd().data, model_.token_embd().quant(), tokens_, x_,
             n, d},
            stream_);

  const float scale =
      1.0F / std::sqrt(static_cast<float>(c.head_dim));
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const DeviceLayer& w = model_.layers()[l];
    const bool sliding = c.IsSliding(l);
    // The unwindowed layers index the cache by position; a ring would drop the
    // beginning of a context they are defined to read whole.
    const std::uint32_t ring = sliding ? ring_ : cache.max_context;

    RmsNormRows({x_, w.attn_norm.f32(), h_, n, d, c.rms_eps}, stream_);
    Project(w.attn_q, h_, n, q_);
    Project(w.attn_k, h_, n, k_);
    Project(w.attn_v, h_, n, v_);
    QkvPost({q_, k_, v_, w.attn_q_norm.f32(), w.attn_k_norm.f32(), cache.k[l],
             cache.v[l], n, c.num_heads, c.kv_heads[l], c.head_dim,
             first_position, ring, c.rms_eps, model_.theta_scale(), sliding},
            stream_);
    AttentionArgs a{};
    a.q = q_;
    a.k = cache.k[l];
    a.v = cache.v[l];
    a.out = attn_;
    a.partials = partials_;
    a.rows = n;
    a.heads = c.num_heads;
    a.kv_heads = c.kv_heads[l];
    a.head_dim = c.head_dim;
    a.first_position = first_position;
    a.window = sliding ? c.sliding_window : 0U;
    a.ring = ring;
    a.key_limit = first_position + n;
    a.scale = scale;
    Attention(a, stream_);
    Project(w.attn_output, attn_, n, o_);
    // The sandwich norm sits on the branch, and the residual is added after it.
    RmsNormRows({o_, w.post_attn_norm.f32(), attn_, n, d, c.rms_eps}, stream_);
    AddRows({x_, attn_, static_cast<std::size_t>(n) * d}, stream_);

    RmsNormRows({x_, w.ffn_norm.f32(), h_, n, d, c.rms_eps}, stream_);
    SharedExpert(w, h_, n, shared_);
    Experts(w, h_, n, moe_out_, moe_weights_);
    MoeFinish({x_, moe_out_, shared_, moe_weights_, w.post_ffn_norm.f32(), n, d,
               c.experts_used, c.rms_eps},
              stream_);
  }

  RmsNormRows({x_, model_.output_norm().f32(), h_, n, d, c.rms_eps}, stream_);
  for (std::size_t i = 0; i < logit_rows.size(); ++i) {
    if (logit_rows[i] >= n) {
      throw std::invalid_argument("kolibri1 logit row outside the forward");
    }
    HIP_CHECK(hipMemcpyAsync(
        logit_x_ + i * d, h_ + static_cast<std::size_t>(logit_rows[i]) * d,
        d * sizeof(float), hipMemcpyDeviceToDevice, stream_));
  }
  Project(model_.output(), logit_x_, static_cast<std::uint32_t>(logit_rows.size()),
          logits_);
}

void Executor::CopyLogits(std::size_t rows, std::vector<float>* out) const {
  const std::size_t count = rows * model_.vocab_size();
  out->resize(count);
  // The forward is still in flight here, and its logits projection is the
  // widest of the model. The executor's stream is non-blocking and shares no
  // ordering with the default one, so a plain copy overtook it and returned a
  // half-written row; the untouched entries are zero, which is how token 0 won
  // a prompt whose true argmax was nowhere near.
  HIP_CHECK(hipMemcpyAsync(out->data(), logits_, count * sizeof(float),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

}  // namespace gufo::models::kolibri1::rocm
