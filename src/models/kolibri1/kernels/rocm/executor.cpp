#include "src/models/kolibri1/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/kolibri1/kernels/rocm/attention.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"
#include "src/models/kolibri1/kernels/rocm/qkv_post.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

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

/// Whether a projection of `rows` rows goes to the shared matrix-core GEMM
/// instead of the batch-invariant GEMV.
///
/// The GEMV is written for at most `kProjectRows` rows and reads a weight
/// matrix once per step of that many, so a wider block costs it a re-read; the
/// tiled GEMM reads it once for the whole block and pays for it by quantizing
/// its input to int8 per 32 values. Where the two cross is a measured thing,
/// not a property of the width, so it is `kMatrixCoreRows` and not one row past
/// the GEMV's own limit; a decode step and the tail below it keep the GEMV's
/// strict rounding.
///
/// The shared launcher covers the two formats of the released artifacts (its
/// own fallback returns without launching anything for a format it does not
/// know, which would leave `y` to whatever the scratch held) and wants a
/// reduction dimension its quantization blocks divide.
[[nodiscard]] bool MatrixCoreProjection(const DeviceTensor& w,
                                        std::uint32_t rows) noexcept {
  if (rows < kMatrixCoreRows) {
    return false;
  }
  return (w.type == core::GgmlType::kQ4_K || w.type == core::GgmlType::kQ6_K) &&
         w.cols % 256 == 0;
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
  // One projection's input rows as Q8_1 tiles. The output projection reads the
  // whole query width and every other one reads at most the hidden size, so the
  // wider of the two bounds the buffer for all of them.
  const std::size_t act_bytes =
      hip::QuantizedActivationBytes(max_rows_, std::max(d, q_dim));
  bytes += ArenaBytes(act_bytes, 1);
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
  act_q8_ = arena.Take<std::uint8_t>(act_bytes);
  act_q8_bytes_ = act_bytes;

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

  // Named like the deepseek runtime's GUFO_DEEPSEEK_ROCM_*_STAGE_PROFILE
  // switches. The events exist only while it is set, so the shipped path keeps
  // its launch sequence untouched.
  stage_profile_ =
      std::getenv("GUFO_KOLIBRI_ROCM_PREFILL_STAGE_PROFILE") != nullptr;
  if (stage_profile_) {
    stage_events_.resize(4U + 5U * c.num_layers);
  }
  // The routed stage's own launches, on a switch of its own: a decode step
  // carries six extra markers per layer, and the five-stage numbers above are
  // quoted from forwards that did not have them.
  if (std::getenv("GUFO_KOLIBRI_ROCM_ROUTED_STAGE_PROFILE") != nullptr) {
    routed_events_.resize(7U * c.num_layers);
  }
  for (std::vector<hipEvent_t>* ring : {&stage_events_, &routed_events_}) {
    for (hipEvent_t& event : *ring) {
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDefault));
    }
  }
}

Executor::~Executor() {
  for (hipEvent_t& event : stage_events_) {
    (void)hipEventDestroy(event);
  }
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
  if (MatrixCoreProjection(w, rows) &&
      hip::QuantizedActivationBytes(rows, k) <= act_q8_bytes_) {
    // One pass over the rows into tiles, then the whole block in one launch:
    // the weight matrix is read once for `rows` rows instead of once per
    // kProjectRows of them.
    hip::LaunchQuantizeActivationQ8_1FromFp32(x, act_q8_, rows, k, stream_);
    hip::LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, act_q8_, y, rows, m,
                                           k, stream_);
    return;
  }
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
  // One group table over as many rows as the table can hold, which at the
  // shipped prefill width is the whole forward.
  //
  // The step size is what decides how often a hot expert is read again: a
  // grouped launch reads every expert it touches once, and a step only knows
  // the slots of its own rows. At sixteen rows a hundred and twenty-eight ran
  // eight mixtures, and the stage measured the same 2.212 ms per token there as
  // at one mixture of a hundred and twenty-eight (2.218) - the traffic of a
  // step over the step's rows does not fall when the steps grow with the
  // forward. Over the forward the visits are the distinct experts of all the
  // rows together, and the cost per token falls with the width as every other
  // part of the model's does.
  //
  // The loop stays for a forward wider than the table: past kMaxGroupSlots the
  // pieces re-read, which is a slower prefill rather than a refused one.
  // The stage's time is six launches wide, and which of them it sits in is
  // what the second switch is for. One piece only: past `kMaxGroupSlots` the
  // pieces re-read experts, and a split of that would attribute a re-read to
  // whichever piece it happened to land in.
  const bool sub = !routed_events_.empty() && count <= kMaxGroupSlots;
  std::uint32_t mark = 0;
  const auto Mark = [&]() {
    if (sub) {
      HIP_CHECK(hipEventRecord(routed_events_[routed_marks_ + mark], stream_));
      ++mark;
    }
  };
  for (std::uint32_t base = 0; base < count; base += kMaxGroupSlots) {
    const std::uint32_t n = std::min(kMaxGroupSlots, count - base);
    const std::uint32_t slots = n * used;
    const float* x = h + static_cast<std::size_t>(base) * d;
    float* w = weights + static_cast<std::size_t>(base) * used;
    Mark();
    HIP_CHECK(hipMemsetAsync(moe_counts_, 0, experts * sizeof(std::uint32_t),
                             stream_));
    Mark();
    MoeRoute({x, l.router.f32(), l.router_bias.f32(), nullptr,
              moe_ids_ + static_cast<std::size_t>(base) * used, w,
              moe_counts_, moe_groups_, n, d, experts, used},
             stream_);
    Mark();
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
    Mark();
    Routed(l.up_exps, x,
           moe_up_ + static_cast<std::size_t>(base) * used * width, width, d,
           used);
    Mark();
    GateUpAct({moe_gate_ + static_cast<std::size_t>(base) * used * width,
               moe_up_ + static_cast<std::size_t>(base) * used * width,
               moe_act_ + static_cast<std::size_t>(base) * used * width,
               static_cast<std::size_t>(slots) * width},
              stream_);
    Mark();
    Routed(l.down_exps,
           moe_act_ + static_cast<std::size_t>(base) * used * width,
           routed + static_cast<std::size_t>(base) * used * d, d, width, 1);
    Mark();
  }
  routed_marks_ += mark;
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

  std::uint32_t mark = 0;
  const auto Stage = [&]() {
    if (stage_profile_) {
      HIP_CHECK(hipEventRecord(stage_events_[mark++], stream_));
    }
  };
  Stage();
  HIP_CHECK(hipMemcpyAsync(tokens_, tokens.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
  EmbedRows({model_.token_embd().data, model_.token_embd().quant(), tokens_, x_,
             n, d},
            stream_);
  Stage();
  routed_marks_ = 0;

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
    Stage();
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
    Stage();
    Project(w.attn_output, attn_, n, o_);
    // The sandwich norm sits on the branch, and the residual is added after it.
    RmsNormRows({o_, w.post_attn_norm.f32(), attn_, n, d, c.rms_eps}, stream_);
    AddRows({x_, attn_, static_cast<std::size_t>(n) * d}, stream_);
    Stage();

    RmsNormRows({x_, w.ffn_norm.f32(), h_, n, d, c.rms_eps}, stream_);
    SharedExpert(w, h_, n, shared_);
    Stage();
    Experts(w, h_, n, moe_out_, moe_weights_);
    MoeFinish({x_, moe_out_, shared_, moe_weights_, w.post_ffn_norm.f32(), n, d,
               c.experts_used, c.rms_eps},
              stream_);
    Stage();
  }

  RmsNormRows({x_, model_.output_norm().f32(), h_, n, d, c.rms_eps}, stream_);
  Stage();
  for (std::size_t i = 0; i < logit_rows.size(); ++i) {
    if (logit_rows[i] >= n) {
      throw std::invalid_argument("kolibri1 logit row outside the forward");
    }
    HIP_CHECK(hipMemcpyAsync(
        logit_x_ + i * d, h_ + static_cast<std::size_t>(logit_rows[i]) * d,
        d * sizeof(float), hipMemcpyDeviceToDevice, stream_));
  }
  Project(model_.output(), logit_x_,
          static_cast<std::uint32_t>(logit_rows.size()), logits_);
  Stage();
  if (!stage_profile_) {
    return;
  }
  // Marker 0 is the entry and 1 the end of the embedding; then five per layer,
  // in the order qkv, attention, output projection, shared expert, routed
  // experts; the last two close the output norm and the logits. Segment i
  // therefore belongs to stage (i - 1) % 5, and a single synchronization of the
  // closing marker covers the whole forward. Recording a marker costs about a
  // microsecond, so a decode step reads a few percent too high - the switch is
  // for prefill attribution, not for a tg number.
  constexpr std::uint32_t kStageCount = 5;
  const std::uint32_t layer_end = 1U + kStageCount * c.num_layers;
  const std::uint32_t last = 3U + kStageCount * c.num_layers;
  HIP_CHECK(hipEventSynchronize(stage_events_[last]));
  float acc[kStageCount] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  float embed = 0.0F;
  float out_norm = 0.0F;
  float head = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&embed, stage_events_[0], stage_events_[1]));
  for (std::uint32_t i = 1; i < layer_end; ++i) {
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, stage_events_[i], stage_events_[i + 1]));
    acc[(i - 1U) % kStageCount] += ms;
  }
  HIP_CHECK(hipEventElapsedTime(&out_norm, stage_events_[layer_end],
                                stage_events_[layer_end + 1]));
  HIP_CHECK(hipEventElapsedTime(&head, stage_events_[layer_end + 1],
                                stage_events_[last]));
  const float per = static_cast<float>(n);
  const float total = embed + out_norm + head + acc[0] + acc[1] + acc[2] +
                      acc[3] + acc[4];
  std::fprintf(stderr,
               "kolibri1 stage rows=%u embed=%.3f qkv=%.3f attn=%.3f "
               "oproj=%.3f shared=%.3f routed=%.3f out_norm=%.3f head=%.3f "
               "total=%.3f (ms/token)\n",
               n, embed / per, acc[0] / per, acc[1] / per, acc[2] / per,
               acc[3] / per, acc[4] / per, out_norm / per, head / per,
               total / per);

  // The routed stage's own launches, when
  // GUFO_KOLIBRI_ROCM_ROUTED_STAGE_PROFILE was set. `MoeFinish` runs after
  // `Experts` returns, so the ring cannot see it: `finish=` is what the stage
  // above holds that this one does not. A forward cut into pieces leaves the
  // ring short and prints nothing.
  if (routed_marks_ == 7U * c.num_layers) {
    float seg[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t l = 0; l < c.num_layers; ++l) {
      const hipEvent_t* e = routed_events_.data() + l * 7U;
      for (std::uint32_t i = 0; i < 6U; ++i) {
        float ms = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&ms, e[i], e[i + 1]));
        seg[i] += ms;
      }
    }
    float ring = 0.0F;
    for (const float s : seg) {
      ring += s;
    }
    std::fprintf(stderr,
                 "kolibri1 routed rows=%u memset=%.3f route=%.3f gate=%.3f "
                 "up=%.3f act=%.3f down=%.3f finish=%.3f (ms/token)\n",
                 n, seg[0] / per, seg[1] / per, seg[2] / per, seg[3] / per,
                 seg[4] / per, seg[5] / per, (acc[4] - ring) / per);
  }
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
