#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe::rocm {

/// The weight types the released artifacts carry. Q5_K holds the attention
/// gate/up projections, the shared-expert gate/up and the routed gate/up
/// halves; Q6_K the joint QKV, the down halves and the LM head; Q4_K appears
/// in the 122B UD file, IQ4_XS its sparsest routed gate/up experts. F32
/// carries every norm, the router, the shared gate vector and the SSM biases.
/// Anything else is refused at load.
enum class WeightFormat : std::uint32_t {
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_0,
  kIQ4_XS,
};

/// ---------------------------------------------------------------------------
/// Block arithmetic (blocks.hip.cpp)
/// ---------------------------------------------------------------------------

/// y[r] = rms(x[r]) * w, the rms over the n consecutive floats of a row.
struct NormRowsArgs {
  const float* x;
  const float* w;  ///< [n], null for an unweighted norm
  float* y;
  std::uint32_t rows;
  std::uint32_t n;
  float eps;
};
void RmsNormRows(const NormRowsArgs& args, hipStream_t stream);

/// x[i] += add[i].
struct AddRowsArgs {
  float* x;
  const float* add;
  std::size_t count;
};
void AddRows(const AddRowsArgs& args, hipStream_t stream);

/// act = silu(gate) * up over `count` elements.
struct GateUpArgs {
  const float* gate;
  const float* up;
  float* act;
  std::size_t count;
};
void GateUpAct(const GateUpArgs& args, hipStream_t stream);

/// out = attn * sigmoid(gate) over `count` elements: the full-attention gate
/// is a sigmoid, not the silu of the expert path.
struct SigmoidGateArgs {
  const float* attn;
  const float* gate;
  float* out;
  std::size_t count;
};
void SigmoidGateAct(const SigmoidGateArgs& args, hipStream_t stream);

/// Scales each row of `shared` in place by sigmoid(gate_vec . h[row]).
/// The dot product is one row of the gate vector against one normalized
/// layer input; the shared expert output waits for it before MoeCombine.
struct GateSharedArgs {
  float* shared;      ///< [rows][hidden], scaled in place
  const float* gate;  ///< [hidden], the shared-expert gate vector
  const float* h;     ///< [rows][hidden], the normalized layer input
  std::uint32_t rows;
  std::uint32_t hidden;
};
void GateShared(const GateSharedArgs& args, hipStream_t stream);

/// ---------------------------------------------------------------------------
/// Routing (route.hip.cpp)
///
/// The router is the Qwen3.5 recipe (`Qwen3_5MoeTopKRouter`): softmax over the
/// `ffn_gate_inp` logits, top-k, renormalise the kept weights to sum 1. No
/// bias, no rescale. Two launches carry a row to its experts - the logits,
/// then the selection over them - and a table over more than one block's rows
/// adds a third, exactly like the kolibri1 router whose shape this follows.
/// ---------------------------------------------------------------------------

/// Rows one grouped launch may carry, which is also the most slots one group
/// can list: an expert is chosen at most once per row.
inline constexpr std::uint32_t kMaxGroupSlots = 256;
/// Rows one block of the selection kernel holds.
inline constexpr std::uint32_t kSelectRows = 16;
/// Ints per group: the expert index, the slot count, then the slots.
inline constexpr std::uint32_t kGroupInts = 2 + kMaxGroupSlots;
/// Ints of a table with `groups` groups, including the leading count.
[[nodiscard]] constexpr std::size_t GroupTableInts(
    std::uint32_t groups) noexcept {
  return 1 + static_cast<std::size_t>(groups) * kGroupInts;
}

/// Experts a gathered table can carry: the gather gives one expert one thread
/// of one block.
inline constexpr std::uint32_t kGatherExperts = 1024;

struct MoeRouteArgs {
  const float* x;       ///< [rows][hidden], the normalized layer input
  const float* router;  ///< F32 [experts][hidden]
  float* logits;        ///< [rows][experts], the raw scores; may be null
  std::int32_t* ids;    ///< [rows][used]
  float* weights;       ///< [rows][used], renormalised top-k probabilities
  /// [experts] assignments, zeroed by the caller. Required whenever `groups`
  /// is asked for beyond `kSelectRows` rows: it is what the gather reads.
  std::uint32_t* counts;
  /// Group table GroupTableInts(min(experts, rows * used)), or null: one group
  /// per distinct expert the rows chose, its slots the assignments of that
  /// expert in increasing slot order, and no empty group. Up to
  /// `kMaxGroupSlots` rows.
  std::int32_t* groups;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t experts;
  std::uint32_t used;
};
void MoeRoute(const MoeRouteArgs& args, hipStream_t stream);

/// ---------------------------------------------------------------------------
/// Grouped projection (gemv.hip.cpp)
///
/// One kernel covers the routed experts and the ordinary projections: a dense
/// projection is the grouped one with a single group holding one expert and
/// the rows as its slots. The arithmetic of a slot must not depend on which
/// other slots share the launch with it, because a batch boundary changes the
/// grouping and the same token has to come out the same either way.
/// ---------------------------------------------------------------------------

/// Rows one grouped launch may carry, which is also the most slots one group
/// can list: an expert is chosen at most once per row.
inline constexpr std::uint32_t kMaxGroupedSlots = 256;

struct GroupedGemvArgs {
  WeightFormat format;
  /// [experts][m][k]: one matrix per expert, k contiguous, experts apart. With
  /// one expert this is a plain [m][k] projection.
  const void* w;
  /// Group table: a count, then per group the expert, its slot count and the
  /// slot indices in increasing order. A slot s reads x[r][0..k) with
  /// r = s / slots_per_row and writes y[s][0..m). Every listed group holds at
  /// least one slot.
  const std::int32_t* groups;
  std::uint32_t max_groups;  ///< launch bound; the table's count is <= this
  const float* x;            ///< [rows][k]
  float* y;                  ///< [slots][m]
  std::uint32_t rows;
  std::uint32_t slots_per_row;  ///< 8 for the routed gate/up, 1 for a dense
                                ///< projection and for the down half, whose
                                ///< input is the activation of the slot itself
  std::uint32_t m;
  std::uint32_t k;
};
/// False without launching when the shape or format is not supported.
[[nodiscard]] bool GroupedGemv(const GroupedGemvArgs& args, hipStream_t stream);

/// One row of the embedding table, dequantized: y[r][0..n) = table[token[r]].
struct EmbedArgs {
  const void* table;  ///< [vocabulary][n]
  WeightFormat format;
  const std::int32_t* tokens;
  float* y;  ///< [rows][n]
  std::uint32_t rows;
  std::uint32_t n;
};
void EmbedRows(const EmbedArgs& args, hipStream_t stream);

/// The end of a mixture block, per row r:
///   out[r] = shared_gated[r] + sum_j weights[r][j] * routed[(r * used + j)]
/// No normalization here: unlike kolibri the trunk adds the raw mixture to
/// the residual (transformers adds `mlp(...)` straight onto it, and so does
/// the graph). The caller adds `out` to the stream with AddRows.
struct MoeCombineArgs {
  float* out;            ///< [rows][hidden]
  const float* routed;   ///< [rows * used][hidden]
  const float* shared;   ///< [rows][hidden], gated shared-expert output
  const float* weights;  ///< [rows][used]
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t used;
};
void MoeCombine(const MoeCombineArgs& args, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
