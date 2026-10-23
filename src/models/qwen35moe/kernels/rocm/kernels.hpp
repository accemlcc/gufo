#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe::rocm {

/// The weight types the released artifacts carry. Q5_K holds the attention
/// gate/up projections, the shared-expert gate/up and the routed gate/up
/// halves; Q6_K the joint QKV, the down halves and the LM head; Q4_K appears
/// in the 122B UD file. F32 carries every norm, the router, the shared gate
/// vector and the SSM biases. Anything else is refused at load.
enum class WeightFormat : std::uint32_t {
  kQ4_K,
  kQ5_K,
  kQ6_K,
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

/// The end of a mixture block, per row r:
///   mix[r] = shared_gated[r] + sum_j weights[r][j] * routed[(r * used + j)]
///   x[r] += rms(mix[r]) * post_norm
/// The shared expert enters already gated by sigmoid(gate_vec . h); the norm
/// sits on the branch, not on the residual.
struct MoeFinishArgs {
  float* x;
  const float* routed;  ///< [rows * used][hidden]
  const float* shared;  ///< [rows][hidden], gated shared-expert output
  const float* weights;  ///< [rows][used]
  const float* post_norm;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t used;
  float eps;
};
void MoeFinish(const MoeFinishArgs& args, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
