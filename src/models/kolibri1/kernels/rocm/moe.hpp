#ifndef GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_MOE_HPP_

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace gufo::models::kolibri1::rocm {

/// The weight types the released artifacts carry. Q4_K holds the attention
/// projections, the gate and up projections of every expert and the embedding
/// table; Q6_K holds value projections, the down projection of every other
/// layer, and the output head. Nothing else appears, so nothing else is
/// implemented: an artifact with another type is refused at load rather than
/// converted on the way through.
enum class WeightFormat : std::uint32_t {
  kQ4_K,
  kQ6_K,
};

/// ---------------------------------------------------------------------------
/// Grouped projection (gemv.hip.cpp)
///
/// One kernel covers the routed experts and the ordinary projections: a dense
/// projection is the grouped one with a single group holding one expert and the
/// rows as its slots. The arithmetic of a slot must not depend on which other
/// slots share the launch with it, because a batch boundary changes the grouping
/// and the same token has to come out the same either way.
/// ---------------------------------------------------------------------------

/// Rows one grouped launch may carry, which is also the most slots one group
/// can list: an expert is chosen at most once per row. The bound is the
/// table's, not the kernel's - the selection kernel covers sixteen rows per
/// block, so a launch wider than that builds its table over the counts the
/// selection leaves behind (see `MoeRoute`).
///
/// It matters because a grouped launch reads the weights of every expert it
/// touches once per pass of its groups: cutting a prefill into narrower
/// mixtures than the forward makes the hot experts be read again in every
/// piece, which is what held the routed stage at the same cost per token from
/// sixteen rows to a hundred and twenty-eight. One table over the forward makes
/// the cost per token fall with the width, as it does for every other part of
/// the model - and it is where the width stops. Raised from 128 to 256 the
/// stage goes 1.279 -> 1.132 ms per token while its expert bytes halve
/// (287 -> 163 MB per token): 0.269 ms of it is `MoeRoute`, which walks every
/// row and so is flat per token at any width, and the three launches below it
/// are not bound by the bytes they save. Widening past the table would only
/// step the forward again, and a step re-reads
/// (`docs/models/kolibri-1/BENCHMARKS.md#forward-width-256-served`).
inline constexpr std::uint32_t kMaxGroupSlots = 256;
/// Rows one block of the selection kernel holds, so `kMaxGroupSlots` rows need
/// `kMaxGroupSlots / kSelectRows` blocks and a table built outside them.
inline constexpr std::uint32_t kSelectRows = 16;
/// Ints per group: the expert index, the slot count, then the slots.
inline constexpr std::uint32_t kGroupInts = 2 + kMaxGroupSlots;
/// Ints of a table with `groups` groups, including the leading count.
[[nodiscard]] constexpr std::size_t GroupTableInts(
    std::uint32_t groups) noexcept {
  return 1 + static_cast<std::size_t>(groups) * kGroupInts;
}

struct GroupedGemvArgs {
  WeightFormat format;
  /// [experts][m][k]: one matrix per expert, k contiguous, experts apart. With
  /// one expert this is a plain [m][k] projection.
  const void* w;
  /// Group table: a count, then per group the expert, its slot count and the
  /// slot indices in increasing order. A slot s reads x[r][0..k) with
  /// r = s / slots_per_row and writes y[s][0..m). Every listed group holds at
  /// least one slot, so the count is at most rows * slots_per_row and a caller
  /// may use that product as its launch bound; a group for an expert that was
  /// not chosen would push the count past the bound and the groups behind it
  /// would never start.
  const std::int32_t* groups;
  std::uint32_t max_groups;  ///< launch bound; the table's count is <= this
  const float* x;            ///< [rows][k]
  float* y;                  ///< [slots][m]
  std::uint32_t rows;
  std::uint32_t slots_per_row;  ///< slots sharing one input row: 6 for the
                                ///< routed gate/up, 1 for a dense projection
                                ///< and for the down half, whose input is the
                                ///< activation of the slot itself
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

/// ---------------------------------------------------------------------------
/// Routing (route.hip.cpp)
///
/// The router is the one part of this model that is not any of the common MoE
/// recipes: the bias moves the selection and nothing else, and the weight an
/// expert contributes is the unbiased sigmoid of its own logit, with no
/// renormalization over the chosen six and no softmax.
///
/// Two launches carry a row to its experts - the logits, then the selection
/// over them - and a table over more than one block's rows adds a third: the
/// selection's per-block table would only see its own sixteen rows, so past
/// that the selection publishes ids and counts and one block gathers the table
/// from them. The order the table has to have - experts increasing, a group's
/// slots increasing - is a property of the gather either way.
/// ---------------------------------------------------------------------------

/// Experts a gathered table can carry: the gather gives one expert one thread
/// of one block, so that block is the model's expert bound on the wide path. A
/// narrower launch builds its table inside the selection block and is not
/// limited this way.
inline constexpr std::uint32_t kGatherExperts = 1024;

struct MoeRouteArgs {
  const float* x;      ///< [rows][hidden], the normalized layer input
  const float* router; ///< F32 [experts][hidden]
  const float* bias;   ///< F32 [experts], added for the selection only
  float* logits;       ///< [rows][experts], the raw scores; may be null
  std::int32_t* ids;   ///< [rows][used]
  float* weights;      ///< [rows][used], sigmoid(logit) of the chosen expert
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

/// act = silu(gate) * up over `count` elements. Both experts and the shared
/// expert use this; the gate and up projections are separate tensors in this
/// artifact, so they are two products and one elementwise step rather than a
/// fused interleaved one.
struct GateUpArgs {
  const float* gate;
  const float* up;
  float* act;
  std::size_t count;
};
void GateUpAct(const GateUpArgs& args, hipStream_t stream);

/// The end of a mixture block, per row r:
///   mix[r] = shared[r] + sum_j weights[r][j] * routed[(r * used + j)]
///   x[r] += rms(mix[r]) * post_norm
/// The shared expert is ungated and enters with weight 1, and the sandwich norm
/// sits on the branch, not on the residual.
struct MoeFinishArgs {
  float* x;
  const float* routed;  ///< [rows * used][hidden]
  const float* shared;  ///< [rows][hidden]
  const float* weights;  ///< [rows][used]
  const float* post_norm;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t used;
  float eps;
};
void MoeFinish(const MoeFinishArgs& args, hipStream_t stream);

}  // namespace gufo::models::kolibri1::rocm

#endif  // GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_MOE_HPP_
