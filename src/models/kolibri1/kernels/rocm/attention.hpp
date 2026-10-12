#ifndef GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_ATTENTION_HPP_
#define GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_ATTENTION_HPP_

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace gufo::models::kolibri1::rocm {

/// Attention over a batch of query rows of one layer.
///
/// The cache layout matches the scalar oracle: a cache row is one position and
/// holds all key heads back to back, so the stride between positions is
/// kv_heads * head_dim. K arrives normed and, on a sliding layer, rotated; the
/// ten layers without positional encoding store their keys the same way and
/// simply never had them rotated, which the kernel cannot and need not tell.
///
/// Q stays FP32 and narrows to binary16 inside the kernel, which is the
/// precision the products run in; the accumulator, the softmax and the output
/// are FP32.
struct AttentionArgs {
  const float* q;             ///< [rows][heads][head_dim]
  const std::uint16_t* k;     ///< binary16 [capacity][kv_heads][head_dim]
  const std::uint16_t* v;     ///< binary16 [capacity][kv_heads][head_dim]
  float* out;                 ///< [rows][heads][head_dim]
  float* partials;            ///< split-K scratch, or null when splits is 1

  std::uint32_t rows;
  std::uint32_t heads;     ///< query heads, a multiple of the key heads
  std::uint32_t kv_heads;  ///< heads per key group is heads / kv_heads
  std::uint32_t head_dim;  ///< 128, the released geometry
  std::uint32_t first_position;
  /// Keys a sliding layer may read, including its own query. Zero means the
  /// layer is unwindowed and reads from the start of the cache.
  std::uint32_t window;
  /// Cache capacity in positions; the slot of a key is key % ring. Zero indexes
  /// the cache by position.
  std::uint32_t ring;
  /// Never read keys at or beyond this index, for a truncated continuation.
  std::uint32_t key_limit;
  float scale;

  /// Split-K: the key range is cut into `splits` runs, of which this launch
  /// computes number `split`. Set by the launcher, not by the caller.
  std::uint32_t splits{1};
  std::uint32_t split{0};
};

/// Attention over the given rows, which hold consecutive positions starting at
/// first_position. Rows past the end of a prefill batch are simply absent; the
/// batch is dense.
///
/// Throws std::invalid_argument for a geometry the kernel is not written for:
/// head_dim other than 128, a query-head count that is not a multiple of the
/// key heads, or a group other than twelve heads.
void Attention(const AttentionArgs& args, hipStream_t stream);

/// Query rows one prefill block covers.
[[nodiscard]] constexpr std::uint32_t AttentionQueryRowsPerBlock() noexcept {
  return 32;
}

/// Splits a decode launch chooses for a context of `keys` keys. One key head
/// alone is four blocks, which idles most of the GPU on a long context, so the
/// range is cut until there is work for every compute unit.
[[nodiscard]] std::uint32_t PickSplits(std::uint32_t keys) noexcept;

/// The largest split count `PickSplits` can return for a cache of `capacity`
/// positions, so a caller can size its scratch once.
[[nodiscard]] std::uint32_t MaxSplits(std::uint32_t capacity) noexcept;

/// FP32 words the split-K scratch needs: one partial output plus its max and
/// sum per row, head and split.
[[nodiscard]] std::size_t AttentionPartialFloats(std::uint32_t rows,
                                                 std::uint32_t heads,
                                                 std::uint32_t head_dim,
                                                 std::uint32_t capacity) noexcept;

}  // namespace gufo::models::kolibri1::rocm

#endif  // GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_ATTENTION_HPP_
