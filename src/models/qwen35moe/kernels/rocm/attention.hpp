#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_ATTENTION_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_ATTENTION_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe::rocm {

/// Single-pass online-softmax grouped-query attention over the F32 cache.
/// One wave per (row, query head): it walks the cached keys of its head's
/// group once for the online softmax and a second time against the values,
/// so a row crosses the keys twice and nothing else. Correctness first: this
/// is the reference shape of the path, not the tiled one.
///
/// Rows are independent: row r of a chunk that starts at `first_position`
/// attends positions [0, first_position + r]. The GQA pairing is adjacent
/// groups, kv head `h / (heads / kv_heads)`, matching the training stack.
struct AttentionArgs {
  const float* q;        ///< [rows][heads][dim], normed and rotated queries
  const float* k_cache;  ///< [kv_heads][max_context][dim]
  const float* v_cache;  ///< [kv_heads][max_context][dim]
  float* out;            ///< [rows][heads * dim]
  std::uint32_t rows;
  std::uint32_t first_position;  ///< cache slot of row 0
  std::uint32_t max_context;     ///< cache capacity in positions
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t dim;  ///< head dim (256)
  float scale;        ///< 1 / sqrt(dim)
};
void Attention(const AttentionArgs& args, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_ATTENTION_HPP_
