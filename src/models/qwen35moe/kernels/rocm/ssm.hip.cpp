// Qwen3.5-MoE's Gated DeltaNet token mixer: the depthwise causal convolution
// with SiLU, then the recurrent gated-delta update. The recurrence is the
// shared gated-delta kernel, instantiated here for the only shape this model
// uses (128x128 FP32 resident state); the convolution is the same four-tap
// causal filter with the same [qkv * 4] history layout.
//
// The K/V heads tile across the value heads (`h % num_key_heads`), which is
// what the released files store: the converter reorders the V side to tiled
// order, so the tiled pairing reads each head with its own key.
#include "src/models/qwen35moe/kernels/rocm/ssm.hpp"

#include <stdexcept>

#include "src/models/qwen/hip/kernels/deltanet_decode.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::models::qwen35moe::rocm {
namespace {

// Depthwise causal Conv1d-4 + SiLU over `rows` rows, updating the persistent
// four-deep history in place. One thread per channel; rows run in order
// inside the thread because each row's window reaches into the rows before
// it, which is exactly what sequential single-row launches would compute.
__global__ void SsmConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out,
                              std::uint32_t qkv_dim, std::uint32_t rows,
                              std::size_t row_stride) {
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim) {
    return;
  }
  const std::size_t c_off = c * 4;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float* row_in = qkv_in + static_cast<std::size_t>(row) * row_stride;
    float* row_out = conv_out + static_cast<std::size_t>(row) * row_stride;
    conv_state[c_off + 0] = conv_state[c_off + 1];
    conv_state[c_off + 1] = conv_state[c_off + 2];
    conv_state[c_off + 2] = conv_state[c_off + 3];
    conv_state[c_off + 3] = row_in[c];
    float dot = 0.0F;
    if (conv_weights != nullptr) {
      for (std::uint32_t k = 0; k < 4; ++k) {
        dot += conv_state[c_off + k] * conv_weights[c * 4 + k];
      }
    } else {
      dot = row_in[c];
    }
    const float sig = 1.0F / (1.0F + expf(-dot));
    row_out[c] = dot * sig;
  }
}

}  // namespace

void LaunchSsm(const SsmArgs& args, hipStream_t stream) {
  if (args.key_dim != 128 || args.head_dim != 128) {
    throw std::invalid_argument(
        "LaunchSsm supports the 128x128 resident state only");
  }
  if (args.rows == 0) {
    return;
  }
  const dim3 conv_grid((args.qkv_dim + 255) / 256);
  SsmConvKernel<<<conv_grid, 256, 0, stream>>>(
      args.qkv, args.conv_w, args.conv_state, args.conv_out, args.qkv_dim,
      args.rows, args.qkv_dim);
  gufo::hip::SsmReplayCapture no_replay;
  gufo::hip::DeltaNetRecurrenceKernel<float, true, true>
      <<<args.num_heads, 128, 0, stream>>>(
          args.conv_out, args.recurrent, args.alpha, args.beta, args.ssm_a,
          args.ssm_dt, args.ssm_norm, args.gate, args.out, 0,
          args.num_key_heads, args.num_heads, args.key_dim, args.value_dim,
          no_replay, args.rows, args.qkv_dim, args.num_heads, args.value_dim);
}

}  // namespace gufo::models::qwen35moe::rocm
