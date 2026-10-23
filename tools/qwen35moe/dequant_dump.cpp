// Dumps dequantized rows for cross-checking against tools/qwen35moe/dequant_check.py:
// one header line "<name> row=<r> ex=<e> cols=<c>" followed by the raw float
// row. Column count is dims[0]; expert tensors address expert e at
// e * erows * row_bytes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {
using gufo::core::GgufReader;
using gufo::core::GgmlType;

void DequantRow(GgmlType type, const void* src, float* dst,
                std::size_t cols) {
  switch (type) {
    case GgmlType::kF32:
      std::memcpy(dst, src, cols * sizeof(float));
      return;
    case GgmlType::kBF16: {
      const auto* s = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < cols; ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(s[i]) << 16;
        float f = 0.0F;
        std::memcpy(&f, &u, sizeof(float));
        dst[i] = f;
      }
      return;
    }
    case GgmlType::kF16: {
      const auto* s = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < cols; ++i) {
        dst[i] = gufo::quant::Fp16ToFloat(s[i]);
      }
      return;
    }
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, dst, cols);
      return;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, dst, cols);
      return;
    case GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, dst, cols);
      return;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, dst, cols);
      return;
    case GgmlType::kQ3_K:
      gufo::quant::DequantizeQ3_K(src, dst, cols);
      return;
    case GgmlType::kQ8_K:
      gufo::quant::DequantizeQ8_K(src, dst, cols);
      return;
    case GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(src, dst, cols);
      return;
    case GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, dst, cols);
      return;
    case GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(src, dst, cols);
      return;
    default:
      break;
  }
  std::fill(dst, dst + cols, 0.0F);
}

struct Req {
  const char* name;
  std::uint64_t row, cols, erows, expert;
};

void DumpRow(const GgufReader& reader, const Req& q) {
  const auto* t = reader.FindTensor(q.name);
  if (t == nullptr) {
    std::fprintf(stderr, "missing %s\n", q.name);
    return;
  }
  const std::size_t rb = gufo::quant::EncodedSizeBytes(t->type, q.cols);
  const std::uint8_t* base = static_cast<const std::uint8_t*>(t->data) +
                             q.expert * q.erows * rb;
  std::vector<float> out(q.cols);
  DequantRow(t->type, base + q.row * rb, out.data(), q.cols);
  double mean = 0, sq = 0, mn = out[0], mx = out[0];
  for (float v : out) {
    mean += v;
    sq += static_cast<double>(v) * v;
    mn = std::min<double>(mn, v);
    mx = std::max<double>(mx, v);
  }
  mean /= q.cols;
  const double std = std::sqrt(sq / q.cols - mean * mean);
  std::printf("%s row=%llu ex=%llu cols=%llu %.6f %.6f %.6f %.6f\n", q.name,
              (unsigned long long)q.row, (unsigned long long)q.expert,
              (unsigned long long)q.cols, mean, std, mn, mx);
  for (std::uint64_t i = 0; i < q.cols; ++i) {
    std::printf("%.6f%c", out[i], i + 1 == q.cols ? '\n' : ' ');
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 2;
  }
  std::string error;
  auto reader = GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  // cols is always dims[0]; erows is dims[1] for 3D expert stacks, 0 else.
  const Req reqs[] = {
      {"token_embd.weight", 760, 2048, 0, 0},
      {"token_embd.weight", 248319, 2048, 0, 0},
      {"output.weight", 19241, 2048, 0, 0},
      {"output.weight", 248319, 2048, 0, 0},
      {"blk.0.attn_qkv.weight", 0, 2048, 0, 0},
      {"blk.0.attn_qkv.weight", 8191, 2048, 0, 0},
      {"blk.0.attn_gate.weight", 0, 2048, 0, 0},
      {"blk.0.attn_gate.weight", 4095, 2048, 0, 0},
      {"blk.0.ssm_alpha.weight", 0, 2048, 0, 0},
      {"blk.0.ssm_beta.weight", 1, 2048, 0, 0},
      {"blk.0.ffn_gate_exps.weight", 0, 2048, 512, 0},
      {"blk.0.ffn_gate_exps.weight", 511, 2048, 512, 255},
      {"blk.0.ffn_down_exps.weight", 0, 512, 2048, 0},
      {"blk.0.ffn_down_exps.weight", 2047, 512, 2048, 255},
      {"blk.3.attn_q.weight", 0, 2048, 0, 0},
      {"blk.3.attn_q.weight", 8191, 2048, 0, 0},
      {"blk.3.attn_output.weight", 0, 4096, 0, 0},
      {"blk.3.attn_output.weight", 2047, 4096, 0, 0},
      {"blk.0.ffn_gate_shexp.weight", 0, 2048, 0, 0},
      {"blk.0.ffn_gate_shexp.weight", 511, 2048, 0, 0},
  };
  for (const auto& q : reqs) {
    DumpRow(*reader, q);
  }
  return 0;
}
