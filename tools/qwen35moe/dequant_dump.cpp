// Dumps dequantized rows for cross-checking against tools/qwen35moe/dequant_check.py:
// each line is "<name> <row> <cols> <mean> <std> <min> <max>" followed by the
// raw float row on the next line (space separated).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {
using gufo::core::GgufReader;
using gufo::core::GgmlType;

void DumpRow(const GgufReader& reader, const std::string& name,
             std::uint64_t row, std::uint64_t cols, GgmlType type,
             const void* data) {
  const std::size_t rb = gufo::quant::EncodedSizeBytes(type, cols);
  std::vector<float> out(cols);
  const auto* src = static_cast<const std::uint8_t*>(data) + row * rb;
  switch (type) {
    case GgmlType::kF32:
      out.assign(static_cast<const float*>(data) + row * cols,
                 static_cast<const float*>(data) + (row + 1) * cols);
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out.data(), cols);
      break;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out.data(), cols);
      break;
    case GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, out.data(), cols);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out.data(), cols);
      break;
    default:
      std::fprintf(stderr, "unsupported %s\n", name.c_str());
      return;
  }
  double mean = 0, sq = 0, mn = out[0], mx = out[0];
  for (float v : out) {
    mean += v;
    sq += static_cast<double>(v) * v;
    mn = std::min<double>(mn, v);
    mx = std::max<double>(mx, v);
  }
  mean /= cols;
  const double std = std::sqrt(sq / cols - mean * mean);
  std::printf("%s %llu %llu %.6f %.6f %.6f %.6f\n", name.c_str(),
              (unsigned long long)row, (unsigned long long)cols, mean, std, mn,
              mx);
  for (std::uint64_t i = 0; i < cols; ++i) {
    std::printf("%.6f%c", out[i], i + 1 == cols ? '\n' : ' ');
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
  const struct Req {
    const char* name;
    std::uint64_t row, cols;
  } reqs[] = {
      {"token_embd.weight", 760, 2048},
      {"output.weight", 19241, 2048},
      {"blk.0.attn_qkv.weight", 0, 8192},
      {"blk.0.attn_gate.weight", 0, 4096},
      {"blk.0.ssm_alpha.weight", 0, 2048},
      {"blk.0.ssm_beta.weight", 1, 2048},
      {"blk.0.ffn_gate_exps.weight", 0, 2048},
      {"blk.0.ffn_down_exps.weight", 0, 512},
      {"blk.3.attn_q.weight", 0, 8192},
      {"blk.3.attn_output.weight", 0, 4096},
      {"blk.0.ffn_gate_shexp.weight", 0, 2048},
      {"blk.0.ssm_a", 0, 32},
      {"blk.0.ssm_dt.bias", 0, 32},
  };
  for (const auto& q : reqs) {
    const auto* t = reader->FindTensor(q.name);
    if (t == nullptr) {
      std::fprintf(stderr, "missing %s\n", q.name);
      continue;
    }
    DumpRow(*reader, q.name, q.row, q.cols, t->type, t->data);
  }
  return 0;
}
