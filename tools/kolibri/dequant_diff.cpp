// Compare gufo's IQ4 dequantizer against ggml's on a real artifact.
//
// A quantized weight is the one place where two implementations have no
// latitude: the stored bits fix the value, so a correct dequantizer agrees with
// ggml bit for bit. A nonzero difference here is a real bug, and it lands in
// every projection of every layer that uses the type - which is why a small
// number here turns into a large logit difference after 50 layers.
//
// Ground truth is ggml's own reference dequantizer (`dequantize_row_iq4_*`),
// not a reimplementation.
//
// Usage: dequant_diff <model.gguf> [tensor-name-substring ...]
// Build: tools/kolibri/build_dequant_diff.sh
#include <ggml.h>
#include <ggml-quants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace {

[[noreturn]] void Die(const std::string& message) {
  std::fprintf(stderr, "dequant_diff: %s\n", message.c_str());
  std::exit(1);
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct Entry {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type{0};
  std::uint64_t offset{0};
};

/// gufo's side of the comparison, dispatched over the quantized types.
bool GufoDequant(std::uint32_t type, const void* src, float* dst,
                 std::size_t cols) {
  switch (type) {
    case GGML_TYPE_Q4_K:
      gufo::quant::DequantizeQ4_K(src, dst, cols);
      return true;
    case GGML_TYPE_Q6_K:
      gufo::quant::DequantizeQ6_K(src, dst, cols);
      return true;
    case GGML_TYPE_Q8_0:
      gufo::quant::DequantizeQ8_0(src, dst, cols);
      return true;
    case GGML_TYPE_IQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, dst, cols);
      return true;
    case GGML_TYPE_IQ4_NL:
      gufo::quant::DequantizeIQ4_NL(src, dst, cols);
      return true;
    default:
      return false;
  }
}

void GgmlDequant(std::uint32_t type, const void* src, float* dst,
                 std::size_t cols) {
  const int64_t k = static_cast<int64_t>(cols);
  switch (type) {
    case GGML_TYPE_Q4_K:
      dequantize_row_q4_K(static_cast<const block_q4_K*>(src), dst, k);
      break;
    case GGML_TYPE_Q6_K:
      dequantize_row_q6_K(static_cast<const block_q6_K*>(src), dst, k);
      break;
    case GGML_TYPE_Q8_0:
      dequantize_row_q8_0(static_cast<const block_q8_0*>(src), dst, k);
      break;
    case GGML_TYPE_IQ4_XS:
      dequantize_row_iq4_xs(static_cast<const block_iq4_xs*>(src), dst, k);
      break;
    default:
      dequantize_row_iq4_nl(static_cast<const block_iq4_nl*>(src), dst, k);
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf> [substring ...]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const std::vector<std::string> needles(argv + 2, argv + argc);

  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    Die("cannot open " + path);
  }
  auto Take = [&](void* dst, std::size_t n) {
    if (std::fread(dst, 1, n, file) != n) {
      Die("truncated header");
    }
  };
  char magic[4];
  Take(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) {
    Die("not a GGUF file");
  }
  std::uint32_t version = 0;
  std::uint64_t n_tensors = 0;
  std::uint64_t n_kv = 0;
  Take(&version, 4);
  Take(&n_tensors, 8);
  Take(&n_kv, 8);

  auto read_string = [&]() {
    std::uint64_t length = 0;
    Take(&length, 8);
    std::string value(length, '\0');
    Take(value.data(), length);
    return value;
  };
  std::function<void(std::uint32_t, int)> skip_value;
  skip_value = [&](std::uint32_t kind, int depth) {
    if (depth > 2) {
      Die("metadata nesting too deep");
    }
    static const std::size_t kSizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 8, 8, 8, 8};
    if (kind == 8) {
      (void)read_string();
      return;
    }
    if (kind == 9) {
      std::uint32_t item = 0;
      std::uint64_t count = 0;
      Take(&item, 4);
      Take(&count, 8);
      for (std::uint64_t i = 0; i < count; ++i) {
        skip_value(item, depth + 1);
      }
      return;
    }
    if (kind > 11) {
      Die("unsupported metadata type");
    }
    std::fseek(file, static_cast<long>(kSizes[kind]), SEEK_CUR);
  };

  std::size_t alignment = 32;
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = read_string();
    std::uint32_t kind = 0;
    Take(&kind, 4);
    if (key == "general.alignment") {
      std::uint32_t value = 0;
      Take(&value, 4);
      alignment = value;
      continue;
    }
    skip_value(kind, 0);
  }

  std::vector<Entry> entries;
  entries.reserve(n_tensors);
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    Entry entry;
    entry.name = read_string();
    std::uint32_t rank = 0;
    Take(&rank, 4);
    for (std::uint32_t d = 0; d < rank; ++d) {
      std::uint64_t dim = 0;
      Take(&dim, 8);
      entry.dims.push_back(dim);
    }
    Take(&entry.type, 4);
    Take(&entry.offset, 8);
    entries.push_back(std::move(entry));
  }
  const std::size_t data_start =
      AlignUp(static_cast<std::size_t>(std::ftell(file)), alignment);

  int mismatches = 0;
  int checked = 0;
  for (const Entry& entry : entries) {
    if (entry.type != GGML_TYPE_Q4_K && entry.type != GGML_TYPE_Q6_K &&
        entry.type != GGML_TYPE_Q8_0 && entry.type != GGML_TYPE_IQ4_XS &&
        entry.type != GGML_TYPE_IQ4_NL) {
      continue;
    }
    bool hit = needles.empty();
    for (const std::string& needle : needles) {
      hit = hit || entry.name.find(needle) != std::string::npos;
    }
    if (!hit) {
      continue;
    }
    const std::size_t cols =
        entry.dims.empty() ? 0 : static_cast<std::size_t>(entry.dims[0]);
    const std::size_t rows_total =
        entry.dims.size() < 2
            ? 1
            : static_cast<std::size_t>(entry.dims[1]) *
                  (entry.dims.size() > 2 ? static_cast<std::size_t>(entry.dims[2])
                                         : 1);
    const std::size_t per_row = ggml_row_size(static_cast<ggml_type>(entry.type),
                                             static_cast<int64_t>(cols));
    if (cols == 0 || per_row == 0) {
      continue;
    }
    std::vector<std::uint8_t> raw(per_row);
    std::vector<float> mine(cols, 0.0F);
    std::vector<float> theirs(cols, 0.0F);
    double max_diff = 0.0;
    double scale = 0.0;
    std::size_t exact = 0;
    std::size_t wrong_value = 0;
    // A fused expert tensor holds every expert, so sampling the first rows
    // would compare one expert only: spread the samples over the tensor.
    const std::size_t samples = std::min<std::size_t>(8, rows_total);
    for (std::size_t s = 0; s < samples; ++s) {
      const std::size_t r = rows_total <= samples ? s : s * (rows_total - 1) / (samples - 1);
      const long long where =
          static_cast<long long>(data_start + entry.offset + r * per_row);
      if (std::fseek(file, where, SEEK_SET) != 0) {
        Die("seek failed for " + entry.name);
      }
      Take(raw.data(), per_row);
      if (!GufoDequant(entry.type, raw.data(), mine.data(), cols)) {
        Die("gufo has no dequantizer for type " + std::to_string(entry.type));
      }
      GgmlDequant(entry.type, raw.data(), theirs.data(), cols);
      for (std::size_t i = 0; i < cols; ++i) {
        const double a = mine[i];
        const double b = theirs[i];
        scale = std::max(scale, std::fabs(b));
        const double diff = std::fabs(a - b);
        if (diff == 0.0) {
          ++exact;
        } else {
          ++wrong_value;
        }
        max_diff = std::max(max_diff, diff);
      }
    }
    ++checked;
    const bool clean = max_diff == 0.0;
    if (!clean) {
      ++mismatches;
    }
    std::printf(
        "%-40s %-8s rows %zu/%zu exact %7.3f%% max|diff| %.4e  %s\n",
        entry.name.c_str(), ggml_type_name(static_cast<ggml_type>(entry.type)),
        samples, rows_total,
        100.0 * static_cast<double>(exact) / static_cast<double>(samples * cols),
        max_diff, clean ? "" : "<== MISMATCH");
    if (wrong_value != 0 && clean) {
      std::printf("  (values differ in sign/zero only: %zu)\n", wrong_value);
    }
  }
  std::fclose(file);
  std::printf("\n%d quantized tensor(s) checked, %d mismatch(es)\n", checked,
              mismatches);
  return mismatches == 0 ? 0 : 1;
}
