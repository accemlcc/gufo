// Drives the qwen35moe scalar oracle over GGUF token ids and prints the
// frontier distribution after every token: one line per frontier,
// "<pos> id:logit ..." for the top 5. The expected distributions come from
// an independent implementation (llama-server on the same artifact); the
// comparison itself lives in tools/qwen35moe/compare_parity.py.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/reference.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace q35 = gufo::models::qwen35moe;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.gguf> <id> [<id> ...]\n", argv[0]);
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "cannot open %s: %s\n", argv[1], error.c_str());
    return 1;
  }
  const auto weights = q35::ModelWeights::LoadFromGguf(*reader, &error);
  if (!weights.has_value()) {
    std::fprintf(stderr, "artifact rejected: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids;
  for (int i = 2; i < argc; ++i) {
    ids.push_back(static_cast<std::int32_t>(std::stol(argv[i])));
  }
  const std::uint64_t vocab =
      weights->token_embd.num_elements / weights->config.hidden_size;
  q35::Reference ref(*weights);
  for (std::size_t k = 0; k < ids.size(); ++k) {
    std::vector<float> logits;
    const std::int32_t tok = ids[k];
    ref.Forward({&tok, 1}, &logits);
    std::vector<std::pair<float, std::uint64_t>> top;
    for (std::uint64_t v = 0; v < vocab; ++v) {
      top.emplace_back(logits[v], v);
    }
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const auto& a, const auto& b) {
                        return a.first > b.first;
                      });
    std::printf("%zu", k + 1);
    for (int j = 0; j < 5; ++j) {
      std::printf(" %llu:%.6f", (unsigned long long)top[j].second,
                  top[j].first);
    }
    std::printf("\n");
    std::fflush(stdout);
    if (const char* dump = std::getenv("QWEN35MOE_DUMP_LOGITS")) {
      std::string path =
          std::string(dump) + "/k" + std::to_string(k + 1) + ".f32";
      FILE* f = std::fopen(path.c_str(), "wb");
      std::fwrite(logits.data(), sizeof(float), logits.size(), f);
      std::fclose(f);
    }
  }
  return 0;
}
