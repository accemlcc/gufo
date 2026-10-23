// Dumps the device top-k logits per position for a fixed token-id sequence:
// prefill the first ids, then decode one token at a time, printing the top
// eight ids with logits per frontier. The companion script compares this
// against llama-server /completion logprobs over the same prefixes
// (teacher-forced agreement rate), which is the quality gate beyond single
// argmax parity.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen35moe/engine.hpp"

namespace q35 = gufo::models::qwen35moe;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.gguf> <id> [<id> ...]\n", argv[0]);
    return 2;
  }
  std::string error;
  q35::ModelOptions options;
  options.max_context = 4096;
  options.prefill_chunk = 16;
  auto model = q35::Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  auto session = model->CreateSession(&error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids;
  for (int i = 2; i < argc; ++i) {
    ids.push_back(static_cast<std::int32_t>(std::stol(argv[i])));
  }
  constexpr std::uint32_t kTop = 8;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const std::int32_t tok = ids[k];
    session->Append({&tok, 1});
    std::span<const float> logits = session->logits();
    std::vector<std::uint32_t> order(logits.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
    std::printf("%zu", k + 1);
    for (std::uint32_t j = 0; j < kTop; ++j) {
      std::printf(" %u:%.4f", order[j], logits[order[j]]);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
