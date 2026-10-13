// Kolibri 1 single-user throughput on this device.
//
// Not a correctness check - kolibri1_device_forward_test owns that. This one
// answers what the engine costs: how long a prompt of N tokens takes to enter
// the cache and how fast one token follows it, on one session, greedy.
//
// Both numbers are read off the wall clock around `Session::Append`, which
// ends with a stream-ordered copy of the logits, so an Append that has
// returned has published its answer. The argmax between decode steps is host
// work and is excluded: the accumulated time is the sum of the Append calls,
// not of the loop around them.
//
// The prefill length is built from real text rather than random ids. Random
// ids would pick experts uniformly at random, which is not the mixture a
// sentence touches, and the routed weights are the bulk of what a forward
// reads.
//
// Usage: throughput <model.gguf> [prefill_tokens] [decode_steps]
//                  [prefill_chunk] [max_context]
//
// The engine's headers name hip types without including the runtime header,
// because every file of the engine is compiled by the HIP compiler. This one
// is a plain host program, so it says the word first.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/kolibri1/engine.hpp"

namespace k1 = gufo::models::kolibri1;

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

/// Real text repeated to exactly `want` tokens. Truncating is deliberate: the
/// length measured is the length asked for, and a prompt cut mid-sentence is
/// still a prompt.
std::vector<k1::TokenId> Prompt(const k1::Tokenizer& tokenizer,
                                std::size_t want) {
  const std::string line =
      "Die Hauptstadt von Deutschland ist Berlin, und der Fluss daran heißt "
      "Spree. Ein Modell, das diesen Satz oft liest, erkennt seine Muster ";
  std::vector<k1::TokenId> ids;
  while (ids.size() < want) {
    const auto part = tokenizer.Encode(line, false, true);
    ids.insert(ids.end(), part.begin(), part.end());
  }
  ids.resize(want);
  return ids;
}

std::int32_t Argmax(std::span<const float> logits) {
  const auto best = std::max_element(logits.begin(), logits.end());
  return static_cast<std::int32_t>(std::distance(logits.begin(), best));
}

std::size_t Argument(int argc, char** argv, int index, std::size_t fallback) {
  if (index >= argc) {
    return fallback;
  }
  return static_cast<std::size_t>(std::strtoull(argv[index], nullptr, 10));
}

}  // namespace

int main(int argc, char** argv) {
  const char* from_env = std::getenv("GUFO_KOLIBRI1_MODEL");
  const std::string path = argc > 1 ? argv[1]
                                    : (from_env != nullptr ? from_env : "");
  if (path.empty()) {
    std::cout << "usage: throughput <model.gguf> [prefill_tokens] "
                 "[decode_steps] [prefill_chunk] [max_context]\n";
    return 2;
  }
  const std::size_t prefill_tokens = Argument(argc, argv, 2, 512);
  const std::size_t decode_steps = Argument(argc, argv, 3, 64);
  const std::size_t chunk = Argument(argc, argv, 4, 16);
  const std::size_t max_context =
      Argument(argc, argv, 5, prefill_tokens + decode_steps + 64);
  if (max_context < prefill_tokens + decode_steps) {
    std::cout << "max_context must hold the prompt and the generated tokens\n";
    return 2;
  }

  std::string error;
  const auto started = Clock::now();
  auto model = k1::Model::Load(
      path, k1::ModelOptions{.max_context = static_cast<std::uint32_t>(
                                           max_context),
                             .prefill_chunk = static_cast<std::uint32_t>(chunk)},
      &error);
  const double load_ms = ElapsedMs(started, Clock::now());
  if (!model) {
    std::cout << "FAIL cannot load " << path << ": " << error << "\n";
    return 1;
  }
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cout << "FAIL cannot create a session: " << error << "\n";
    return 1;
  }
  std::cout << "model " << path << "\n"
            << "weights " << model->resident_bytes() / (1024U * 1024U)
            << " MiB in " << load_ms / 1000.0 << " s; context " << max_context
            << ", prefill chunk " << chunk << "\n";

  const std::vector<k1::TokenId> prompt =
      Prompt(model->tokenizer(), prefill_tokens);

  // One token first, outside every measurement: the first forward of a fresh
  // process pays for the module load and whatever the runtime does on demand,
  // and that is not what either number means.
  session->Append({prompt.data(), 1});
  session->Reset();

  const auto prefill_start = Clock::now();
  session->Append(prompt);
  const double prefill_ms = ElapsedMs(prefill_start, Clock::now());

  double decode_ms = 0.0;
  for (std::size_t step = 0; step < decode_steps; ++step) {
    const std::int32_t next = Argmax(session->logits());
    const auto step_start = Clock::now();
    session->Append({&next, 1});
    decode_ms += ElapsedMs(step_start, Clock::now());
  }

  const double pp = prompt.size() / (prefill_ms / 1000.0);
  const double tg = decode_steps / (decode_ms / 1000.0);
  std::cout << "prefill " << prompt.size() << " tokens: " << prefill_ms
            << " ms -> " << pp << " tok/s\n"
            << "decode  " << decode_steps << " tokens: " << decode_ms
            << " ms -> " << tg << " tok/s\n";
  return 0;
}
