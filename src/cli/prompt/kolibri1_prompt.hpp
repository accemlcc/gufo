#ifndef GUFO_CLI_PROMPT_KOLIBRI1_PROMPT_HPP_
#define GUFO_CLI_PROMPT_KOLIBRI1_PROMPT_HPP_

#include <chrono>

#include "src/cli/prompt/prompt.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::cli {

[[nodiscard]] bool IsKolibri1(const core::GgufReader& reader);

/// `gufo prompt` and `gufo chat` for Kolibri 1 targets (ROCm only).
[[nodiscard]] int RunKolibri1Prompt(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);
[[nodiscard]] int RunKolibri1Chat(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::cli

#endif  // GUFO_CLI_PROMPT_KOLIBRI1_PROMPT_HPP_
