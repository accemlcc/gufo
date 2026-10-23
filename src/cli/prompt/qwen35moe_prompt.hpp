#ifndef GUFO_CLI_PROMPT_QWEN35MOE_PROMPT_HPP_
#define GUFO_CLI_PROMPT_QWEN35MOE_PROMPT_HPP_

#include <chrono>

#include "src/cli/prompt/prompt.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::cli {

[[nodiscard]] bool IsQwen35Moe(const core::GgufReader& reader);

/// `gufo prompt` and `gufo chat` for Ornith-1.5 targets (ROCm only).
[[nodiscard]] int RunQwen35MoePrompt(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);
[[nodiscard]] int RunQwen35MoeChat(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::cli

#endif  // GUFO_CLI_PROMPT_QWEN35MOE_PROMPT_HPP_
