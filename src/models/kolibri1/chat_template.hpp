#ifndef GUFO_MODELS_KOLIBRI1_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_KOLIBRI1_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::kolibri1 {

/// Rendering controls, after the provider-neutral reasoning options have been
/// resolved against this template's defaults.
struct ChatOptions {
  bool add_generation_prompt{true};
  /// The template always writes a reasoning instruction: effort `none` swaps
  /// it for the "go straight to the answer" sentence and suppresses the empty
  /// think block in the generation prompt. `kDisabled` stands for that effort.
  enum class Effort : std::uint8_t {
    kDisabled,
    kLow,
    kMedium,
    kHigh,
  };
  Effort effort{Effort::kHigh};
  /// Replay the thoughts of earlier assistant turns. Off means only turns after
  /// the last real user question keep a think block, which is what the model
  /// was trained to produce.
  bool preserve_thinking{false};
};

/// Maps `reasoning.enabled` and `reasoning.effort` onto the four levels this
/// template knows. An explicit effort wins over `enabled`, as it does in the
/// Jinja source; `minimal` collapses to `low`, `xhigh` and `max` to `high`.
[[nodiscard]] ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning);

struct RenderedPrompt {
  std::string text;
  /// Byte offset where the generation prompt starts; equals text.size()
  /// without one. Everything before it is the stable conversation prefix.
  std::size_t generation_prompt_offset{0};
  /// Byte offset of the prefix the next request is expected to share: the
  /// generation prompt, or the end of the turn before a trailing text-only
  /// user turn. Agent clients replace that turn per request, so caching past
  /// it would rebuild a prefix that never repeats.
  std::size_t stable_prefix_offset{0};
};

/// Compiled renderer of the Kolibri 1 chat template. The GGUF template is only
/// accepted when its SHA-256 matches the revision qualified on the artifact,
/// so a silently re-templated download fails at load instead of producing a
/// prompt the weights were never trained on.
///
/// Documented approximations of the Jinja source:
/// - tool objects render from `definition_json`, the verbatim HTTP object,
///   because `tojson` there is Python's JSON with `", "` and `": "` separators;
/// - an assistant message carries its thought in `thought`, the field the
///   server fills from `reasoning`/`reasoning_content` or by splitting a
///   `
/// ` pair out of the content, as the template's own
///   fallback does;
/// - JSON numbers that arrive as integral floats print without a decimal
///   point, which is what `tojson` does for an int and what the HTTP layer
///   already normalizes them to.
class ChatTemplate {
public:
  /// SHA-256 of `tokenizer.chat_template` on Hob-forge/Kolibri-1-GGUF.
  [[nodiscard]] static constexpr std::string_view TemplateSha256() noexcept {
    return "9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97";
  }
  [[nodiscard]] static constexpr std::string_view TemplateId() noexcept {
    return "kolibri1-9ba35d4b";
  }

  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<RenderedPrompt> Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
      std::string* error_msg = nullptr);
};

/// The reasoning channel. The closing tag is stored bare in the vocabulary
/// (id 127908); the newlines here are what the template writes around it.
inline constexpr std::string_view kThinkStart = "<think>\n";
inline constexpr std::string_view kThinkEnd = "\n</think>\n\n";
inline constexpr std::string_view kToolCallStart = "<tool_call>\n";
inline constexpr std::string_view kToolCallEnd = "\n</tool_call>";
inline constexpr std::string_view kToolResponseStart = "<tool_response>\n";
inline constexpr std::string_view kToolResponseEnd = "\n</tool_response>";

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_CHAT_TEMPLATE_HPP_
