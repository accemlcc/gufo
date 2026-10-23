#ifndef GUFO_MODELS_QWEN35MOE_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_QWEN35MOE_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::qwen35moe {

/// Which released chat template the file carries. The 35B and 122B artifacts
/// share the Qwen3.5 ChatML core (im_start/im_end framing, always-on think
/// block, tool_response wrapping) but differ at the edges: the 35B template
/// merges up to two leading system/developer messages, the 122B one renders a
/// lone leading system message and rejects anything else outside user,
/// assistant and tool roles.
enum class TemplateVariant : std::uint8_t {
  kMergedSystem,  ///< Ornith-1.5-35B-A3B file.
  kSingleSystem,  ///< Unsloth 122B UD-Q4_K_XL file.
};

/// Rendering controls. Reasoning stays on unless the caller opts out, which
/// is what the template's own `else` branch does.
struct ChatOptions {
  bool add_generation_prompt{true};
  bool enable_thinking{true};
};

struct RenderedPrompt {
  std::string text;
  /// Byte offset where the generation prompt starts; equals text.size()
  /// without one. Everything before it is the stable conversation prefix.
  std::size_t generation_prompt_offset{0};
  /// Byte offset of the prefix the next request is expected to share: the
  /// generation prompt, or the end of the turn before a trailing text-only
  /// user turn.
  std::size_t stable_prefix_offset{0};
};

/// Compiled renderer of the Qwen3.5 chat template. The GGUF template is only
/// accepted when its SHA-256 matches one of the revisions qualified on the
/// artifacts, so a silently re-templated download fails at load instead of
/// producing a prompt the weights were never trained on.
///
/// Documented approximations of the Jinja sources:
/// - text-only messages; image or video parts are refused because the engine
///   has no vision path, as are tool definitions (the serve layer passes an
///   empty tool list, the way it does when the tool choice is none);
/// - the Jinja `|trim` is an ASCII whitespace strip, which agrees with the
///   template engine on the ASCII framing the renderer compares;
/// - an assistant message carries its thought in `thought`, the field the
///   server fills from `reasoning`/`reasoning_content` or by splitting a
///   `<think>` pair out of the content, as the template's own fallback does.
class ChatTemplate {
public:
  /// SHA-256 of `tokenizer.chat_template` on the 35B file.
  [[nodiscard]] static constexpr std::string_view TemplateSha25635B() noexcept {
    return "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b";
  }
  /// SHA-256 of `tokenizer.chat_template` on the 122B shards.
  [[nodiscard]] static constexpr std::string_view
  TemplateSha256122B() noexcept {
    return "9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235";
  }

  /// Accepts the file's template and reports which variant it is.
  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, TemplateVariant* variant,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::optional<RenderedPrompt> Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
      TemplateVariant variant, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_CHAT_TEMPLATE_HPP_
