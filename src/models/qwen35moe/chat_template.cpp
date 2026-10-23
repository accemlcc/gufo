// Qwen3.5-MoE's compiled chat template: the ChatML framing both released
// artifacts share, with the per-file system/developer edge each one pins by
// SHA. Text path only: image parts and tool definitions are refused because
// the engine serves neither, the way the Jinja sources raise on content the
// model was never trained to read.
#include "src/models/qwen35moe/chat_template.hpp"

#include <string>
#include <string_view>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen35moe {
namespace {

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";

bool IsSpace(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

std::string Trim(std::string_view value) {
  std::size_t first = 0;
  while (first < value.size() && IsSpace(value[first])) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && IsSpace(value[last - 1])) {
    --last;
  }
  return std::string(value.substr(first, last - first));
}

std::string RStripNewlines(std::string_view value) {
  std::size_t last = value.size();
  while (last > 0 && value[last - 1] == '\n') {
    --last;
  }
  return std::string(value.substr(0, last));
}

std::string LStripNewlines(std::string_view value) {
  std::size_t first = 0;
  while (first < value.size() && value[first] == '\n') {
    ++first;
  }
  return std::string(value.substr(first));
}

bool IsSystemLike(tokenization::ChatRole role) noexcept {
  return role == tokenization::ChatRole::kSystem ||
         role == tokenization::ChatRole::kDeveloper;
}

// The template's own assistant fallback: an explicit thought wins, otherwise
// a <think> pair is split out of the content with newline-only trims.
void SplitThought(const tokenization::ChatMessage& message, std::string* out,
                  std::string* thought) {
  *out = message.content;
  if (!message.thought.empty()) {
    *thought = message.thought;
    return;
  }
  *thought = "";
  const std::size_t close = out->find(kThinkClose);
  if (close == std::string::npos) {
    return;
  }
  const std::string before = out->substr(0, close);
  const std::string after = out->substr(close + kThinkClose.size());
  const std::size_t open = before.rfind(kThinkOpen);
  std::string core = open == std::string::npos
                         ? before
                         : before.substr(open + kThinkOpen.size());
  *thought = Trim(RStripNewlines(LStripNewlines(core)));
  *out = LStripNewlines(after);
}

void AppendUserTurn(std::string* text, const std::string& content) {
  text->append(kImStart);
  text->append("user\n");
  text->append(content);
  text->append(kImEnd);
  text->append("\n");
}

void AppendAssistantTurn(std::string* text, const std::string& reasoning,
                         const std::string& content) {
  text->append(kImStart);
  text->append("assistant\n<think>\n");
  text->append(reasoning);
  text->append("\n</think>\n\n");
  text->append(content);
  text->append(kImEnd);
  text->append("\n");
}

}  // namespace

bool ChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                        TemplateVariant* variant,
                                        std::string* error_msg) {
  const auto text = reader.GetMetadataString("tokenizer.chat_template");
  if (!text.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF metadata missing 'tokenizer.chat_template'";
    }
    return false;
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(text->data());
  const std::string hash = crypto::Sha256Hex({begin, text->size()});
  if (hash == TemplateSha25635B()) {
    if (variant != nullptr) {
      *variant = TemplateVariant::kMergedSystem;
    }
    return true;
  }
  if (hash == TemplateSha256122B()) {
    if (variant != nullptr) {
      *variant = TemplateVariant::kSingleSystem;
    }
    return true;
  }
  if (error_msg != nullptr) {
    *error_msg =
        "qwen35moe chat template is not a qualified revision (sha256 " +
        hash.substr(0, 16) + "...)";
  }
  return false;
}

std::optional<RenderedPrompt> ChatTemplate::Render(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
    TemplateVariant variant, std::string* error_msg) {
  auto fail = [&](const std::string& what) -> std::optional<RenderedPrompt> {
    if (error_msg != nullptr) {
      *error_msg = what;
    }
    return std::nullopt;
  };
  if (messages.empty()) {
    return fail("qwen35moe chat needs at least one message");
  }
  if (!tools.empty()) {
    return fail("qwen35moe chat does not serve tool definitions");
  }
  for (const auto& message : messages) {
    if (!message.images.empty()) {
      return fail("qwen35moe chat does not serve image content");
    }
  }
  RenderedPrompt out;
  // Leading system block: the 35B file merges up to two system/developer
  // messages, the 122B file renders a lone leading system message.
  std::size_t first_turn = 0;
  if (variant == TemplateVariant::kMergedSystem) {
    if (!messages.empty() && IsSystemLike(messages[0].role)) {
      std::string merged = Trim(messages[0].content);
      first_turn = 1;
      if (messages.size() > 1 && IsSystemLike(messages[1].role)) {
        merged += "\n" + Trim(messages[1].content);
        first_turn = 2;
      }
      out.text += std::string(kImStart) + "system\n" + merged +
                  std::string(kImEnd) + "\n";
    }
  } else {
    if (!messages.empty() &&
        messages[0].role == tokenization::ChatRole::kSystem) {
      out.text += std::string(kImStart) + "system\n" +
                  Trim(messages[0].content) + std::string(kImEnd) + "\n";
      first_turn = 1;
    }
  }
  bool in_tool_group = false;
  // Start of the trailing text-only user turn, if the messages end with one:
  // agent clients replace that turn per request, so caching past it would
  // rebuild a prefix that never repeats.
  std::size_t trailing_user_turn = std::string::npos;
  for (std::size_t i = first_turn; i < messages.size(); ++i) {
    const auto& message = messages[i];
    const bool last = i + 1 == messages.size();
    if (message.role == tokenization::ChatRole::kUser) {
      if (in_tool_group) {
        out.text += std::string(kImEnd) + "\n";
        in_tool_group = false;
      }
      trailing_user_turn = out.text.size();
      AppendUserTurn(&out.text, Trim(message.content));
    } else if (message.role == tokenization::ChatRole::kAssistant) {
      if (in_tool_group) {
        out.text += std::string(kImEnd) + "\n";
        in_tool_group = false;
      }
      std::string content;
      std::string reasoning;
      SplitThought(message, &content, &reasoning);
      AppendAssistantTurn(&out.text, Trim(reasoning), content);
      trailing_user_turn = std::string::npos;
    } else if (message.role == tokenization::ChatRole::kTool) {
      trailing_user_turn = std::string::npos;
      if (!in_tool_group) {
        out.text += std::string(kImStart) + "user";
        in_tool_group = true;
      }
      out.text +=
          "\n<tool_response>\n" + Trim(message.content) + "\n</tool_response>";
      if (last) {
        out.text += std::string(kImEnd) + "\n";
        in_tool_group = false;
      } else if (messages[i + 1].role != tokenization::ChatRole::kTool) {
        out.text += std::string(kImEnd) + "\n";
        in_tool_group = false;
      }
    } else {
      // The 35B file skips the system/developer turns it merged above; the
      // 122B file rejects anything outside user, assistant and tool roles.
      if (variant == TemplateVariant::kSingleSystem) {
        return fail("qwen35moe chat rejects this message role");
      }
    }
  }
  if (in_tool_group) {
    out.text += std::string(kImEnd) + "\n";
  }
  out.stable_prefix_offset = trailing_user_turn == std::string::npos
                                 ? out.text.size()
                                 : trailing_user_turn;
  out.generation_prompt_offset = out.text.size();
  if (options.add_generation_prompt) {
    out.stable_prefix_offset =
        std::min(out.stable_prefix_offset, out.generation_prompt_offset);
    out.text += std::string(kImStart) + "assistant\n";
    if (!options.enable_thinking) {
      out.text += "<think>\n\n</think>\n\n";
    } else {
      out.text += std::string(kThinkOpen) + "\n";
    }
  }
  return out;
}

}  // namespace gufo::models::qwen35moe
