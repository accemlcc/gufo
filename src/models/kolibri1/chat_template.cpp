#include "src/models/kolibri1/chat_template.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::kolibri1 {
namespace {

/// The turn delimiters and the channel markers. `\u003c` and `\u003e` are the
/// brackets: spelling the tags that way keeps this file free of byte sequences
/// that are also special tokens of this vocabulary.
constexpr std::string_view kTurnStart = "<|";
constexpr std::string_view kTurnEnd = "|>";

[[nodiscard]] std::string Turn(std::string_view name) {
  return std::string(kTurnStart) + std::string(name) + std::string(kTurnEnd);
}

const std::string kImStart = Turn("im_start");
const std::string kImEnd = Turn("im_end");

/// The markers without the newlines the template adds around them, for the
/// `startswith`/`endswith` tests the Jinja source makes.
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kToolResponseOpen = "<tool_response>";
constexpr std::string_view kToolResponseClose = "</tool_response>";

/// The four reasoning instructions, byte for byte from the template. The
/// template always renders one of them, so even a request with no system
/// message gets a system block.
constexpr std::string_view kNoReasoning =
    "Reasoning is disabled. Proceed straight to answering according to the "
    "user's instructions.";
constexpr std::string_view kLowReasoning =
    "Reasoning effort is set to low. Think briefly through only the essential "
    "steps in the user's language, then proceed directly to the answer.";
constexpr std::string_view kMediumReasoning =
    "Reasoning effort is set to medium. Think through the task methodically in "
    "the user's language, check key assumptions, and provide a well-supported "
    "answer.";
constexpr std::string_view kHighReasoning =
    "Reasoning effort is set to high. Think carefully through the task in the "
    "user's language, validate key assumptions, consider plausible "
    "alternatives, and prioritize correctness and clarity.";

[[nodiscard]] std::string_view SentenceFor(ChatOptions::Effort effort) {
  switch (effort) {
    case ChatOptions::Effort::kDisabled:
      return kNoReasoning;
    case ChatOptions::Effort::kLow:
      return kLowReasoning;
    case ChatOptions::Effort::kMedium:
      return kMediumReasoning;
    case ChatOptions::Effort::kHigh:
      return kHighReasoning;
  }
  return kHighReasoning;
}

/// JSON the way Jinja's tojson writes it: object keys sorted, the separators of
/// Python's json module, and every character that is not printable ASCII - plus
/// the four markup characters the sandbox escapes - as backslash-u escapes. The
/// template rendered this form while the model was trained, so a schema that
/// keeps its incoming member order or its raw umlauts is not the byte sequence
/// the weights saw.
void AppendEscape(std::string& out, std::uint32_t codepoint) {
  static constexpr char kHex[] = "0123456789abcdef";
  out += "\\u";
  out += kHex[(codepoint >> 12) & 0xFU];
  out += kHex[(codepoint >> 8) & 0xFU];
  out += kHex[(codepoint >> 4) & 0xFU];
  out += kHex[codepoint & 0xFU];
}

void AppendJsonString(std::string& out, std::string_view text) {
  out += '"';
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto byte = static_cast<unsigned char>(text[i]);
    if (byte == '"') {
      out += "\\\"";
    } else if (byte == '\\') {
      out += "\\\\";
    } else if (byte == '\b') {
      out += "\\b";
    } else if (byte == '\f') {
      out += "\\f";
    } else if (byte == '\n') {
      out += "\\n";
    } else if (byte == '\r') {
      out += "\\r";
    } else if (byte == '\t') {
      out += "\\t";
    } else if (byte < 0x20U || byte == '<' || byte == '>' || byte == '&' ||
               byte == '\'') {
      AppendEscape(out, byte);
    } else if (byte < 0x80U) {
      out += static_cast<char>(byte);
    } else {
      // Non-ASCII goes out as an escape too. A byte that does not start a valid
      // sequence cannot be a codepoint, so it escapes as itself.
      std::uint32_t codepoint = 0;
      std::size_t extra = 0;
      if ((byte & 0xE0U) == 0xC0U) { codepoint = byte & 0x1FU; extra = 1; }
      else if ((byte & 0xF0U) == 0xE0U) { codepoint = byte & 0x0FU; extra = 2; }
      else if ((byte & 0xF8U) == 0xF0U) { codepoint = byte & 0x07U; extra = 3; }
      bool valid = extra > 0 && i + extra < text.size();
      for (std::size_t k = 1; valid && k <= extra; ++k) {
        const auto next = static_cast<unsigned char>(text[i + k]);
        if ((next & 0xC0U) != 0x80U) { valid = false; break; }
        codepoint = (codepoint << 6) | (next & 0x3FU);
      }
      if (!valid) { AppendEscape(out, byte); continue; }
      i += extra;
      if (codepoint > 0xFFFFU) {
        const std::uint32_t lowered = codepoint - 0x10000U;
        AppendEscape(out, 0xD800U + (lowered >> 10));
        AppendEscape(out, 0xDC00U + (lowered & 0x3FFU));
      } else {
        AppendEscape(out, codepoint);
      }
    }
  }
  out += '"';
}

void AppendJson(std::string& out, const json::Value& value) {
  if (value.is_null()) {
    out += "null";
  } else if (value.is_bool()) {
    out += value.as_bool() ? "true" : "false";
  } else if (value.is_number()) {
    const double number = value.as_double();
    if (std::isfinite(number) && std::trunc(number) == number &&
        std::fabs(number) < 9007199254740992.0) {
      // Python keeps an int an int: 3, not 3.0.
      out += std::to_string(static_cast<long long>(number));
    } else {
      char buffer[64];
      const auto converted =
          std::to_chars(buffer, buffer + sizeof(buffer), number);
      out.append(buffer, converted.ptr);
    }
  } else if (value.is_string()) {
    AppendJsonString(out, value.str());
  } else if (value.is_array()) {
    out += '[';
    for (std::size_t i = 0; i < value.items().size(); ++i) {
      if (i != 0) {
        out += ", ";
      }
      AppendJson(out, value.items()[i]);
    }
    out += ']';
  } else {
    std::vector<const json::Value::Member*> members;
    members.reserve(value.members().size());
    for (const auto& member : value.members()) {
      members.push_back(&member);
    }
    std::sort(members.begin(), members.end(),
              [](const json::Value::Member* a,
                 const json::Value::Member* b) { return a->first < b->first; });
    out += '{';
    for (std::size_t i = 0; i < members.size(); ++i) {
      if (i != 0) {
        out += ", ";
      }
      AppendJsonString(out, members[i]->first);
      out += ": ";
      AppendJson(out, members[i]->second);
    }
    out += '}';
  }
}

[[nodiscard]] std::string ToJson(const json::Value& value) {
  std::string out;
  AppendJson(out, value);
  return out;
}

/// The tool dict as the request gave it. The server keeps the validated HTTP
/// object verbatim; a tool without one - a fixture, or a definition built
/// inside the server - is assembled from its parts.
[[nodiscard]] json::Value ToolValue(const tokenization::ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    try {
      return json::parse(tool.definition_json);
    } catch (const std::exception&) {
      // Fall through to the parts; the server already validated them.
    }
  }
  json::Value parameters = json::Value::object();
  if (!tool.parameters_json.empty()) {
    try {
      json::Value parsed = json::parse(tool.parameters_json);
      if (parsed.is_object()) {
        parameters = std::move(parsed);
      }
    } catch (const std::exception&) {
    }
  }
  json::Value function = json::Value::object();
  function.append_member("name", json::Value(tool.name));
  function.append_member("description", json::Value(tool.description));
  function.append_member("parameters", std::move(parameters));
  json::Value value = json::Value::object();
  value.append_member("type", json::Value("function"));
  value.append_member("function", std::move(function));
  return value;
}

[[nodiscard]] std::string ToolJson(const tokenization::ChatTool& tool) {
  return ToJson(ToolValue(tool));
}

[[nodiscard]] std::string ToolsBlock(
    std::span<const tokenization::ChatTool> tools) {
  std::string out =
      "# Tools\n\nYou may call one or more functions to assist with the user "
      "query.\n\nYou are provided with function signatures within <tools>"
      "</tools> XML tags:\n<tools>";
  for (const auto& tool : tools) {
    out += '\n';
    out += ToolJson(tool);
  }
  out +=
      "\n</tools>\n\nFor each function call, return a json object with "
      "function name and arguments within <tool_call></tool_call> XML "
      "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
      "<args-json-object>}\n</tool_call>";
  return out;
}

/// The arguments of a replayed call, through the same renderer. A non-string
/// argument arrives as JSON text and is parsed back so a number keeps its type.
[[nodiscard]] std::string ArgumentsJson(
    const tokenization::ChatMessage::ToolCall& call) {
  json::Value object = json::Value::object();
  for (const auto& argument : call.arguments) {
    if (argument.is_string) {
      object.append_member(argument.name, json::Value(argument.value));
      continue;
    }
    try {
      object.append_member(argument.name, json::parse(argument.value));
    } catch (const std::exception&) {
      object.append_member(argument.name, json::Value(argument.value));
    }
  }
  return ToJson(object);
}

[[nodiscard]] std::string_view StripTrailingNewlines(std::string_view text) {
  while (!text.empty() && text.back() == '\n') {
    text.remove_suffix(1);
  }
  return text;
}

[[nodiscard]] std::string_view StripLeadingNewlines(std::string_view text) {
  while (!text.empty() && text.front() == '\n') {
    text.remove_prefix(1);
  }
  return text;
}

/// Where an assistant turn's thought and body are. The server fills `thought`
/// from `reasoning`/`reasoning_content`; when a client replays plain content
/// that still carries a think pair, the thought is cut out of it here, exactly
/// as the template's own fallback branch does.
struct AssistantTurn {
  std::string_view thought;
  std::string_view content;
};

[[nodiscard]] AssistantTurn SplitAssistant(
    const tokenization::ChatMessage& message) {
  const std::string_view raw = message.content;
  if (!message.thought.empty()) {
    return {/*thought=*/message.thought, /*content=*/raw};
  }
  const std::size_t close = raw.find(kThinkClose);
  if (close == std::string_view::npos) {
    return {/*thought=*/{}, /*content=*/raw};
  }
  const std::size_t open = raw.rfind(kThinkOpen, close);
  const std::size_t begin = open == std::string_view::npos
                                ? 0
                                : open + kThinkOpen.size();
  return {/*thought=*/
          StripLeadingNewlines(StripTrailingNewlines(raw.substr(begin,
                                                                close > begin
                                                                    ? close - begin
                                                                    : 0))),
          /*content=*/StripLeadingNewlines(raw.substr(close +
                                                     kThinkClose.size()))};
}

}  // namespace

ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning) {
  ChatOptions options;
  // An explicit effort wins over `enabled`, as it does in the Jinja source.
  if (reasoning.effort.has_value()) {
    switch (*reasoning.effort) {
      case ReasoningEffort::kMinimal:
      case ReasoningEffort::kLow:
        options.effort = ChatOptions::Effort::kLow;
        break;
      case ReasoningEffort::kMedium:
        options.effort = ChatOptions::Effort::kMedium;
        break;
      case ReasoningEffort::kHigh:
      case ReasoningEffort::kXHigh:
      case ReasoningEffort::kMax:
        options.effort = ChatOptions::Effort::kHigh;
        break;
    }
  } else if (reasoning.enabled.has_value() && !*reasoning.enabled) {
    options.effort = ChatOptions::Effort::kDisabled;
  }
  if (reasoning.preserve_thinking.has_value()) {
    options.preserve_thinking = *reasoning.preserve_thinking;
  }
  return options;
}

bool ChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg) {
  const auto source = reader.GetMetadataString("tokenizer.chat_template");
  if (!source.has_value() || source->empty()) {
    if (error_msg != nullptr) {
      *error_msg = "kolibri1 GGUF has no tokenizer.chat_template";
    }
    return false;
  }
  const std::string digest = crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(source->data()), source->size()));
  if (digest != TemplateSha256()) {
    if (error_msg != nullptr) {
      *error_msg = "chat template digest " + digest +
                   " is not the qualified revision " +
                   std::string(TemplateSha256());
    }
    return false;
  }
  return true;
}

std::optional<RenderedPrompt> ChatTemplate::Render(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
    std::string* error_msg) {
  (void)error_msg;
  RenderedPrompt rendered;
  std::string& out = rendered.text;

  // `ns.last_query_index`: the newest message that is a real question. A tool
  // response is sent as a user message but continues the model's own turn, so
  // a run of them at the end is skipped.
  std::size_t last_query_index = messages.empty() ? 0 : messages.size() - 1;
  bool multi_step_tool = !messages.empty();
  for (std::size_t i = messages.size(); i-- > 0;) {
    const auto& message = messages[i];
    const bool tool_response =
        message.content.starts_with(kToolResponseOpen) &&
        message.content.ends_with(kToolResponseClose);
    if (multi_step_tool && message.role == tokenization::ChatRole::kUser &&
        !tool_response) {
      multi_step_tool = false;
      last_query_index = i;
    }
  }

  // System block. The reasoning instruction lives here and the template always
  // has one, so this block is written even with no system message.
  const bool has_system =
      !messages.empty() &&
      (messages[0].role == tokenization::ChatRole::kSystem ||
       messages[0].role == tokenization::ChatRole::kDeveloper);
  out += kImStart;
  out += "system\n";
  if (has_system) {
    out += messages[0].content;
    // Server-authored framing follows the caller's system text and precedes the
    // sections the template writes itself.
    out += messages[0].framing_suffix;
    out += "\n\n";
  }
  out += "# Reasoning effort\n\n";
  out += SentenceFor(options.effort);
  if (!tools.empty()) {
    out += "\n\n";
    out += ToolsBlock(tools);
  }
  out += kImEnd;
  out += '\n';

  // Where each message's rendering begins, so the reusable prefix can stop at
  // the boundary of the trailing question turn.
  std::vector<std::size_t> turn_start(messages.size());
  for (std::size_t i = 0; i < messages.size(); ++i) {
    turn_start[i] = out.size();
    const auto& message = messages[i];
    const bool first = i == 0;
    if (message.role == tokenization::ChatRole::kUser ||
        (message.role == tokenization::ChatRole::kSystem && !first)) {
      out += kImStart;
      out += tokenization::ToString(message.role);
      out += '\n';
      out += message.content;
      out += message.framing_suffix;
      out += kImEnd;
      out += '\n';
    } else if (message.role == tokenization::ChatRole::kAssistant) {
      const AssistantTurn turn = SplitAssistant(message);
      out += kImStart;
      out += "assistant\n";
      // A turn after the last question is the model's current one and always
      // carries a thinking block; earlier ones only when the client asked to
      // preserve thoughts.
      if (i > last_query_index || options.preserve_thinking) {
        out += kThinkStart;
        if (!StripTrailingNewlines(turn.thought).empty()) {
          out += StripTrailingNewlines(turn.thought);
        }
        out += kThinkEnd;
      }
      out += StripLeadingNewlines(turn.content);
      for (std::size_t c = 0; c < message.tool_calls.size(); ++c) {
        const auto& call = message.tool_calls[c];
        // The template separates the calls from the text and from each other
        // with one newline, and never opens a call on an empty line at the
        // start of a turn.
        if ((c == 0 && !turn.content.empty()) || c > 0) {
          out += '\n';
        }
        out += kToolCallStart;
        out += "{\"name\": \"";
        out += call.name;
        out += "\", \"arguments\": ";
        out += ArgumentsJson(call);
        out += "}";
        out += kToolCallEnd;
      }
      out += kImEnd;
      out += '\n';
    } else if (message.role == tokenization::ChatRole::kTool) {
      const bool opens =
          first || messages[i - 1].role != tokenization::ChatRole::kTool;
      const bool closes =
          i + 1 == messages.size() ||
          messages[i + 1].role != tokenization::ChatRole::kTool;
      if (opens) {
        out += kImStart;
        out += "user";
      }
      out += '\n';
      out += kToolResponseStart;
      out += message.content;
      out += kToolResponseEnd;
      if (closes) {
        out += kImEnd;
        out += '\n';
      }
    }
  }

  // Everything rendered so far is the conversation prefix. An agent's next
  // request replaces the trailing question turn, because that is where it
  // inserts per-turn context, so the reusable prefix stops at that turn's
  // start. A tool response belongs to the model turn it answers, so a trailing
  // run of them leaves the boundary at the end of the render.
  std::size_t stable = out.size();
  if (!messages.empty() && last_query_index + 1 == messages.size() &&
      messages[last_query_index].role == tokenization::ChatRole::kUser) {
    stable = turn_start[last_query_index];
  }
  if (options.add_generation_prompt) {
    rendered.generation_prompt_offset = out.size();
    // The prompt itself is never part of the reusable prefix, and neither is a
    // trailing question the caller will rewrite.
    stable = std::min(stable, rendered.generation_prompt_offset);
    out += kImStart;
    out += "assistant\n";
    if (options.effort == ChatOptions::Effort::kDisabled) {
      // The model still opens and closes a thinking block. Emptying it in the
      // prompt is what keeps a disabled run from producing a thought.
      out += kThinkStart;
      out += kThinkEnd;
    }
    rendered.stable_prefix_offset = stable;
    return rendered;
  }
  rendered.stable_prefix_offset = stable;
  return rendered;
}

}  // namespace gufo::models::kolibri1
