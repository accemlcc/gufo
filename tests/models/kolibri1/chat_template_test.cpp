// Contract of the Kolibri 1 prompt renderer: the byte sequence a request turns
// into, checked against the released Jinja template rather than against this
// implementation's own structure. Brackets are written as escapes so that no
// string in this file is also a special token of the vocabulary.
#include "src/models/kolibri1/chat_template.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using gufo::ReasoningEffort;
using gufo::ReasoningOptions;
using gufo::models::kolibri1::ChatOptions;
using gufo::models::kolibri1::ChatTemplate;
using gufo::models::kolibri1::ResolveChatOptions;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTool;

constexpr char kLt = '\u003c';
constexpr char kGt = '\u003e';

std::string Tag(std::string_view name) {
  std::string out(1, kLt);
  out += name;
  out += kGt;
  return out;
}

/// The turn delimiters carry pipes around the role word; the channel markers do
/// not, so they use Tag above.
std::string TurnTag(std::string_view name) {
  std::string out(1, kLt);
  out += '|';
  out += name;
  out += '|';
  out += kGt;
  return out;
}

const std::string kStart = TurnTag("im_start");
const std::string kEnd = TurnTag("im_end");
const std::string kThinkOpen = Tag("think");
const std::string kThinkClose = Tag("/think");
const std::string kCallOpen = Tag("tool_call");
const std::string kCallClose = Tag("/tool_call");
const std::string kResponseOpen = Tag("tool_response");
const std::string kResponseClose = Tag("/tool_response");

const std::string kHigh =
    "Reasoning effort is set to high. Think carefully through the task in the "
    "user's language, validate key assumptions, consider plausible "
    "alternatives, and prioritize correctness and clarity.";
const std::string kDisabled =
    "Reasoning is disabled. Proceed straight to answering according to the "
    "user's instructions.";

int failures = 0;

bool dbg_on = false;

std::string Safe(std::string_view text);

/// Dumps a render with brackets swapped for square ones and newlines escaped,
/// so the transcript of a failed check shows the bytes without looking like a
/// control sequence to the terminal.
void Dump(const std::string& what, const std::string& text) {
  if (!dbg_on) {
    return;
  }
  std::cout << "---- " << what << "\n" << Safe(text) << "\n----\n";
}

std::string Safe(std::string_view text) {
  std::string out;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\n') {
      out += "\n";
    } else if (c == kLt) {
      out += '[';
    } else if (c == kGt) {
      out += ']';
    } else {
      out += c;
    }
  }
  return out;
}

void Check(const std::string& what, bool ok, std::string_view found = {},
           std::string_view needle = {}) {
  if (ok) {
    std::cout << "PASS " << what << '\n';
    return;
  }
  ++failures;
  std::cout << "FAIL " << what;
  if (!needle.empty()) {
    std::cout << "\n  needle: [" << Safe(needle) << ']';
  }
  if (!found.empty()) {
    std::cout << "\n  found:  [" << Safe(found) << ']';
  }
  std::cout << '\n';
}

void CheckEquals(const std::string& what, const std::string& expected,
                 const std::string& actual) {
  if (expected == actual) {
    std::cout << "PASS " << what << '\n';
    return;
  }
  ++failures;
  std::cout << "FAIL " << what << "\n  expected: [" << expected
            << "]\n  actual:   [" << actual << "]\n";
}

void CheckContains(const std::string& what, const std::string& haystack,
                   std::string_view needle) {
  Check(what, haystack.find(needle) != std::string::npos, haystack, needle);
}

ChatMessage Message(ChatRole role, std::string content) {
  ChatMessage message;
  message.role = role;
  message.content = std::move(content);
  return message;
}

std::string Render(const std::vector<ChatMessage>& messages,
                   const std::vector<ChatTool>& tools,
                   const ChatOptions& options) {
  const auto rendered =
      ChatTemplate::Render(messages, tools, options, nullptr);
  if (!rendered.has_value()) {
    ++failures;
    std::cout << "FAIL render returned no value\n";
    return {};
  }
  return rendered->text;
}

// A bare question still gets a system block: the template always writes the
// reasoning instruction, and that instruction lives in the system turn.
void TestAlwaysHasEffortSection() {
  const std::string expected = kStart + "system\n" +
                               "# Reasoning effort\n\n" + kHigh + kEnd + "\n" +
                               kStart + "user\nWarum ist der Himmel blau?" +
                               kEnd + "\n" + kStart + "assistant\n";
  CheckEquals("bare question renders effort section and generation prompt",
              expected,
              Render({Message(ChatRole::kUser, "Warum ist der Himmel blau?")}, {},
                     {}));
}

void TestSystemContentPrecedesSections() {
  const std::string expected =
      kStart + "system\nAntworte auf Deutsch.\n\n# Reasoning effort\n\n" +
      kHigh + kEnd + "\n" + kStart + "user\nHallo?" + kEnd + "\n" + kStart +
      "assistant\n";
  std::vector<ChatMessage> messages;
  messages.push_back(Message(ChatRole::kSystem, "Antworte auf Deutsch."));
  messages.push_back(Message(ChatRole::kUser, "Hallo?"));
  CheckEquals("caller system text is kept ahead of the effort section", expected,
              Render(messages, {}, {}));
}

// Effort none swaps the instruction and empties the thinking block in the
// prompt, which is what stops the model from opening a thought.
void TestDisabledEffort() {
  ChatOptions options;
  options.effort = ChatOptions::Effort::kDisabled;
  const std::string text = Render({Message(ChatRole::kUser, "1+1?")}, {}, options);
  CheckContains("disabled effort states reasoning is off", text, kDisabled);
  const std::string tail = kStart + "assistant\n" + kThinkOpen + "\n\n" +
                             kThinkClose + "\n\n";
  CheckEquals("disabled effort closes an empty think block", tail,
              text.substr(text.size() - tail.size()));
}

void TestEffortSentences() {
  const std::pair<ReasoningEffort, std::string> cases[] = {
      {ReasoningEffort::kMinimal, "Reasoning effort is set to low."},
      {ReasoningEffort::kLow, "Reasoning effort is set to low."},
      {ReasoningEffort::kMedium, "Reasoning effort is set to medium."},
      {ReasoningEffort::kHigh, "Reasoning effort is set to high."},
      {ReasoningEffort::kXHigh, "Reasoning effort is set to high."},
      {ReasoningEffort::kMax, "Reasoning effort is set to high."},
  };
  for (const auto& [effort, prefix] : cases) {
    ReasoningOptions reasoning;
    reasoning.effort = effort;
    const ChatOptions options = ResolveChatOptions(reasoning);
    const std::string text = Render({Message(ChatRole::kUser, "x")}, {}, options);
    CheckContains(std::string("effort level maps to its sentence: ") + prefix,
                  text, prefix);
  }
  // An explicit effort beats reasoning.enabled.
  ReasoningOptions reasoning;
  reasoning.enabled = true;
  reasoning.effort = ReasoningEffort::kMedium;
  CheckContains("explicit effort wins over enabled",
                Render({Message(ChatRole::kUser, "x")}, {},
                       ResolveChatOptions(reasoning)),
                "Reasoning effort is set to medium.");
  // Disabled with no effort at all.
  ReasoningOptions off;
  off.enabled = false;
  Check("reasoning.enabled false disables",
        ResolveChatOptions(off).effort == ChatOptions::Effort::kDisabled);
}

void TestToolsRenderAsToJson() {
  ChatTool tool;
  tool.name = "web_search";
  tool.description = "Sucht \u00e4hnliche <schnelle> & 'beste' Ergebnisse";
  tool.parameters_json =
      "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},"
      "\"required\":[\"query\"]}";
  tool.definition_json = "{\"function\":{\"description\":\"Sucht \u00e4hnliche "
                         "<schnelle> & 'beste' Ergebnisse\",\"name\":"
                         "\"web_search\",\"parameters\":{\"properties\":"
                         "{\"query\":{\"type\":\"string\"}},\"required\":"
                         "[\"query\"],\"type\":\"object\"}},\"type\":\"function\"}";
  const std::string text = Render({Message(ChatRole::kUser, "suche")}, {tool}, {});
  // tojson sorts members, keeps Python's separators, and escapes the four
  // markup characters plus everything outside ASCII.
  CheckContains(
      "tool object is sorted, separated and escaped like tojson", text,
      "{\"function\": {\"description\": \"Sucht \\u00e4hnliche \\u003cschnelle"
      "\\u003e \\u0026 \\u0027beste\\u0027 Ergebnisse\", \"name\": "
      "\"web_search\", \"parameters\": {\"properties\": {\"query\": {\"type\": "
      "\"string\"}}, \"required\": [\"query\"], \"type\": \"object\"}}, "
      "\"type\": \"function\"}");
  CheckContains("tools block opens with the instruction", text,
                "# Tools\n\nYou may call one or more functions");
  CheckContains("tools block closes with the call contract", text,
                "\n</tool_call>");
  CheckContains("tool section follows the effort section", text,
                "clarity.\n\n# Tools");
}

void TestToolCallReplay() {
  ChatMessage answer = Message(ChatRole::kAssistant, "");
  answer.thought = "ich suche";
  ChatMessage::ToolCall call;
  call.name = "web_search";
  call.arguments = {{/*name=*/"query", /*value=*/"Strix Halo", /*is_string=*/true},
                    {/*name=*/"limit", /*value=*/"3", /*is_string=*/false}};
  answer.tool_calls.push_back(call);
  std::vector<ChatMessage> messages;
  messages.push_back(Message(ChatRole::kUser, "suche was"));
  messages.push_back(answer);
  messages.push_back(Message(ChatRole::kTool, "Ergebnis: 42"));
  messages.push_back(Message(ChatRole::kUser, "danke"));
  const std::string text = Render(messages, {}, {});
  Dump("tool call replay", text);
  const std::string call_text =
      kCallOpen + "\n{\"name\": \"web_search\", \"arguments\": "
                  "{\"limit\": 3, \"query\": \"Strix Halo\"}}\n" + kCallClose;
  Check("call carries name and typed arguments", text.find(call_text) != std::string::npos, text,
        call_text);
  // One tool message shares a user turn with its siblings; the closing tag only
  // appears once, when the run of responses ends.
  const std::string response_turn = kStart + "user\n" + kResponseOpen +
                                    "\nErgebnis: 42\n" + kResponseClose +
                                    kEnd + "\n";
  Check("tool response opens a user turn", text.find(response_turn) != std::string::npos, text,
        response_turn);
  const std::string thought_block =
      kThinkOpen + "\nich suche\n" + kThinkClose + "\n\n";
  Check("a call turn the question already moved past drops its thought",
        text.find(thought_block) == std::string::npos &&
            text.find("ich suche") == std::string::npos,
        text, thought_block);
  ChatOptions replayed;
  replayed.preserve_thinking = true;
  const std::string kept = Render(messages, {}, replayed);
  Check("preserve_thinking replays the thought of the call turn",
        kept.find(thought_block) != std::string::npos, kept, thought_block);
}

// Thoughts of turns before the newest question only survive when asked for.
void TestPreserveThinking() {
  std::vector<ChatMessage> messages;
  messages.push_back(Message(ChatRole::kUser, "erste Frage"));
  ChatMessage first = Message(ChatRole::kAssistant, "erste Antwort");
  first.thought = "erster Gedanke";
  messages.push_back(first);
  messages.push_back(Message(ChatRole::kUser, "zweite Frage"));

  ChatOptions dropped;
  const std::string without = Render(messages, {}, dropped);
  Dump("no preserve", without);
  Check("earlier thought is dropped by default",
        without.find("erster Gedanke") == std::string::npos, without);
  Check("a silent past turn renders no think block at all",
        without.find(kThinkOpen) == std::string::npos, without, kThinkOpen);

  // The turn the model is in - after the newest question - always carries a
  // block, empty when the client has no thought to hand back.
  std::vector<ChatMessage> current;
  current.push_back(Message(ChatRole::kUser, "erste Frage"));
  current.push_back(Message(ChatRole::kAssistant, "antwort"));
  const std::string inside = Render(current, {}, dropped);
  const std::string empty_block = kThinkOpen + "\n\n" + kThinkClose + "\n\n";
  Check("the current turn keeps an empty think block",
        inside.find(empty_block) != std::string::npos, inside, empty_block);

  ChatOptions kept;
  kept.preserve_thinking = true;
  const std::string with = Render(messages, {}, kept);
  CheckContains("preserve_thinking replays the earlier thought", with,
                kThinkOpen + "\nerster Gedanke\n" + kThinkClose + "\n\n" +
                    "erste Antwort" + kEnd + "\n");
}

// A thought the client replays inside content is separated from the answer, as
// the template's own fallback does.
void TestThoughtSplitOutOfContent() {
  ChatMessage replay =
      Message(ChatRole::kAssistant,
              kThinkOpen + "\n\nkurz \u00fcberlegt\n" + kThinkClose +
                  "\n\nfertige Antwort");
  // The newest assistant turn is the one whose block is written, so this is the
  // shape where the fallback split is observable.
  std::vector<ChatMessage> messages;
  messages.push_back(Message(ChatRole::kUser, "frage"));
  messages.push_back(replay);
  const std::string text = Render(messages, {}, {});
  Dump("inline thought", text);
  const std::string lifted = kThinkOpen + "\nkurz \u00fcberlegt\n" +
                             kThinkClose + "\n\nfertige Antwort";
  Check("inline thought is lifted out of the content", text.find(lifted) != std::string::npos,
        text, lifted);
}

// Cache boundaries: a trailing question is the part an agent rewrites, so the
// reusable prefix has to stop before it.
void TestStablePrefixOffsets() {
  std::vector<ChatMessage> messages;
  messages.push_back(Message(ChatRole::kSystem, "Rolle"));
  messages.push_back(Message(ChatRole::kUser, "alte Frage"));
  ChatMessage answer = Message(ChatRole::kAssistant, "alte Antwort");
  answer.thought = "alt";
  messages.push_back(answer);
  messages.push_back(Message(ChatRole::kUser, "neue Frage"));

  const auto with_prompt =
      ChatTemplate::Render(messages, {}, {}, nullptr)->text;
  const auto rendered =
      ChatTemplate::Render(messages, {}, {}, nullptr);
  CheckEquals("generation prompt is the assistant header",
              kStart + "assistant\n",
              with_prompt.substr(rendered->generation_prompt_offset));
  const std::string stable = with_prompt.substr(0, rendered->stable_prefix_offset);
  Check("stable prefix stops before the trailing question",
        stable.find("neue Frage") == std::string::npos &&
            stable.find("alte Antwort") != std::string::npos,
        stable);
  // The next request sends a different question but the same prefix.
  messages.back().content = "völlig andere Frage";
  const auto next = ChatTemplate::Render(messages, {}, {}, nullptr);
  Check("prefix of the follow-up request is byte identical",
        next->text.compare(0, next->stable_prefix_offset, stable) == 0,
        next->text.substr(0, next->stable_prefix_offset));

  // Without a generation prompt, a conversation ending in a tool response is
  // stable to its end: that response belongs to the model turn.
  std::vector<ChatMessage> tool_turn;
  tool_turn.push_back(Message(ChatRole::kUser, "frage"));
  ChatMessage call_turn = Message(ChatRole::kAssistant, "");
  ChatMessage::ToolCall call;
  call.name = "web_search";
  call_turn.tool_calls.push_back(call);
  tool_turn.push_back(call_turn);
  tool_turn.push_back(Message(ChatRole::kTool, "antwort des tools"));
  ChatOptions no_prompt;
  no_prompt.add_generation_prompt = false;
  const auto rendered_tool =
      ChatTemplate::Render(tool_turn, {}, no_prompt, nullptr);
  Check("a trailing tool response leaves the prefix whole",
        rendered_tool->stable_prefix_offset == rendered_tool->text.size(),
        rendered_tool->text);
}

}  // namespace

int main() {
  dbg_on = std::getenv("KOLIBRI_DUMP") != nullptr;
  TestAlwaysHasEffortSection();
  TestSystemContentPrecedesSections();
  TestDisabledEffort();
  TestEffortSentences();
  TestToolsRenderAsToJson();
  TestToolCallReplay();
  TestPreserveThinking();
  TestThoughtSplitOutOfContent();
  TestStablePrefixOffsets();
  if (failures != 0) {
    std::cout << failures << " chat template checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 chat template matches the qualified revision\n";
  return 0;
}
