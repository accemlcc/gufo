// Qwen3.5-MoE's compiled chat template against Jinja2 ground truth: the
// fixtures are rendered from the released files' own templates by
// tools/qwen35moe/make_chat_fixtures.py, so the check pins the C++ port to
// the artifacts instead of to a second implementation of the same guess.
//
// Needs no device; without the fixture directory the check skips with 77.
#ifndef GUFO_QWEN35MOE_FIXTURES
#define GUFO_QWEN35MOE_FIXTURES "."
#endif

#include "src/models/qwen35moe/chat_template.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/models/qwen35moe/check.hpp"

namespace q35 = gufo::models::qwen35moe;
namespace tok = gufo::tokenization;
using qwen35moe_test::Require;

namespace {

int failures = 0;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
}

std::string Fixture(const std::string& name) {
  return std::string(GUFO_QWEN35MOE_FIXTURES) + "/chat/" + name + ".txt";
}

std::string ReadFile(const std::string& path, bool& ok) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    ok = false;
    return {};
  }
  std::string out;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
    out.append(buf, n);
  }
  std::fclose(f);
  ok = true;
  return out;
}

tok::ChatMessage Msg(tok::ChatRole role, const std::string& content,
                     const std::string& thought = {}) {
  tok::ChatMessage m;
  m.role = role;
  m.content = content;
  m.thought = thought;
  return m;
}

using tok::ChatRole;

void RenderCase(const std::string& name, q35::TemplateVariant variant,
                const std::vector<tok::ChatMessage>& messages,
                const q35::ChatOptions& options) {
  bool ok = false;
  const std::string want = ReadFile(Fixture(name), ok);
  if (!ok) {
    Report(("chat.missing_" + name).c_str(), false, "no fixture");
    return;
  }
  std::string error;
  std::vector<tok::ChatTool> tools;
  auto got =
      q35::ChatTemplate::Render(messages, tools, options, variant, &error);
  if (!got.has_value()) {
    Report(("chat." + name).c_str(), false, error);
    return;
  }
  Report(("chat." + name).c_str(), got->text == want,
         got->text == want ? "exact bytes"
                           : "got=" + std::to_string(got->text.size()) +
                                 " want=" + std::to_string(want.size()));
}

void RenderMatrix(q35::TemplateVariant variant, const std::string& tag) {
  q35::ChatOptions gen;
  q35::ChatOptions no_think;
  no_think.enable_thinking = false;
  q35::ChatOptions no_prompt;
  no_prompt.add_generation_prompt = false;
  const auto U = ChatRole::kUser;
  const auto S = ChatRole::kSystem;
  const auto A = ChatRole::kAssistant;
  const auto D = ChatRole::kDeveloper;
  const auto T = ChatRole::kTool;
  RenderCase(tag + "_user_only", variant, {Msg(U, "Hello world")}, gen);
  RenderCase(tag + "_system_user", variant,
             {Msg(S, "You are helpful."), Msg(U, "What is 2+2?")}, gen);
  RenderCase(tag + "_multiturn_thought", variant,
             {Msg(S, "Be concise."), Msg(U, "Hi"),
              Msg(A, "Hey there!", "greet back"), Msg(U, "Bye")},
             gen);
  RenderCase(tag + "_think_extract", variant,
             {Msg(U, "Think this over"),
              Msg(A, "<think>\npondering\n</think>\n\nDone.")},
             gen);
  RenderCase(tag + "_no_thinking", variant, {Msg(U, "Short answer")}, no_think);
  RenderCase(tag + "_no_gen_prompt", variant, {Msg(S, "S."), Msg(U, "Q?")},
             no_prompt);
  RenderCase(tag + "_tool_roundtrip", variant,
             {Msg(U, "How are you?"), Msg(A, "Let me check."), Msg(T, "fine"),
              Msg(T, "indeed"), Msg(U, "Thanks")},
             gen);
  if (variant == q35::TemplateVariant::kMergedSystem) {
    // The single-system template rejects these at render time instead, so
    // their fixtures exist for the merged variant only (covered as refusals
    // below).
    RenderCase(tag + "_double_system", variant,
               {Msg(S, "First."), Msg(S, "Second."), Msg(U, "Go")}, gen);
    RenderCase(tag + "_developer", variant, {Msg(D, "Dev note."), Msg(U, "Go")},
               gen);
  }
}

void Refusals(q35::TemplateVariant variant, const std::string& tag) {
  std::string error;
  std::vector<tok::ChatTool> no_tools;
  std::vector<tok::ChatMessage> empty;
  q35::ChatOptions gen;
  bool ok = !q35::ChatTemplate::Render(empty, no_tools, gen, variant, &error)
                 .has_value();
  Report(("chat." + tag + "_refuses_empty").c_str(), ok, error);
  std::vector<tok::ChatMessage> one = {
      Msg(ChatRole::kUser, "Hi"),
  };
  std::vector<tok::ChatTool> tools(1);
  tools[0].name = "f";
  error.clear();
  ok = !q35::ChatTemplate::Render(one, tools, gen, variant, &error).has_value();
  Report(("chat." + tag + "_refuses_tools").c_str(), ok, error);
  std::vector<tok::ChatMessage> image = {
      Msg(ChatRole::kUser, "Look"),
  };
  image[0].images.push_back({});
  error.clear();
  ok = !q35::ChatTemplate::Render(image, no_tools, gen, variant, &error)
            .has_value();
  Report(("chat." + tag + "_refuses_images").c_str(), ok, error);
  if (variant == q35::TemplateVariant::kSingleSystem) {
    std::vector<tok::ChatMessage> developer = {
        Msg(ChatRole::kDeveloper, "Dev note."),
        Msg(ChatRole::kUser, "Go"),
    };
    error.clear();
    ok = !q35::ChatTemplate::Render(developer, no_tools, gen, variant, &error)
              .has_value();
    Report(("chat." + tag + "_refuses_developer").c_str(), ok, error);
    std::vector<tok::ChatMessage> late_system = {
        Msg(ChatRole::kUser, "Hi"),
        Msg(ChatRole::kSystem, "Late."),
    };
    error.clear();
    ok = !q35::ChatTemplate::Render(late_system, no_tools, gen, variant, &error)
              .has_value();
    Report(("chat." + tag + "_refuses_late_system").c_str(), ok, error);
  }
}

void Validate(const std::string& path) {
  if (path.empty()) {
    std::cout << "SKIP: no model\n";
    return;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    Report("chat.validate.reader", false, error);
    return;
  }
  q35::TemplateVariant variant = q35::TemplateVariant::kMergedSystem;
  if (!q35::ChatTemplate::ValidateGgufTemplate(*reader, &variant, &error)) {
    Report("chat.validate.template", false, error);
    return;
  }
  const bool is_122b = path.find("122B") != std::string::npos;
  const auto want = is_122b ? q35::TemplateVariant::kSingleSystem
                            : q35::TemplateVariant::kMergedSystem;
  Report("chat.validate.template", variant == want,
         is_122b ? "122b single-system" : "35b merged-system");
}

}  // namespace

int main() {
  RenderMatrix(q35::TemplateVariant::kMergedSystem, "35b");
  Refusals(q35::TemplateVariant::kMergedSystem, "35b");
  RenderMatrix(q35::TemplateVariant::kSingleSystem, "122b");
  Refusals(q35::TemplateVariant::kSingleSystem, "122b");
  const char* model = std::getenv("GUFO_QWEN35MOE_MODEL");
  Validate(model != nullptr ? model : "");
  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
