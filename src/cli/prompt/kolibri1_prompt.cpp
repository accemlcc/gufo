#include "src/cli/prompt/kolibri1_prompt.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/tokenization/chat_tokenization.hpp"
#include "src/core/sampling.hpp"
#include "src/models/kolibri1/chat_template.hpp"
#include "src/models/kolibri1/engine.hpp"

namespace gufo::cli {
namespace {

namespace k1 = gufo::models::kolibri1;

/// The context one conversation may use. The cache holds full key/value rows
/// for every position - the global layers attend over all of them - so this is
/// the number that decides how much memory a session costs, and 8k of it is
/// under a gigabyte against 47 GiB of weights.
constexpr std::uint32_t kKolibriContext = 8192;

double Elapsed(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

[[nodiscard]] std::unique_ptr<k1::Model> LoadModel(
    const PromptOptions& opt, std::chrono::steady_clock::time_point load_start) {
  std::string error;
  const auto model = k1::Model::Load(opt.model_path,
                                     k1::ModelOptions{
                                         .max_context = kKolibriContext,
                                         .prefill_chunk = 16,
                                     },
                                     &error);
  if (!model) {
    std::cerr << "Kolibri 1 load failed: " << error << '\n';
    return nullptr;
  }
  if (opt.verbose) {
    std::cout << "Model loaded in " << Elapsed(load_start) << " s; "
              << model->resident_bytes() / (1024U * 1024U)
              << " MiB of weights on the device\n";
  }
  return model;
}

/// Tokenizes a rendered prompt. Control tokens in the template text are matched
/// as specials, which is what makes the role markers single tokens.
[[nodiscard]] std::vector<k1::TokenId> TokenizePrompt(const k1::Model& model,
                                                      const std::string& text) {
  return model.tokenizer().Encode(text, true, true);
}

/// Prefills `prompt` and generates until the model stops or `opt.max_tokens`
/// are out. `reply` collects the text of the turn when the caller keeps history.
int Generate(const PromptOptions& opt, const k1::Model& model,
             k1::Session& session, std::span<const k1::TokenId> prompt,
             std::string* reply) {
  if (prompt.empty()) {
    std::cerr << "Kolibri 1 prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > model.max_context()) {
    std::cerr << "Kolibri 1 prompt and output exceed the "
              << model.max_context() << "-token CLI context\n";
    return 1;
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  // A turn is a whole state: whatever this session answered before is dropped.
  session.Reset();
  session.Append(prompt);
  const double prefill_seconds = Elapsed(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: Kolibri 1 ROCm (gfx1151)\n"
              << "Prompt tokens: " << prompt.size() << '\n'
              << "--- Generation Output ---\n";
  }
  sampling::SamplerState sampler(opt.sampling,
                                 std::span<const sampling::TokenId>(
                                     prompt.data(), prompt.size()));
  const auto decode_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const k1::TokenId token = sampler.Sample(session.logits());
    sampler.Accept(token);
    if (model.tokenizer().IsEndOfGeneration(token)) {
      break;
    }
    const std::string piece = model.tokenizer().TokenText(token, false);
    if (reply != nullptr) {
      reply->append(piece);
    }
    std::cout << piece << std::flush;
    // The token just drawn is the next row: one forward per token, the cache
    // already holding every position before it.
    session.Append(std::span<const k1::TokenId>(&token, 1));
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds = Elapsed(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in " << prefill_seconds
              << " s (" << static_cast<double>(prompt.size()) / prefill_seconds
              << " tok/s)\nGenerated " << generated << " tokens in " << seconds
              << " s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
  }
  return 0;
}

[[nodiscard]] std::vector<tokenization::ChatMessage> Messages(
    const PromptOptions& opt) {
  std::vector<tokenization::ChatMessage> messages;
  if (!opt.system_prompt.empty()) {
    messages.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
  }
  return messages;
}

[[nodiscard]] std::vector<k1::TokenId> Render(const k1::Model& model,
                                              std::span<const tokenization::ChatMessage> messages,
                                              const PromptOptions& opt) {
  std::string error;
  const auto rendered = k1::ChatTemplate::Render(
      messages, {}, k1::ResolveChatOptions(PromptReasoningOptions(opt)),
      &error);
  if (!rendered.has_value()) {
    throw std::invalid_argument("kolibri1 chat template: " + error);
  }
  return TokenizePrompt(model, rendered->text);
}

}  // namespace

bool IsKolibri1(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "kolibri1";
}

int RunKolibri1Prompt(const PromptOptions& opt,
                      std::chrono::steady_clock::time_point load_start) {
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "Kolibri 1 session creation failed: " << error << '\n';
    return 1;
  }
  try {
    std::vector<k1::TokenId> prompt;
    if (opt.use_chat_template) {
      auto messages = Messages(opt);
      messages.emplace_back(tokenization::ChatRole::kUser, opt.prompt_text);
      prompt = Render(*model, messages, opt);
    } else {
      prompt = TokenizePrompt(*model, opt.prompt_text);
    }
    return Generate(opt, *model, *session, prompt, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "Kolibri 1 prompt failed: " << e.what() << '\n';
    return 1;
  }
}

int RunKolibri1Chat(const PromptOptions& opt,
                    std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Kolibri 1 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "Kolibri 1 session creation failed: " << error << '\n';
    return 1;
  }
  auto history = Messages(opt);
  std::cout << "=== Gufo Interactive Chat (Kolibri 1) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.emplace_back(tokenization::ChatRole::kUser, input);
    std::vector<k1::TokenId> prompt;
    try {
      prompt = Render(*model, history, opt);
    } catch (const std::exception& e) {
      std::cerr << "Kolibri 1 chat template failed: " << e.what() << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, prompt, &reply) != 0) {
      return 1;
    }
    // Every turn re-prefills the whole conversation: the cache is the shape of
    // the conversation, and the thinking the template drops from history is
    // not in it either.
    history.emplace_back(tokenization::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace gufo::cli
