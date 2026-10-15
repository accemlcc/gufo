// Binds the real Kolibri 1 tokenizer named by GUFO_KOLIBRI1_MODEL and pins the
// vocabulary facts this tokenizer's semantics rest on: the byte-level layout,
// the control ids the chat template needs, and lossless round-trips. Without
// the file the check skips (77), so the hosted suite never reads 44 GiB.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kolibri1/tokenizer.hpp"
#include "tests/models/kolibri1/check.hpp"

namespace k1 = gufo::models::kolibri1;
using k1::TokenId;
using k1::TokenType;
using kolibri1_test::Require;

namespace {

/// `tokenizer.ggml.token_type` layout of the released vocabulary: 127900
/// normal byte and word tokens, then the controls, then the user-defined
/// chat tokens, then the two unused padding entries.
constexpr TokenId kFirstControl = 127900;
constexpr TokenId kFirstUserDefined = 127907;
constexpr TokenId kFirstUnused = 127998;
constexpr std::size_t kVocabSize = 128000;

/// The controls by id, in file order. `<|endoftext|>` is the file's padding
/// token, `<|im_end|>` its EOS; there is no BOS token.
constexpr std::array<std::pair<TokenId, std::string_view>, 7> kControls = {{
    {127900, "<|text|>"},
    {127901, "<|endoftext|>"},
    {127902, "<|pad|>"},
    {127903, "<|chat|>"},
    {127904, "<|im_start|>"},
    {127905, "<|/role|>"},
    {127906, "<|im_end|>"},
}};

/// The user-defined tokens the chat template and the tool parser emit. The
/// file stores them bare - `tokenizer.ggml.tokens[127908]` is the closing think
/// tag on its own, not the newline-wrapped form the template writes around it -
/// so the template has to supply the newlines itself.
constexpr std::array<std::pair<TokenId, std::string_view>, 6> kUserDefined = {{
    {127907, "<think>"},
    {127908, "</think>"},
    {127909, "<tool_call>"},
    {127910, "</tool_call>"},
    {127911, "<tool_response>"},
    {127912, "</tool_response>"},
}};

void CheckLayout(const k1::Tokenizer& t) {
  Require(t.VocabSize() == kVocabSize,
          "the released vocabulary has " + std::to_string(kVocabSize) +
              " entries, got " + std::to_string(t.VocabSize()));
  // Byte level: ids 0..255 are the bytes_to_unicode symbols in byte order, so
  // every byte is one token and decodes back to itself.
  for (int byte = 0; byte < 256; ++byte) {
    const std::string text(1, static_cast<char>(byte));
    const auto ids = t.Encode(text, false, false);
    Require(ids.size() == 1 && ids[0] == byte,
            "byte " + std::to_string(byte) + " must be token " +
                std::to_string(byte) + ", got [" + std::to_string(ids[0]) +
                "]");
    Require(t.Decode(ids, false) == text,
            "byte " + std::to_string(byte) + " must decode back to itself");
  }
  Require(t.Type(kFirstControl) == TokenType::kControl &&
              t.Type(kFirstUserDefined) == TokenType::kUserDefined &&
              t.Type(kFirstUnused) == TokenType::kUnused,
          "the control, user-defined and unused ranges start where the file "
          "puts them");
  Require(t.Type(kFirstControl - 1) == TokenType::kNormal,
          "word tokens run up to the control range");
  Require(t.TokenText(kFirstUnused, true).empty(),
          "unused tokens carry no text");
}

void CheckSpecialTokens(const k1::Tokenizer& t) {
  Require(t.EosToken() == 127906, "tokenizer.ggml.eos_token_id is <|im_end|>");
  Require(t.BosToken() < 0, "the file names no BOS token");
  for (const auto& [id, text] : kControls) {
    Require(t.Type(id) == TokenType::kControl,
            "control token type at " + std::to_string(id));
    const auto found = t.FindToken(text);
    Require(found.has_value() && *found == id,
            std::string("control token ") + std::string(text) + " must be " +
                std::to_string(id));
    Require(t.TokenText(id, true) == text,
            "control text must render with special at " + std::to_string(id));
    Require(t.TokenText(id, false).empty(),
            "control text must stay hidden without special");
    // Specials are addressed by id and never re-merged.
    const auto ids = t.Encode(text, false, true);
    Require(ids.size() == 1 && ids[0] == id,
            "control text must encode to one token: " + std::string(text));
  }
  for (const auto& [id, text] : kUserDefined) {
    Require(t.Type(id) == TokenType::kUserDefined,
            "user-defined token type at " + std::to_string(id));
    Require(t.FindToken(text).value_or(-1) == id,
            "user-defined token " + std::string(text));
    // Tool and thinking tags match even when special text is not parsed.
    const auto ids = t.Encode(text, false, false);
    Require(ids.size() == 1 && ids[0] == id,
            "user-defined text must encode to one token: " + std::string(text));
    Require(t.TokenText(id, false) == text,
            "user-defined text renders without special");
  }
}

/// The end-of-generation set of the released vocabulary: the file's EOS plus
/// the controls that can close a turn or the whole sequence.
void CheckEndOfGeneration(const k1::Tokenizer& t) {
  const std::vector<TokenId> expected = {127901, 127905, 127906};
  const auto reported = t.EndOfGenerationTokens();
  Require(std::equal(reported.begin(), reported.end(), expected.begin()) &&
              reported.size() == expected.size(),
          "end-of-generation must be <|endoftext|>, <|/role|> and <|im_end|>");
  for (const TokenId id : expected) {
    Require(t.IsEndOfGeneration(id),
            "end-of-generation must hold for " + std::to_string(id));
  }
  Require(!t.IsEndOfGeneration(127904),
          "<|im_start|> opens a turn, it never ends one");
  Require(!t.IsEndOfGeneration(0), "a byte token is not end-of-generation");
}

/// A German sentence, the model's first language, must come back unchanged,
/// and the chat framing must cut the text at the controls.
void CheckRoundTrip(const k1::Tokenizer& t) {
  const std::string sentence =
      "Größenordnung hallo: die Straße war weiß, über alle Berge und süß "
      "zugleich — 3,14 °C, »leider« naïv.\n";
  const auto ids = t.Encode(sentence, false, false);
  Require(!ids.empty(), "the sentence must encode");
  Require(t.Decode(ids, false) == sentence,
          "Encode/Decode must be lossless: " + t.Decode(ids, false));
  // Nothing matched a control, so every id is a word token.
  for (const TokenId id : ids) {
    Require(
        t.Type(id) == TokenType::kNormal,
        "plain text must not produce a control token: " + std::to_string(id));
  }
  // A run of letters and a run of whitespace stay inside their pieces; the
  // number of tokens is far below the number of bytes, so merging happened.
  Require(ids.size() * 2 < sentence.size(),
          "the sentence must merge: " + std::to_string(ids.size()) +
              " tokens for " + std::to_string(sentence.size()) + " bytes");

  const std::string turn = "<|im_start|>assistant\nGuten Tag<|im_end|>\n";
  const auto framed = t.Encode(turn, false, true);
  Require(framed.front() == 127904, "the turn must start with <|im_start|>");
  Require(t.Decode(framed, true) == turn,
          "a framed turn must decode back with its controls");
  Require(std::find(framed.begin(), framed.end(), 127906) != framed.end(),
          "the framed turn must carry <|im_end|>");
  // Without special parsing the same text is word text and grows.
  Require(t.Encode(turn, false, false).size() > framed.size(),
          "unparsed controls must encode as word text");

  // The file clears add_bos_token, so nothing is prepended.
  Require(t.Encode(sentence, true, false) == ids,
          "add_bos must be a no-op for this vocabulary");

  // Invalid UTF-8 is not rewritten: the bytes come back as they went in.
  const std::string invalid = "Größe\xff\xc3\xa4";
  Require(t.Decode(t.Encode(invalid, false, false), false) == invalid,
          "invalid bytes must round-trip unchanged");
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_KOLIBRI1_MODEL");
  if (model == nullptr || *model == '\0') {
    std::cout << "SKIP: GUFO_KOLIBRI1_MODEL is not set\n";
    return 77;
  }
  return kolibri1_test::Run([&] {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto tokenizer = k1::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer rejected: " + error);
    CheckLayout(*tokenizer);
    CheckSpecialTokens(*tokenizer);
    CheckEndOfGeneration(*tokenizer);
    CheckRoundTrip(*tokenizer);
    std::cout << "tokenized the released vocabulary: " << tokenizer->VocabSize()
              << " tokens, eos " << tokenizer->EosToken() << "\n";
  });
}
