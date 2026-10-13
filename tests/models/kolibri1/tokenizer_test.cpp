// Contract check for the kolibri1 tokenizer against the fixtures of
// make_tokenizer_fixture.py: a synthetic gpt2 vocabulary plus the ids an
// independent Python implementation of the same qwen2 split and GPT-2 merge
// rules produced. Nothing here needs the real artifact; the checks that do run
// in tokenizer_model_test.cpp.
#include "src/models/kolibri1/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "tests/models/kolibri1/check.hpp"
#include "tests/models/kolibri1/gguf_builder.hpp"

namespace k1 = gufo::models::kolibri1;
using kolibri1_test::Require;
using gufo::json::Value;
using k1::TokenId;
using k1::TokenType;

#ifndef GUFO_KOLIBRI1_FIXTURES
#error "GUFO_KOLIBRI1_FIXTURES must name the fixture directory"
#endif

namespace {

[[nodiscard]] std::string ReadFixture(const std::string& name) {
  const std::string path = std::string(GUFO_KOLIBRI1_FIXTURES) + "/" + name;
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot read " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

[[nodiscard]] const Value& Member(const Value& object, std::string_view key) {
  const auto* value = object.find(std::string(key));
  Require(value != nullptr,
          "fixture member missing: " + std::string(key));
  return *value;
}

[[nodiscard]] std::vector<std::string> TextArray(const Value& array,
                                                 const std::string& what) {
  Require(array.is_array(), what + " must be an array");
  std::vector<std::string> out;
  out.reserve(array.size());
  for (const auto& item : array.items()) {
    Require(item.is_string(), what + " must hold strings");
    out.push_back(item.str());
  }
  return out;
}

[[nodiscard]] std::vector<TokenId> IdArray(const Value& array,
                                           const std::string& what) {
  Require(array.is_array(), what + " must be an array");
  std::vector<TokenId> out;
  out.reserve(array.size());
  for (const auto& item : array.items()) {
    out.push_back(static_cast<TokenId>(item.as_size()));
  }
  return out;
}

[[nodiscard]] std::string Unhex(const std::string& hex) {
  const auto digit = [](char c) {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : c - 'A' + 10;
  };
  Require(hex.size() % 2 == 0, "fixture hex has odd length");
  std::string out;
  out.reserve(hex.size() / 2);
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    out.push_back(static_cast<char>((digit(hex[i]) << 4) | digit(hex[i + 1])));
  }
  return out;
}

[[nodiscard]] std::string Describe(const std::vector<TokenId>& ids) {
  std::string out;
  for (const TokenId id : ids) {
    out += std::to_string(id) + " ";
  }
  return out;
}

struct Fixture {
  k1::Tokenizer::Vocabulary vocabulary;
  std::unique_ptr<k1::Tokenizer> tokenizer;
  std::vector<std::string> control_texts;
  std::vector<std::string> user_texts;
  std::vector<std::string> eog_texts;

  [[nodiscard]] TokenId Id(std::string_view text) const {
    const auto id = tokenizer->FindToken(text);
    Require(id.has_value(), "fixture token missing: " + std::string(text));
    return *id;
  }
};

/// The vocabulary of the fixture, with the token types the file's
/// `tokenizer.ggml.token_type` array would carry.
Fixture Load() {
  const Value vocabulary =
      gufo::json::parse(ReadFixture("tokenizer_vocab.json"));
  Require(Member(vocabulary, "model").str() == "gpt2",
          "the fixture vocabulary must be gpt2");
  Require(Member(vocabulary, "pre").str() == "kolibri1",
          "the fixture vocabulary must carry the kolibri1 pre-tokenizer");
  // The released file names neither BOS nor an unknown token: neither key is
  // present at all, and the fixture copies that.
  Require(Member(vocabulary, "bos").is_null(),
          "the fixture must not name a BOS token");
  Require(Member(vocabulary, "unknown_text").is_null(),
          "the fixture must not name an unknown token");

  Fixture fixture;
  fixture.vocabulary.tokens = TextArray(Member(vocabulary, "tokens"), "tokens");
  fixture.vocabulary.merges = TextArray(Member(vocabulary, "merges"), "merges");
  fixture.control_texts =
      TextArray(Member(vocabulary, "control_texts"), "control_texts");
  fixture.user_texts =
      TextArray(Member(vocabulary, "user_defined_texts"), "user_defined_texts");
  fixture.eog_texts = TextArray(Member(vocabulary, "eog_texts"), "eog_texts");
  fixture.vocabulary.types.assign(fixture.vocabulary.tokens.size(),
                                  TokenType::kNormal);
  const auto type_of = [&](const std::vector<std::string>& texts,
                           TokenType type) {
    for (const auto& text : texts) {
      const auto where = std::find(fixture.vocabulary.tokens.begin(),
                                   fixture.vocabulary.tokens.end(), text);
      Require(where != fixture.vocabulary.tokens.end(),
              "fixture token missing: " + text);
      fixture.vocabulary.types[static_cast<std::size_t>(
          where - fixture.vocabulary.tokens.begin())] = type;
    }
  };
  type_of(fixture.control_texts, TokenType::kControl);
  type_of(fixture.user_texts, TokenType::kUserDefined);
  type_of({Member(vocabulary, "pad_text").str()}, TokenType::kControl);
  const auto eos = std::find(fixture.vocabulary.tokens.begin(),
                             fixture.vocabulary.tokens.end(),
                             Member(vocabulary, "eos_text").str());
  Require(eos != fixture.vocabulary.tokens.end(),
          "the fixture has no EOS token");
  fixture.vocabulary.eos = static_cast<TokenId>(
      eos - fixture.vocabulary.tokens.begin());
  fixture.vocabulary.add_bos = Member(vocabulary, "add_bos").as_bool();

  std::string error;
  fixture.tokenizer = k1::Tokenizer::Create(fixture.vocabulary, &error);
  Require(fixture.tokenizer != nullptr,
          "fixture vocabulary rejected: " + error);
  return fixture;
}

void Expect(const Fixture& f, std::string_view text, bool add_bos,
            bool parse_special, const std::vector<TokenId>& expected,
            const std::string& what) {
  const auto got = f.tokenizer->Encode(text, add_bos, parse_special);
  if (got != expected) {
    std::size_t index = 0;
    while (index < got.size() && index < expected.size() &&
           got[index] == expected[index]) {
      ++index;
    }
    throw std::runtime_error(
        what + ": first difference at " + std::to_string(index) + " (want " +
        (index < expected.size() ? std::to_string(expected[index]) : "end") +
        ", got " + (index < got.size() ? std::to_string(got[index]) : "end") +
        ", " + std::to_string(expected.size()) + " vs " +
        std::to_string(got.size()) + " ids)\n  expected: [" +
        Describe(expected) + "]\n  got:      [" + Describe(got) + "]");
  }
}

/// The recorded ids, and the recorded split: encoding every pre-tokenizer
/// piece on its own must give exactly the ids of the whole text, because a
/// merge never crosses a piece boundary.
void CheckGoldens(const Fixture& f, const Value& golden) {
  const auto& entries = Member(golden, "entries");
  Require(entries.is_array() && entries.size() >= 12,
          "the golden fixture must cover the fixed input set");
  for (const auto& entry : entries.items()) {
    const std::string name = Member(entry, "name").str();
    const std::string text = Unhex(Member(entry, "text_hex").str());
    const auto plain = IdArray(Member(entry, "ids_plain"), name + " ids_plain");
    Expect(f, text, false, false, plain, name + ": plain ids");
    Expect(f, text, false, true,
           IdArray(Member(entry, "ids_parse_special"), name),
           name + ": parse_special ids");
    std::vector<TokenId> per_piece;
    for (const auto& piece : Member(entry, "pieces_hex").items()) {
      const auto ids = f.tokenizer->Encode(Unhex(piece.str()), false, false);
      per_piece.insert(per_piece.end(), ids.begin(), ids.end());
    }
    Require(per_piece == plain,
            name + ": the recorded split does not reproduce the ids");
  }
}

/// Every byte is addressable through its GPT-2 symbol, so any text survives
/// Encode/Decode - including invalid UTF-8, which nothing rewrites.
void CheckBytesRoundTrip(const Fixture& f) {
  Require(f.tokenizer->VocabSize() == f.vocabulary.tokens.size(),
          "vocabulary size");
  for (int byte = 0; byte < 256; ++byte) {
    const std::string text(1, static_cast<char>(byte));
    const auto ids = f.tokenizer->Encode(text, false, false);
    Require(ids.size() == 1,
            "byte " + std::to_string(byte) + " must be one symbol token");
    Require(f.tokenizer->Decode(ids, false) == text,
            "byte " + std::to_string(byte) + " must decode back to itself");
    Require(f.tokenizer->Type(ids[0]) == TokenType::kNormal,
            "byte symbols are normal tokens");
  }
  const std::string invalid =
      "\xff" "caf\xff\xe9" "\xc3" "\x80\x81" "na\xc3\xa4ve" "\xe0\x80\xaf"
      "end";
  const auto ids = f.tokenizer->Encode(invalid, false, false);
  Require(f.tokenizer->Decode(ids, false) == invalid,
          "invalid UTF-8 must round-trip unchanged");
}

void CheckSpecials(const Fixture& f) {
  for (const auto& text : f.user_texts) {
    // User-defined tokens match whether or not special text is parsed.
    Expect(f, text, false, false, {f.Id(text)},
           "user-defined token without parse_special");
  }
  for (const auto& text : f.control_texts) {
    Expect(f, text, false, true, {f.Id(text)},
           "control token with parse_special");
    const auto plain = f.tokenizer->Encode(text, false, false);
    Require(plain != std::vector<TokenId>{f.Id(text)} && plain.size() > 1,
            "control text must stay word text without parse_special: " + text);
  }
  // A matched special cuts the text: nothing merges across the boundary.
  const std::string control = f.control_texts.front();
  std::vector<TokenId> expected = f.tokenizer->Encode("a", false, true);
  expected.push_back(f.Id(control));
  const auto tail = f.tokenizer->Encode("b", false, true);
  expected.insert(expected.end(), tail.begin(), tail.end());
  Expect(f, "a" + control + "b", false, true, expected,
         "text around a control token");
}

void CheckTokenTextAndEog(const Fixture& f) {
  for (const auto& text : f.control_texts) {
    const TokenId id = f.Id(text);
    Require(f.tokenizer->TokenText(id, false).empty(),
            "control text must stay hidden without special");
    Require(f.tokenizer->TokenText(id, true) == text,
            "control text must render with special");
  }
  for (const auto& text : f.user_texts) {
    Require(f.tokenizer->TokenText(f.Id(text), false) == text,
            "user-defined text renders without special");
  }
  std::vector<TokenId> expected;
  for (const auto& text : f.eog_texts) {
    const TokenId id = f.Id(text);
    Require(f.tokenizer->IsEndOfGeneration(id),
            "end-of-generation token missing: " + text);
    Require(f.tokenizer->Type(id) == TokenType::kControl,
            "end-of-generation tokens behave as control tokens: " + text);
    expected.push_back(id);
  }
  std::sort(expected.begin(), expected.end());
  const auto reported = f.tokenizer->EndOfGenerationTokens();
  Require(std::equal(reported.begin(), reported.end(), expected.begin()) &&
              reported.size() == expected.size(),
          "the reported set is exactly " + Describe(expected) + ", got [" +
              Describe(std::vector<TokenId>(reported.begin(), reported.end())) +
              "]");
  Require(f.tokenizer->IsEndOfGeneration(f.tokenizer->EosToken()),
          "EOS is always end-of-generation");
  Require(f.tokenizer->Type(-1) == TokenType::kUndefined &&
              f.tokenizer->TokenText(-1, true).empty() &&
              f.tokenizer->TokenText(
                  static_cast<TokenId>(f.tokenizer->VocabSize()), false)
                  .empty(),
          "ids outside the vocabulary are undefined");
}

/// The released file clears `add_bos_token` and names no BOS at all, so
/// prepending is a no-op for two independent reasons; a vocabulary that does
/// carry both gets exactly one BOS.
void CheckBos(const Fixture& f) {
  const std::string text = "Hello world, ist es schoen?";
  const auto plain = f.tokenizer->Encode(text, false, false);
  Expect(f, text, true, false, plain, "add_bos with a cleared flag");

  std::string error;
  auto with_bos = f.vocabulary;
  with_bos.add_bos = true;
  with_bos.bos = with_bos.eos;
  const auto prepend = k1::Tokenizer::Create(with_bos, &error);
  Require(prepend != nullptr, "vocabulary with a BOS rejected: " + error);
  auto expected = plain;
  expected.insert(expected.begin(), with_bos.bos);
  Require(prepend->Encode(text, true, false) == expected,
          "the flag and the token together prepend exactly one BOS");
  Require(prepend->Encode(text, false, false) == plain,
          "the flag alone must not add a BOS");

  auto without = f.vocabulary;
  without.add_bos = true;  // flag set, but the file names no BOS token
  const auto bare = k1::Tokenizer::Create(without, &error);
  Require(bare != nullptr, "vocabulary without a BOS rejected: " + error);
  Require(bare->BosToken() < 0, "no BOS key means no BOS token");
  Require(bare->Encode(text, true, false) == plain,
          "a vocabulary without a BOS token cannot prepend one");
}

/// A run of letters is one piece, and pieces over a threshold are cached; a
/// cached piece must encode like a fresh one, wherever it appears.
void CheckPieceCache(const Fixture& f) {
  std::string piece;
  while (piece.size() < 64) {
    piece += "supercalifragilistic";
  }
  const auto once = f.tokenizer->Encode(piece, false, false);
  Require(f.tokenizer->Encode(piece, false, false) == once,
          "a repeated piece must encode identically");
  auto twice = once;
  twice.insert(twice.end(), once.begin(), once.end());
  Require(f.tokenizer->Encode(piece + piece, false, false) == twice,
          "two cached pieces in one text");
  const auto prefixed = f.tokenizer->Encode(" " + piece, false, false);
  auto joined = once;
  joined.insert(joined.end(), prefixed.begin(), prefixed.end());
  Require(f.tokenizer->Encode(piece + " " + piece, false, false) == joined,
          "a piece keeps its ids inside other text");
}

/// A different pre-tokenizer would split words differently and silently change
/// every id, so the GGUF loader accepts exactly one combination.
void CheckGgufGate(const Fixture& f) {
  kolibri1_test::I32Array types;
  for (const auto type : f.vocabulary.types) {
    types.values.push_back(static_cast<std::int32_t>(type));
  }
  const auto build = [&](const std::string& model, const std::string& pre,
                         bool with_types) {
    kolibri1_test::Metadata metadata;
    metadata["tokenizer.ggml.model"] = model;
    metadata["tokenizer.ggml.pre"] = pre;
    metadata["tokenizer.ggml.tokens"] =
        kolibri1_test::StringArray{f.vocabulary.tokens};
    metadata["tokenizer.ggml.merges"] =
        kolibri1_test::StringArray{f.vocabulary.merges};
    if (with_types) {
      metadata["tokenizer.ggml.token_type"] = types;
    }
    metadata["tokenizer.ggml.eos_token_id"] =
        static_cast<std::uint32_t>(f.vocabulary.eos);
    metadata["tokenizer.ggml.padding_token_id"] =
        static_cast<std::uint32_t>(f.vocabulary.eos);
    metadata["tokenizer.ggml.add_bos_token"] = false;
    return kolibri1_test::GgufImage(metadata);
  };

  const auto accepted = build("gpt2", "kolibri1", true);
  Require(accepted.error().empty(), accepted.error());
  std::string error;
  const auto tokenizer = k1::Tokenizer::CreateFromGguf(*accepted.reader(),
                                                       &error);
  Require(tokenizer != nullptr, "gpt2 with kolibri1 rejected: " + error);
  Require(tokenizer->VocabSize() == f.tokenizer->VocabSize(),
          "the GGUF path must load the same vocabulary");
  Require(tokenizer->Encode("Hello world", false, false) ==
              f.tokenizer->Encode("Hello world", false, false),
          "the GGUF path must encode identically");
  Require(tokenizer->EosToken() == f.tokenizer->EosToken(),
          "EOS comes from the metadata");
  Require(tokenizer->BosToken() < 0, "a missing BOS key stays unset");

  const auto wrong_model = build("qwen2", "kolibri1", true);
  error.clear();
  Require(k1::Tokenizer::CreateFromGguf(*wrong_model.reader(), &error) ==
              nullptr,
          "a non-gpt2 model must be rejected");
  Require(error.find("gpt2") != std::string::npos,
          "the rejection must name the expected model: " + error);

  const auto wrong_pre = build("gpt2", "qwen2", true);
  error.clear();
  Require(k1::Tokenizer::CreateFromGguf(*wrong_pre.reader(), &error) == nullptr,
          "a non-kolibri1 pre-tokenizer must be rejected");
  Require(error.find("kolibri1") != std::string::npos,
          "the rejection must name the expected pre-tokenizer: " + error);

  const auto no_types = build("gpt2", "kolibri1", false);
  error.clear();
  Require(
      k1::Tokenizer::CreateFromGguf(*no_types.reader(), &error) == nullptr,
      "a vocabulary without token types must be rejected");
  Require(error.find("token_type") != std::string::npos,
          "the rejection must name the missing key: " + error);
}

/// What this tokenizer cannot work with is refused, not worked around.
void CheckCreateRejects() {
  std::string error;

  k1::Tokenizer::Vocabulary mismatched;
  mismatched.tokens = {"a", "b"};
  mismatched.types = {TokenType::kNormal};
  mismatched.eos = 0;
  error.clear();
  Require(k1::Tokenizer::Create(mismatched, &error) == nullptr,
          "token and type counts must match");
  Require(error.find("size") != std::string::npos, error);

  k1::Tokenizer::Vocabulary no_eos;
  no_eos.tokens = {"a"};
  no_eos.types = {TokenType::kNormal};
  error.clear();
  Require(k1::Tokenizer::Create(no_eos, &error) == nullptr,
          "a vocabulary without an EOS must be refused");
  Require(error.find("EOS") != std::string::npos, error);

  k1::Tokenizer::Vocabulary no_bytes;
  no_bytes.tokens = {"<|im_end|>"};
  no_bytes.types = {TokenType::kControl};
  no_bytes.eos = 0;
  error.clear();
  Require(k1::Tokenizer::Create(no_bytes, &error) == nullptr,
          "a vocabulary without byte symbols must be refused");
  Require(error.find("GPT-2 symbol") != std::string::npos, error);

  k1::Tokenizer::Vocabulary malformed;
  malformed.tokens = {"A", "B", "<|im_end|>"};
  malformed.types = {TokenType::kNormal, TokenType::kNormal,
                     TokenType::kControl};
  malformed.eos = 2;
  malformed.merges = {"AB"};  // no separator between the pair
  error.clear();
  Require(k1::Tokenizer::Create(malformed, &error) == nullptr,
          "a merge without a separator must be refused");
  Require(error.find("malformed") != std::string::npos, error);

  k1::Tokenizer::Vocabulary dangling;
  dangling.tokens = {"A", "B", "<|im_end|>"};
  dangling.types = {TokenType::kNormal, TokenType::kNormal,
                    TokenType::kControl};
  dangling.eos = 2;
  dangling.merges = {"A B"};  // "AB" is not in the vocabulary
  error.clear();
  Require(k1::Tokenizer::Create(dangling, &error) == nullptr,
          "a merge outside the vocabulary must be refused");
  Require(error.find("outside the vocabulary") != std::string::npos, error);
}

}  // namespace

int main() {
  return kolibri1_test::Run([&] {
    const Fixture fixture = Load();
    const Value golden =
        gufo::json::parse(ReadFixture("tokenizer_golden.json"));
    CheckGoldens(fixture, golden);
    CheckBytesRoundTrip(fixture);
    CheckSpecials(fixture);
    CheckTokenTextAndEog(fixture);
    CheckBos(fixture);
    CheckPieceCache(fixture);
    CheckGgufGate(fixture);
    CheckCreateRejects();
    std::cout << "checked " << fixture.tokenizer->VocabSize()
              << " tokens against the goldens\n";
  });
}
