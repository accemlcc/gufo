#include "src/models/kolibri1/tokenizer.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::kolibri1 {
namespace {

/// End-of-generation token texts of the Kolibri 1 vocabulary: the file's eos
/// (`<|im_end|>`, which closes every turn of the chat template), the role
/// terminator (`<|/role|>`) and the sequence end (`<|endoftext|>`, which the
/// file also uses as padding).
constexpr std::array<std::string_view, 3> kEndOfGeneration = {
    "<|im_end|>", "<|/role|>", "<|endoftext|>"};

/// Pieces from this length on are cached (shorter ones encode faster than a
/// lookup pays off), within this many bytes of text and ids.
constexpr std::size_t kCachedPieceBytes = 64;
constexpr std::size_t kPieceCacheBytes = std::size_t{64} << 20;

// No valid pair has this key: token ids stay below 2^31.
constexpr std::uint64_t kEmptyPair = ~std::uint64_t{0};

std::uint64_t PairKey(TokenId left, TokenId right) noexcept {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
         static_cast<std::uint32_t>(right);
}

std::size_t PairSlot(std::uint64_t key, std::uint32_t shift) noexcept {
  return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift);
}

void AppendUtf8(std::uint32_t cpt, std::string* out) {
  if (cpt <= 0x7F) {
    out->push_back(static_cast<char>(cpt));
  } else if (cpt <= 0x7FF) {
    out->push_back(static_cast<char>(0xC0 | (cpt >> 6)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt <= 0xFFFF) {
    out->push_back(static_cast<char>(0xE0 | (cpt >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cpt >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
}

/// GPT-2's `bytes_to_unicode`: printable ASCII and the two Latin-1 bands keep
/// their own code point, every other byte moves into U+0100 and up, so all
/// 256 symbols are printable and distinct.
std::uint32_t SymbolCodePoint(std::uint8_t byte) noexcept {
  const auto b = static_cast<int>(byte);
  if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE) {
    return b;
  }
  int rank = 0;
  for (int i = 0; i < b; ++i) {
    const bool printable = (i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) ||
                           i >= 0xAE;
    if (!printable) {
      ++rank;
    }
  }
  return 0x100 + rank;
}

/// The symbol text of every byte, as the vocabulary stores it.
const std::array<std::string, 256>& ByteSymbols() {
  static const auto table = [] {
    std::array<std::string, 256> symbols;
    for (std::size_t byte = 0; byte < symbols.size(); ++byte) {
      AppendUtf8(SymbolCodePoint(static_cast<std::uint8_t>(byte)),
                 &symbols[byte]);
    }
    return symbols;
  }();
  return table;
}

/// Symbol code point back to its byte, or -1.
std::int32_t SymbolByte(std::uint32_t cpt) noexcept {
  if ((cpt >= '!' && cpt <= '~') || (cpt >= 0xA1 && cpt <= 0xAC) ||
      (cpt >= 0xAE && cpt <= 0xFF)) {
    return static_cast<std::int32_t>(cpt);
  }
  if (cpt < 0x100 || cpt > 0x100 + 255) {
    return -1;
  }
  std::uint32_t rank = cpt - 0x100;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                           b >= 0xAE;
    if (!printable) {
      if (rank == 0) {
        return b;
      }
      --rank;
    }
  }
  return -1;
}

/// Stored token text with the byte symbols resolved back to bytes.
std::string UnescapeSymbols(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  const auto* data = reinterpret_cast<const uint8_t*>(text.data());
  std::int32_t offset = 0;
  const std::int32_t length = static_cast<std::int32_t>(text.size());
  while (offset < length) {
    UChar32 cpt = 0xFFFD;
    U8_NEXT(data, offset, length, cpt);
    const std::int32_t byte = SymbolByte(static_cast<std::uint32_t>(cpt));
    if (byte >= 0) {
      out.push_back(static_cast<char>(byte));
    } else {
      AppendUtf8(static_cast<std::uint32_t>(cpt), &out);
    }
  }
  return out;
}

bool IsLetter(UChar32 cpt) noexcept {
  return (U_GET_GC_MASK(cpt) & U_GC_L_MASK) != 0;
}

bool IsNumber(UChar32 cpt) noexcept {
  const auto category = static_cast<UCharCategory>(u_charType(cpt));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsWhitespace(UChar32 cpt) noexcept { return u_isUWhiteSpace(cpt) != 0; }

bool IsNewline(UChar32 cpt) noexcept { return cpt == '\r' || cpt == '\n'; }

struct CodePoint {
  UChar32 value{0xFFFD};
  std::size_t length{1};
};

/// ICU's strict decoder: a malformed or truncated sequence yields U+FFFD and
/// advances one byte, which is how RE2 walks the same text.
CodePoint Next(std::string_view text, std::size_t offset) {
  const auto* data = reinterpret_cast<const uint8_t*>(text.data());
  std::int32_t index = static_cast<std::int32_t>(offset);
  const std::int32_t length = static_cast<std::int32_t>(text.size());
  UChar32 cpt = 0xFFFD;
  U8_NEXT(data, index, length, cpt);
  return {cpt, static_cast<std::size_t>(index - static_cast<std::int32_t>(
                                                 offset))};
}

/// `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d` regardless of case.
std::size_t ContractionEnd(std::string_view text, std::size_t offset) noexcept {
  if (text[offset] != '\'') {
    return offset;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {
      "s", "t", "re", "ve", "m", "ll", "d",
  };
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const CodePoint next = Next(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) != letter) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) {
      return end;
    }
  }
  return offset;
}

/// End of the maximal `\p{L}+` run starting at `from`.
std::size_t LetterRunEnd(std::string_view text, std::size_t from) {
  std::size_t end = from;
  while (end < text.size()) {
    const CodePoint letter = Next(text, end);
    if (!IsLetter(letter.value)) {
      break;
    }
    end += letter.length;
  }
  return end;
}

/// End of the piece starting at `offset`, following llama.cpp's `qwen2` split
/// expression alternative by alternative, first match wins:
///
///   (?i:'s|'t|'re|'ve|'m|'ll|'d)
///   |[^\r\n\p{L}\p{N}]?\p{L}+
///   |\p{N}
///   | ?[^\s\p{L}\p{N}]+[\r\n]*
///   |\s*[\r\n]+
///   |\s+(?!\S)
///   |\s+
std::size_t PieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t contraction = ContractionEnd(text, offset);
      contraction != offset) {
    return contraction;
  }

  const CodePoint first = Next(text, offset);
  if (IsLetter(first.value)) {
    return LetterRunEnd(text, offset + first.length);
  }
  // The optional [^\r\n\p{L}\p{N}] prefix of the second alternative.
  if (!IsNewline(first.value) && !IsNumber(first.value) &&
      offset + first.length < text.size() &&
      IsLetter(Next(text, offset + first.length).value)) {
    return LetterRunEnd(text, offset + first.length);
  }

  if (IsNumber(first.value)) {
    return offset + first.length;
  }

  // The leading ` ?` belongs to the punctuation run only when what follows it
  // is punctuation as well; otherwise the space is whitespace and the run
  // starts after it.
  std::size_t punctuation_start = offset;
  if (first.value == ' ' && offset + first.length < text.size()) {
    const CodePoint next = Next(text, offset + first.length);
    if (!IsWhitespace(next.value) && !IsLetter(next.value) &&
        !IsNumber(next.value)) {
      punctuation_start += first.length;
    }
  }
  if (const CodePoint punctuation = Next(text, punctuation_start);
      !IsWhitespace(punctuation.value) && !IsLetter(punctuation.value) &&
      !IsNumber(punctuation.value)) {
    std::size_t end = punctuation_start;
    while (end < text.size()) {
      const CodePoint next = Next(text, end);
      if (IsWhitespace(next.value) || IsLetter(next.value) ||
          IsNumber(next.value)) {
        break;
      }
      end += next.length;
    }
    while (end < text.size()) {
      const CodePoint next = Next(text, end);
      if (!IsNewline(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (IsWhitespace(first.value)) {
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const CodePoint next = Next(text, end);
      if (!IsWhitespace(next.value)) {
        break;
      }
      last_start = end;
      end += next.length;
      if (IsNewline(next.value)) {
        last_newline_end = end;
      }
    }
    // \s*[\r\n]+ wins as soon as the run reaches a newline.
    if (last_newline_end != offset) {
      return last_newline_end;
    }
    // \s+(?!\S) consumes all trailing whitespace, but backtracks one
    // code point before non-whitespace so the next piece can own its prefix.
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return offset + first.length;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };
  const auto model = reader.GetMetadataString("tokenizer.ggml.model");
  if (model != std::optional<std::string_view>{"gpt2"}) {
    return fail("tokenizer.ggml.model is not gpt2: " +
                (model ? std::string(*model) : std::string("<missing>")));
  }
  const auto pre = reader.GetMetadataString("tokenizer.ggml.pre");
  if (pre != std::optional<std::string_view>{"kolibri1"}) {
    return fail("tokenizer.ggml.pre is not kolibri1: " +
                (pre ? std::string(*pre) : std::string("<missing>")));
  }
  Vocabulary v;
  for (auto token : reader.GetMetadataStringArray("tokenizer.ggml.tokens")) {
    v.tokens.emplace_back(token);
  }
  for (auto merge : reader.GetMetadataStringArray("tokenizer.ggml.merges")) {
    v.merges.emplace_back(merge);
  }
  const auto* types = reader.FindMetadata("tokenizer.ggml.token_type");
  if (types == nullptr) {
    return fail("missing tokenizer.ggml.token_type");
  }
  const auto push_type = [&](auto value) {
    v.types.push_back(value >= 0 && value <= 6 ? static_cast<TokenType>(value)
                                               : TokenType::kUndefined);
  };
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&types->value)) {
    for (auto value : *s) {
      push_type(value);
    }
  } else if (const auto* u =
                 std::get_if<std::vector<std::uint64_t>>(&types->value)) {
    for (auto value : *u) {
      push_type(static_cast<std::int64_t>(std::min<std::uint64_t>(value, 7)));
    }
  } else {
    return fail("tokenizer.ggml.token_type must be an integer array");
  }
  const auto id = [&](std::string_view key) -> TokenId {
    const auto value = reader.GetMetadataUint32(key);
    return value ? static_cast<TokenId>(*value) : -1;
  };
  // Kolibri 1 names no BOS and no unknown token at all.
  v.bos = id("tokenizer.ggml.bos_token_id");
  v.eos = id("tokenizer.ggml.eos_token_id");
  v.pad = id("tokenizer.ggml.padding_token_id");
  v.unknown = id("tokenizer.ggml.unknown_token_id");
  v.add_bos =
      reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  return Create(std::move(v), error_msg);
}

std::unique_ptr<Tokenizer> Tokenizer::Create(Vocabulary v,
                                             std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };
  if (v.tokens.empty() || v.tokens.size() > 0x7fffffffU ||
      v.types.size() != v.tokens.size()) {
    return fail("tokenizer vocabulary and token types must match in size");
  }
  const auto n = static_cast<TokenId>(v.tokens.size());
  const auto in_range = [&](TokenId t) { return t >= 0 && t < n; };
  if (!in_range(v.eos)) {
    return fail("tokenizer EOS token id is out of range");
  }
  if (v.bos >= 0 && !in_range(v.bos)) {
    return fail("tokenizer BOS token id is out of range");
  }

  std::unique_ptr<Tokenizer> t(new Tokenizer());
  t->tokens_ = std::move(v.tokens);
  t->types_ = std::move(v.types);
  t->bos_ = v.bos;
  t->eos_ = v.eos;
  t->add_bos_ = v.add_bos;
  t->token_to_id_.reserve(t->tokens_.size());
  for (TokenId i = 0; i < n; ++i) {
    // The first occurrence wins, as in llama.cpp's text_to_token map.
    t->token_to_id_.emplace(t->tokens_[i], i);
  }

  // End-of-generation tokens behave as control tokens even when the file
  // types them otherwise.
  const auto add_eog = [&](TokenId token) {
    if (in_range(token) &&
        std::find(t->eog_.begin(), t->eog_.end(), token) == t->eog_.end()) {
      t->eog_.push_back(token);
      t->types_[token] = TokenType::kControl;
    }
  };
  add_eog(v.eos);
  for (auto text : kEndOfGeneration) {
    if (auto token = t->FindToken(text)) {
      add_eog(*token);
    }
  }
  std::sort(t->eog_.begin(), t->eog_.end());

  for (TokenId i = 0; i < n; ++i) {
    const TokenType type = t->types_[i];
    if (type == TokenType::kControl || type == TokenType::kUserDefined ||
        type == TokenType::kUnknown) {
      if (!t->tokens_[i].empty()) {
        t->special_.push_back(i);
      }
    }
  }
  // Longest text first; equal lengths keep id order for determinism.
  std::stable_sort(t->special_.begin(), t->special_.end(),
                   [&](TokenId a, TokenId b) {
                     return t->tokens_[a].size() > t->tokens_[b].size();
                   });

  // A power-of-two table at most half full.
  std::uint32_t bits = 4;
  while ((std::size_t{1} << bits) < v.merges.size() * 2) {
    ++bits;
  }
  t->merge_shift_ = 64 - bits;
  t->merges_.assign(std::size_t{1} << bits, {kEmptyPair, 0, -1});
  const std::size_t mask = t->merges_.size() - 1;
  for (std::size_t rank = 0; rank < v.merges.size(); ++rank) {
    const std::string& merge = v.merges[rank];
    // Token texts are byte-level symbols and never contain a space, so the
    // first space separates the pair.
    const std::size_t split = merge.find(' ', 1);
    if (split == std::string::npos) {
      return fail("tokenizer merge " + std::to_string(rank) + " is malformed");
    }
    const auto left = t->FindToken(std::string_view(merge).substr(0, split));
    const auto right = t->FindToken(std::string_view(merge).substr(split + 1));
    const auto merged =
        t->FindToken(merge.substr(0, split) + merge.substr(split + 1));
    if (!left || !right || !merged) {
      // Symbols are tracked by id, so every merge must stay in the vocabulary.
      return fail("tokenizer merge " + std::to_string(rank) +
                  " references text outside the vocabulary");
    }
    const std::uint64_t key = PairKey(*left, *right);
    for (std::size_t slot = PairSlot(key, t->merge_shift_);;
         slot = (slot + 1) & mask) {
      MergeSlot& entry = t->merges_[slot];
      if (entry.key == key) {
        break;  // an earlier rank already holds this pair
      }
      if (entry.key == kEmptyPair) {
        entry = {key, static_cast<std::uint32_t>(rank), *merged};
        break;
      }
    }
  }

  // Every byte needs its GPT-2 symbol token: a vocabulary without one could
  // not represent arbitrary text, and dropping the byte would hide that.
  for (std::size_t byte = 0; byte < t->byte_tokens_.size(); ++byte) {
    const auto symbol = t->FindToken(ByteSymbols()[byte]);
    if (!symbol) {
      return fail("vocabulary has no GPT-2 symbol token for byte " +
                  std::to_string(byte));
    }
    t->byte_tokens_[byte] = *symbol;
  }
  return t;
}

std::optional<TokenId> Tokenizer::FindToken(
    std::string_view text) const noexcept {
  const auto it = token_to_id_.find(text);
  if (it == token_to_id_.end()) {
    return std::nullopt;
  }
  return it->second;
}

TokenType Tokenizer::Type(TokenId token) const noexcept {
  if (token < 0 || static_cast<std::size_t>(token) >= types_.size()) {
    return TokenType::kUndefined;
  }
  return types_[token];
}

bool Tokenizer::IsEndOfGeneration(TokenId token) const noexcept {
  return std::binary_search(eog_.begin(), eog_.end(), token);
}

Tokenizer::Merge Tokenizer::FindMerge(TokenId left,
                                      TokenId right) const noexcept {
  if (left < 0 || right < 0) {
    return {-1, -1};
  }
  const std::uint64_t key = PairKey(left, right);
  const std::size_t mask = merges_.size() - 1;
  for (std::size_t slot = PairSlot(key, merge_shift_);;
       slot = (slot + 1) & mask) {
    const MergeSlot& entry = merges_[slot];
    if (entry.key == key) {
      return {entry.rank, entry.merged};
    }
    if (entry.key == kEmptyPair) {
      return {-1, -1};
    }
  }
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text, bool add_bos,
                                       bool parse_special) const {
  // Fragments are either raw text or an already matched special token.
  struct Fragment {
    std::string_view text;
    TokenId token{-1};
  };
  std::vector<Fragment> fragments;
  if (!text.empty()) {
    fragments.push_back({text, -1});
  }
  for (const TokenId special : special_) {
    const TokenType type = types_[special];
    if (!parse_special &&
        (type == TokenType::kControl || type == TokenType::kUnknown)) {
      continue;
    }
    const std::string_view needle = tokens_[special];
    std::vector<Fragment> next;
    next.reserve(fragments.size());
    for (const Fragment& f : fragments) {
      if (f.token >= 0) {
        next.push_back(f);
        continue;
      }
      std::size_t start = 0;
      while (true) {
        const std::size_t match = f.text.find(needle, start);
        if (match == std::string_view::npos) {
          break;
        }
        if (match > start) {
          next.push_back({f.text.substr(start, match - start), -1});
        }
        next.push_back({{}, special});
        start = match + needle.size();
      }
      if (start < f.text.size()) {
        next.push_back({f.text.substr(start), -1});
      }
    }
    fragments = std::move(next);
  }

  std::vector<TokenId> out;
  if (add_bos && add_bos_ && bos_ >= 0) {
    out.push_back(bos_);
  }
  for (const Fragment& f : fragments) {
    if (f.token >= 0) {
      out.push_back(f.token);
    } else {
      EncodePieces(f.text, &out);
    }
  }
  return out;
}

void Tokenizer::EncodePieces(std::string_view text,
                             std::vector<TokenId>* out) const {
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = PieceEnd(text, start);
    const std::string_view piece = text.substr(start, end - start);
    start = end;
    if (piece.size() < kCachedPieceBytes) {
      EncodePiece(piece, out);
      continue;
    }
    {
      std::lock_guard lock(piece_cache_mutex_);
      if (const auto it = piece_index_.find(piece); it != piece_index_.end()) {
        piece_cache_.splice(piece_cache_.end(), piece_cache_, it->second);
        out->insert(out->end(), it->second->ids.begin(), it->second->ids.end());
        continue;
      }
    }
    CachedPiece entry{std::string(piece), {}};
    EncodePiece(piece, &entry.ids);
    out->insert(out->end(), entry.ids.begin(), entry.ids.end());
    const std::size_t bytes =
        entry.text.size() + entry.ids.size() * sizeof(TokenId);
    if (bytes > kPieceCacheBytes) {
      continue;
    }
    std::lock_guard lock(piece_cache_mutex_);
    if (piece_index_.contains(piece)) {
      continue;  // another request encoded it meanwhile
    }
    piece_cache_.push_back(std::move(entry));
    piece_index_.emplace(piece_cache_.back().text,
                         std::prev(piece_cache_.end()));
    piece_cache_bytes_ += bytes;
    while (piece_cache_bytes_ > kPieceCacheBytes) {
      const CachedPiece& oldest = piece_cache_.front();
      piece_cache_bytes_ -=
          oldest.text.size() + oldest.ids.size() * sizeof(TokenId);
      piece_index_.erase(oldest.text);
      piece_cache_.pop_front();
    }
  }
}

void Tokenizer::EncodePiece(std::string_view piece,
                            std::vector<TokenId>* out) const {
  // Byte level: every byte starts as its own symbol token.
  std::vector<TokenId> ids;
  ids.reserve(piece.size());
  for (const char c : piece) {
    ids.push_back(byte_tokens_[static_cast<std::uint8_t>(c)]);
  }

  struct Symbol {
    TokenId id;
    int prev;
    int next;
  };
  std::vector<Symbol> symbols;
  symbols.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    symbols.push_back({ids[i], static_cast<int>(i) - 1,
                       i + 1 == ids.size() ? -1 : static_cast<int>(i) + 1});
  }

  struct Bigram {
    std::int64_t rank;
    int left;
    int right;
    TokenId left_id;
    TokenId right_id;
    TokenId merged;
  };
  // Lowest rank first, then the leftmost pair (llama.cpp's comparator).
  const auto later = [](const Bigram& a, const Bigram& b) {
    return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
  };
  std::priority_queue<Bigram, std::vector<Bigram>, decltype(later)> queue(
      later);
  const auto add = [&](int left, int right) {
    if (left < 0 || right < 0) {
      return;
    }
    const Merge merge = FindMerge(symbols[left].id, symbols[right].id);
    if (merge.rank >= 0) {
      queue.push({merge.rank, left, right, symbols[left].id, symbols[right].id,
                  merge.merged});
    }
  };
  for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
    add(i - 1, i);
  }
  while (!queue.empty()) {
    const Bigram b = queue.top();
    queue.pop();
    Symbol& left = symbols[b.left];
    Symbol& right = symbols[b.right];
    // Stale when either side has changed since the pair was queued.
    if (left.id != b.left_id || right.id != b.right_id ||
        left.next != b.right) {
      continue;
    }
    left.id = b.merged;
    left.next = right.next;
    if (right.next >= 0) {
      symbols[right.next].prev = b.left;
    }
    right.id = -1;
    add(left.prev, b.left);
    add(b.left, left.next);
  }

  for (int i = 0; i >= 0 && i < static_cast<int>(symbols.size());
       i = symbols[i].next) {
    out->push_back(symbols[i].id);
  }
}

std::string Tokenizer::TokenText(TokenId token, bool special) const {
  switch (Type(token)) {
    case TokenType::kControl:
    case TokenType::kUnknown:
      return special ? tokens_[token] : std::string();
    case TokenType::kUserDefined:
      return tokens_[token];
    case TokenType::kNormal:
      return UnescapeSymbols(tokens_[token]);
    default:
      return {};
  }
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens,
                              bool special) const {
  std::string out;
  for (TokenId token : tokens) {
    out += TokenText(token, special);
  }
  return out;
}

}  // namespace gufo::models::kolibri1
