#ifndef GUFO_MODELS_KOLIBRI1_TOKENIZER_HPP_
#define GUFO_MODELS_KOLIBRI1_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::kolibri1 {

using TokenId = std::int32_t;

/// GGUF `tokenizer.ggml.token_type` values.
enum class TokenType : std::uint8_t {
  kUndefined = 0,
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// The `kolibri1` tokenizer: GPT-2 byte-level BPE (`tokenizer.ggml.model =
/// gpt2`) over the qwen2 split (`tokenizer.ggml.pre = kolibri1`).
///
/// The vocabulary is byte-level: each of the 256 bytes is stored as its
/// GPT-2 `bytes_to_unicode` symbol, and the 127644 merges concatenate those
/// symbols. Consequently text is taken verbatim - no space prefix, no
/// normalization, no U+FFFD rewrite for invalid UTF-8 and no `<0xXX>` byte
/// tokens, which this vocabulary does not contain. Decoding runs the symbol
/// text back through the same table.
///
/// Before merging, text is split by llama.cpp's `qwen2` pre-tokenizer:
/// contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), maximal letter
/// runs with at most one leading non-letter, single numeric code points,
/// punctuation runs with their trailing newlines, whitespace runs that reach a
/// newline, and trailing whitespace keeps one code point for the next piece.
/// Merging always takes the lowest-ranked adjacent pair, leftmost on a tie.
///
/// Specials are partitioned out of the text first, longest text first: control
/// tokens only when special text is parsed, user-defined tokens always. The
/// vocabulary carries no BOS token at all, so a request to prepend one is a
/// no-op. End-of-generation tokens behave as control tokens.
class Tokenizer {
public:
  struct Vocabulary {
    std::vector<std::string> tokens;
    std::vector<TokenType> types;
    std::vector<std::string> merges;  ///< "left right", rank = index.
    TokenId bos{-1};                  ///< -1 when the file names none.
    TokenId eos{-1};
    TokenId pad{-1};
    TokenId unknown{-1};  ///< -1 when the file names none.
    bool add_bos{false};
  };

  /// Rejects anything but `tokenizer.ggml.model = gpt2` and
  /// `tokenizer.ggml.pre = kolibri1`: a different pre-tokenizer would split
  /// words differently and silently change every id.
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
  [[nodiscard]] static std::unique_ptr<Tokenizer> Create(
      Vocabulary vocabulary, std::string* error_msg = nullptr);

  /// `parse_special` also matches control tokens in the text; user-defined
  /// tokens are always matched. `add_bos` follows the vocabulary flag, which
  /// the Kolibri 1 file clears.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text, bool add_bos,
                                            bool parse_special) const;

  /// Text of one token, with the GPT-2 byte symbols resolved back to bytes.
  /// Control and unknown tokens render only with `special`; user-defined
  /// tokens always render. A character may span several tokens.
  [[nodiscard]] std::string TokenText(TokenId token, bool special) const;
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens,
                                   bool special) const;

  [[nodiscard]] std::size_t VocabSize() const noexcept {
    return tokens_.size();
  }
  [[nodiscard]] TokenId BosToken() const noexcept { return bos_; }
  [[nodiscard]] TokenId EosToken() const noexcept { return eos_; }
  [[nodiscard]] bool IsEndOfGeneration(TokenId token) const noexcept;
  [[nodiscard]] std::span<const TokenId> EndOfGenerationTokens()
      const noexcept {
    return eog_;
  }
  [[nodiscard]] std::optional<TokenId> FindToken(
      std::string_view text) const noexcept;
  [[nodiscard]] TokenType Type(TokenId token) const noexcept;

private:
  Tokenizer() = default;

  /// Splits `text` into pre-tokenizer pieces and merges each one.
  void EncodePieces(std::string_view text, std::vector<TokenId>* out) const;
  /// Byte-level BPE over a single piece.
  void EncodePiece(std::string_view piece, std::vector<TokenId>* out) const;
  /// A pair's merge: its rank (-1 when the pair never merges) and the token
  /// of the concatenated text.
  struct Merge {
    std::int64_t rank;
    TokenId merged;
  };
  [[nodiscard]] Merge FindMerge(TokenId left, TokenId right) const noexcept;

  std::vector<std::string> tokens_;
  std::vector<TokenType> types_;
  std::unordered_map<std::string_view, TokenId> token_to_id_;
  /// Open-addressing merge table keyed by the (left, right) id pair; the
  /// lowest rank wins for a repeated pair.
  struct MergeSlot {
    std::uint64_t key;
    std::uint32_t rank;
    TokenId merged;
  };
  std::vector<MergeSlot> merges_;
  std::uint32_t merge_shift_{64};
  /// Token of each byte, addressed through its GPT-2 symbol, else -1.
  std::array<TokenId, 256> byte_tokens_{};

  /// Recently encoded long pieces (a run of letters or of punctuation is one
  /// unbounded piece), least recently used first: a conversation resends every
  /// earlier word with each turn.
  struct CachedPiece {
    std::string text;
    std::vector<TokenId> ids;
  };
  mutable std::mutex piece_cache_mutex_;
  mutable std::list<CachedPiece> piece_cache_;
  mutable std::unordered_map<std::string_view, std::list<CachedPiece>::iterator>
      piece_index_;
  mutable std::size_t piece_cache_bytes_{0};

  std::vector<TokenId> special_;  ///< Partition order: longest text first.
  std::vector<TokenId> eog_;
  TokenId bos_{-1};
  TokenId eos_{-1};
  bool add_bos_{false};
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_TOKENIZER_HPP_
