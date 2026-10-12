#!/usr/bin/env python3
"""Regenerate the kolibri1 tokenizer fixtures.

    python3 tests/models/kolibri1/make_tokenizer_fixture.py

Writes `fixtures/tokenizer_vocab.json` (a small synthetic vocabulary) and
`fixtures/tokenizer_golden.json` (expected token ids for a fixed input set).
Standard library only, no network, no model file: it is deterministic, so
re-running it must produce byte-identical output.

Why a reimplementation instead of the real tokenizer
----------------------------------------------------
Kolibri 1 ships `tokenizer.ggml.model = gpt2` (byte-level BPE) with
`tokenizer.ggml.pre = kolibri1`. That spelling does not exist upstream, so the
authoritative split is the one the vocabulary was built for: llama.cpp's
`LLAMA_VOCAB_PRE_TYPE_QWEN2` expression, quoted verbatim in the `qwen2`
section below and implemented a second time, independently, by
`src/models/kolibri1/tokenizer.cpp`. The two must agree; that agreement is
what `kolibri1.tokenizer` checks.

Translating the regex without third-party packages
--------------------------------------------------
Python's `re` has no `\\p{...}` classes and `\\w` is NOT `\\p{L}` (it adds
`\\p{N}`, `\\p{M}` and the connector punctuation `_`), so every Unicode class
becomes an explicit code-point range built from `unicodedata.category`:

  `\\p{L}`  -> categories Lu, Ll, Lt, Lm, Lo
  `\\p{N}`  -> categories Nd, Nl, No
  `\\s`     -> the Unicode White_Space property (see `WHITE_SPACE`). RE2's
              `\\s` covers the same set for every code point the fixture uses;
              ICU's `u_isUWhiteSpace`, which the C++ side calls, is that same
              property. U+001C..U+001F, which RE2 also folds into `\\s`, are
              deliberately excluded here and on the C++ side, so both sides
              agree.

Alternation order is preserved: Python's `re` is leftmost-first, like RE2's
default (non-POSIX) matcher, which is what llama.cpp uses.

Invalid UTF-8
-------------
RE2 walks invalid bytes as one `RuneError` (U+FFFD) each, so an invalid byte
is neither a letter nor a number nor whitespace and lands in the punctuation
branch. The scanner below mirrors that: strict, shortest-form UTF-8 only, one
U+FFFD per rejected byte, and the byte ranges are kept so BPE still runs over
the original bytes (the vocabulary maps every byte to its `bytes_to_unicode`
symbol, so nothing is lost). `tokenizer.cpp` does the same.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata

# --------------------------------------------------------------------------
# GPT-2 byte-level alphabet
# --------------------------------------------------------------------------


def bytes_to_unicode() -> dict:
    """The standard GPT-2 table: byte -> the one-character symbol the vocab
    uses for it. Printable ASCII and U+00A1..U+00AC, U+00AE..U+00FF map to
    themselves; every other byte maps into U+0100.., so all 256 symbols are
    printable and no byte is unrepresentable."""
    kept = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(
        range(0xAE, 0x100))
    bytes_list = list(kept)
    codes = list(kept)
    extra = 0
    for byte in range(256):
        if byte not in bytes_list:
            bytes_list.append(byte)
            codes.append(0x100 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(bytes_list, codes)}


BYTE_TO_SYMBOL = bytes_to_unicode()
SYMBOL_TO_BYTE = {symbol: byte for byte, symbol in BYTE_TO_SYMBOL.items()}
BYTE_SYMBOLS = [BYTE_TO_SYMBOL[byte] for byte in range(256)]


def escape(text: str) -> str:
    """Text as the vocabulary stores it: every byte through the table."""
    return "".join(BYTE_TO_SYMBOL[byte] for byte in text.encode("utf-8"))


def unescape(symbol_text: str) -> bytes:
    out = bytearray()
    for char in symbol_text:
        out.append(SYMBOL_TO_BYTE[char])
    return bytes(out)


# --------------------------------------------------------------------------
# Unicode classes for the split expression
# --------------------------------------------------------------------------

LETTER_CATEGORIES = frozenset(("Lu", "Ll", "Lt", "Lm", "Lo"))
NUMBER_CATEGORIES = frozenset(("Nd", "Nl", "No"))

# The Unicode White_Space property, listed because `unicodedata` does not
# expose it: Zs/Zl/Zp plus the Cc members that carry it.
WHITE_SPACE = (
    list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680] +
    list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


def _code_points(categories: frozenset) -> list:
    return [cp for cp in range(0x110000)
            if unicodedata.category(chr(cp)) in categories]


def _escape(cp: int) -> str:
    """Code point as a regex escape, so no metacharacter can leak."""
    return "\\u%04X" % cp if cp < 0x10000 else "\\U%08X" % cp


def _class_body(code_points: list) -> str:
    """Code points as a regex character-class body, run-length collapsed."""
    parts = []
    start = previous = code_points[0]
    for cp in code_points[1:] + [None]:
        if cp is not None and cp == previous + 1:
            previous = cp
            continue
        parts.append(_escape(start) if start == previous else
                     "%s-%s" % (_escape(start), _escape(previous)))
        if cp is not None:
            start = previous = cp
    return "".join(parts)


def build_split_pattern() -> str:
    """llama.cpp's qwen2 expression, classes translated.

    qwen2 (src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN2):
      (?i:'s|'t|'re|'ve|'m|'ll|'d)|
      [^\\r\\n\\p{L}\\p{N}]?\\p{L}+|
      \\p{N}|
       ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|
      \\s*[\\r\\n]+|
      \\s+(?!\\S)|
      \\s+
    """
    letter = "[" + _class_body(_code_points(LETTER_CATEGORIES)) + "]"
    number = "[" + _class_body(_code_points(NUMBER_CATEGORIES)) + "]"
    space = "[" + _class_body(sorted(set(WHITE_SPACE))) + "]"
    not_letter_number = ("[^\\r\\n" + letter[1:-1] + number[1:-1] + "]")
    not_space_letter_number = ("[^" + space[1:-1] + letter[1:-1] +
                               number[1:-1] + "]")
    not_space = "[^" + space[1:-1] + "]"
    return (
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        + "|" + not_letter_number + "?" + letter + "+"
        + "|" + number
        + "| ?" + not_space_letter_number + "+[\\r\\n]*"
        + "|" + space + "*[\\r\\n]+"
        + "|" + space + "+(?!" + not_space + ")"
        + "|" + space + "+")


SPLIT_PATTERN = build_split_pattern()
SPLIT_RE = re.compile(SPLIT_PATTERN)


# --------------------------------------------------------------------------
# Strict UTF-8 scanning with byte ranges
# --------------------------------------------------------------------------


def scan(text: bytes) -> tuple:
    """Return `(scan_text, offsets)`.

    `scan_text[i]` is the code point RE2 would see at the i-th rune: the
    decoded character, or U+FFFD for a rejected byte (overlong forms,
    surrogates, out-of-range values, truncated and stray continuation bytes
    all reject one byte at a time). `offsets[i] = (byte_start, byte_length)`
    maps that rune back to the input.
    """
    chars = []
    offsets = []
    i = 0
    n = len(text)
    while i < n:
        lead = text[i]
        if lead < 0x80:
            chars.append(chr(lead))
            offsets.append((i, 1))
            i += 1
            continue
        if lead < 0xC2 or lead > 0xF4:
            chars.append("\uFFFD")
            offsets.append((i, 1))
            i += 1
            continue
        length = 2 if lead < 0xE0 else (3 if lead < 0xF0 else 4)
        if i + length > n:
            chars.append("\uFFFD")
            offsets.append((i, 1))
            i += 1
            continue
        continuations = text[i + 1:i + length]
        if any(byte & 0xC0 != 0x80 for byte in continuations):
            chars.append("\uFFFD")
            offsets.append((i, 1))
            i += 1
            continue
        if length == 2:
            code = ((lead & 0x1F) << 6) | (continuations[0] & 0x3F)
            valid = code >= 0x80
        elif length == 3:
            code = (((lead & 0x0F) << 12) | ((continuations[0] & 0x3F) << 6) |
                    (continuations[1] & 0x3F))
            valid = code >= 0x800 and not 0xD800 <= code <= 0xDFFF
        else:
            code = (((lead & 0x07) << 18) | ((continuations[0] & 0x3F) << 12) |
                    ((continuations[1] & 0x3F) << 6) | (continuations[2] & 0x3F))
            valid = 0x10000 <= code <= 0x10FFFF
        if not valid:
            chars.append("\uFFFD")
            offsets.append((i, 1))
            i += 1
            continue
        chars.append(chr(code))
        offsets.append((i, length))
        i += length
    return "".join(chars), offsets


def pieces_of(text: bytes) -> list:
    """Pre-tokenizer pieces as `(start, length)` byte ranges of `text`."""
    scan_text, offsets = scan(text)
    out = []
    for match in SPLIT_RE.finditer(scan_text):
        start, end = match.span()
        byte_start = offsets[start][0]
        byte_end = offsets[end - 1][0] + offsets[end - 1][1]
        out.append((byte_start, byte_end - byte_start))
    # The expression covers every rune, so the ranges tile the input.
    cursor = 0
    for start, length in out:
        assert start == cursor, (start, cursor, text)
        cursor = start + length
    assert cursor == len(text), (cursor, len(text), text)
    return out


# --------------------------------------------------------------------------
# GPT-2 BPE over one piece
# --------------------------------------------------------------------------


def merge_ranks(merges: list) -> dict:
    """Merge text ("left right") -> rank, first occurrence winning."""
    ranks = {}
    for rank, merge in enumerate(merges):
        left, right = merge.split(" ", 1)
        ranks.setdefault((left, right), rank)
    return ranks


def bpe(piece: bytes, ranks: dict, token_ids: dict) -> list:
    """Lowest-rank adjacent pair first, leftmost on a tie, exactly once per
    step. Symbols are vocabulary texts; a pair without a rank never merges."""
    symbols = [BYTE_TO_SYMBOL[byte] for byte in piece]
    if len(symbols) < 2:
        return [token_ids[symbol] for symbol in symbols]
    while True:
        best_rank = None
        best = -1
        for index in range(len(symbols) - 1):
            rank = ranks.get((symbols[index], symbols[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best = index
        if best < 0:
            break
        merged = symbols[best] + symbols[best + 1]
        if merged not in token_ids:
            break
        symbols[best:best + 2] = [merged]
    return [token_ids[symbol] for symbol in symbols]


def encode(text: bytes, ranks: dict, token_ids: dict) -> list:
    out = []
    for start, length in pieces_of(text):
        out.extend(bpe(text[start:start + length], ranks, token_ids))
    return out


# --------------------------------------------------------------------------
# Training the small synthetic vocabulary
# --------------------------------------------------------------------------


def train_merges(texts: list, max_merges: int, min_freq: int = 2) -> list:
    """Bottom-up BPE training over the fixture corpus, so the merge ladder is
    a real one instead of a hand-written guess. Deterministic: the most
    frequent pair wins, ties by the lexicographically smallest pair."""
    words = []
    for raw in texts:
        words.append(tuple(BYTE_TO_SYMBOL[byte] for byte in raw))
    merges = []
    while len(merges) < max_merges:
        counts = {}
        for word in words:
            for index in range(len(word) - 1):
                pair = (word[index], word[index + 1])
                counts[pair] = counts.get(pair, 0) + 1
        candidates = [(count, pair) for pair, count in counts.items()
                      if count >= min_freq]
        if not candidates:
            break
        count, pair = min(candidates, key=lambda item: (-item[0], item[1]))
        merged = pair[0] + pair[1]
        merges.append("%s %s" % (pair[0], pair[1]))
        words = [_apply_pair(word, pair, merged) for word in words]
    return merges


def _apply_pair(word: tuple, pair: tuple, merged: str) -> tuple:
    out = []
    index = 0
    while index < len(word):
        if word[index:index + 2] == pair:
            out.append(merged)
            index += 2
        else:
            out.append(word[index])
            index += 1
    return tuple(out)


# Control tokens of the real vocabulary that the fixture reproduces, in file
# order. `<|endoftext|>` is `tokenizer.ggml.padding_token_id` (127901 there)
# and `<|im_end|>` is `tokenizer.ggml.eos_token_id` (127906 there); the file
# carries no `bos_token_id` and no `unknown_token_id`.
CONTROL_TEXTS = [
    "<|text|>",
    "<|endoftext|>",
    "<|pad|>",
    "<|chat|>",
    "<|im_start|>",
    "<|/role|>",
    "<|im_end|>",
    "<|reserved-token-2|>",
]
USER_DEFINED_TEXTS = [
    "<think>",
    "\n</think>\n\n",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]
EOS_TEXT = "<|im_end|>"
PAD_TEXT = "<|endoftext|>"
# End-of-generation set: the file's eos plus the controls a turn can close
# with (`<|/role|>`) and the sequence-end control (`<|endoftext|>`).
EOG_TEXTS = [EOS_TEXT, "<|/role|>", PAD_TEXT]

FIXED_INPUTS = [
    ("ascii", "Hello world, this is a test."),
    ("german", "Die Gr\u00f6\u00dfe der Stra\u00dfe ist \u00fcberaus sch\u00f6"
               "n \u2014 wei\u00dft du?"),
    ("emoji", "Feier \U0001f389 mit \U0001f468\u200d\U0001f469\u200d"
              "\U0001f467 und \U0001f1e9\U0001f1ea Flaggen"),
    ("code", "def f(x):\n    if x != 3:\n        return 'a'\telse: \"b\"\n"),
    ("space_runs", "a  \n\n  b\t\tc\r\n\nd"),
    ("trailing_spaces", "word       "),
    ("contractions", "don't stop, we've'd'll say it's mine"),
    ("digits", "1234567890 and 42 items, 7\u00b0C"),
    ("controls_in_text",
     "before <|im_start|>assistant\nhello<|im_end|>\nafter <|pad|> and "
     "<tool_call>\n{}\n</tool_call> done <think>think</think> "
     "<tool_response>r</tool_response>"),
    ("invalid_utf8", "\xff" "caf\xff\xe9" "\xc3" "\x80\x81" "na\xc3\xa4ve"
                     "\xe0\x80\xaf" "end"),
    ("long_word", "supercalifragilisticexpialidocious" * 4),
    ("mixed_script", "\u4f60\u597d \u041f\u0440\u0438\u0432\u0435\u0442 "
                     "\u05e9\u05dc\u05d5\u05dd caf\u00e9 \u1e63\u0101\u1e63"),
]


def raw_of(name: str, text: str) -> bytes:
    """The golden inputs are written as str for readability; the invalid-UTF-8
    case is spelled as latin-1 escapes so its bytes survive verbatim."""
    return text.encode("latin-1") if name == "invalid_utf8" else \
        text.encode("utf-8")



# --------------------------------------------------------------------------
# Deterministic split fuzz
# --------------------------------------------------------------------------

# Atoms chosen so that every branch of the split expression is reachable:
# letter runs, numeric code points, whitespace runs of all shapes, apostrophe
# contractions, marks (which are NOT letters for this expression), symbols and
# the byte sequences a strict UTF-8 decoder rejects.
FUZZ_ATOMS = [
    b"a", b"b", b"Z", b"\xc3\x9f",            # a b Z ss
    b"\xc3\xb6", b"\xce\xa9",                  # o-umlaut Omega
    b"\xd0\xb5", b"\xd7\x90",                  # Cyrillic ue, Hebrew alef
    b"\xe4\xb8\xad",                           # CJK
    b"0", b"7",
    b" ", b"  ", b"   ", b"\n", b"\r\n", b"\t", b"\n\n", b" \n ", b"\r",
    b".", b",", b"(", b"<", b"|", b">", b"'", b"-",
    b"\xe2\x80\x94",                          # em dash
    b"'s", b"'ll", b"'T", b"n\xe2\x80\x99t",   # contractions, incl. U+2019
    b"\xf0\x9f\x8e\x89",                      # party popper
    b"\xf0\x9f\x87\xa9\xf0\x9f\x87\xaa",  # regional pair
    b"\xe2\x80\x8d",                          # zero width joiner (Cf)
    b"\xcc\x81", b"e\xcc\x81",                # combining acute (Mn)
    b"\xff", b"\x80", b"\xe0\x80\xaf", b"\xc3", b"\xed\xa0\x80",
    b"<|", b"<|im_start|>", b"<tool_call>", b"im",
]


class _Split64:
    """splitmix64: deterministic across Python versions, unlike `random`."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return self.next() % bound


def fuzz_inputs(count: int = 150, seed: int = 0x20261012) -> list:
    """`(name, bytes)` pairs of random atom sequences."""
    rng = _Split64(seed)
    out = []
    for index in range(count):
        raw = b"".join(
            FUZZ_ATOMS[rng.below(len(FUZZ_ATOMS))]
            for _ in range(1 + rng.below(12)))
        out.append(("fuzz-%03d" % index, raw))
    return out


def corpus() -> list:
    """Everything the vocabulary must cover: the golden inputs plus plain
    words, so the trained ladder reaches real multi-character tokens."""
    texts = [raw_of(name, text) for name, text in FIXED_INPUTS]
    texts += [
        " the hello world this is a test of tokenizer German stra\u00dfe "
        "gr\u00f6\u00dfe sch\u00f6n \u00fcber",
        " function return value name arguments name name arguments",
        " one two three four five six seven eight nine ten",
        " system user assistant tool response think call",
        "  \n \n\n \t \r\n \n\n\n",
        " word       word      word",
        " don't we've it's we'd that'll",
        " 12345 42 7 1234567890 42 12345",
        " caf\u00e9 na\u00efve \u00dcber \u00d6l",
    ]
    return [text if isinstance(text, bytes) else text.encode("utf-8")
            for text in texts]


def build_fixture() -> tuple:
    merges = train_merges(corpus(), max_merges=140)
    trained = []
    for merge in merges:
        left, right = merge.split(" ", 1)
        trained.append(left + right)
    tokens = list(BYTE_SYMBOLS)
    for text in trained:
        if text not in tokens:
            tokens.append(text)
    specials = []
    for text in CONTROL_TEXTS + USER_DEFINED_TEXTS:
        if text not in tokens:
            tokens.append(text)
            specials.append(text)
    # A trained token could collide with a special text; keep the special's
    # type by dropping the duplicate merge instead.
    assert not any(escape(text) in trained for text in CONTROL_TEXTS +
                   USER_DEFINED_TEXTS), "special text collides with a merge"
    del specials
    token_ids = {}
    for token_id, text in enumerate(tokens):
        token_ids.setdefault(text, token_id)
    ranks = merge_ranks(merges)

    vocabulary = {
        "schema": "gufo.kolibri1-tokenizer-vocab.v1",
        "note": ("Synthetic gpt2 byte-level vocabulary for hermetic tests: "
                 "ids 0..255 are the bytes_to_unicode symbols, then merges "
                 "trained on the golden corpus, then the control and "
                 "user-defined tokens of the real vocabulary in file order."),
        "model": "gpt2",
        "pre": "kolibri1",
        "bos": None,
        "eos_text": EOS_TEXT,
        "pad_text": PAD_TEXT,
        "unknown_text": None,
        "add_bos": False,
        "control_texts": CONTROL_TEXTS,
        "user_defined_texts": USER_DEFINED_TEXTS,
        "eog_texts": EOG_TEXTS,
        "tokens": tokens,
        "merges": merges,
    }

    entries = []
    inputs = [(name, raw_of(name, text)) for name, text in FIXED_INPUTS]
    inputs += fuzz_inputs()
    for name, raw in inputs:
        entries.append({
            "name": name,
            "text_hex": raw.hex(),
            "pieces_hex": [piece.hex() for piece in pieces_of_text(raw,
                                                                   tokens)],
            # `parse_special` off still resolves user-defined tokens; control
            # tokens stay ordinary text and are split like any other symbol.
            "ids_plain": encode_with_specials(raw, ranks, token_ids, tokens,
                                              kinds=(4,)),
            "ids_parse_special": encode_with_specials(raw, ranks, token_ids,
                                                      tokens),
        })
    golden = {
        "schema": "gufo.kolibri1-tokenizer-goldens.v1",
        "reference": ("stdlib GPT-2 byte-level BPE plus the llama.cpp qwen2 "
                      "split expression; see make_tokenizer_fixture.py"),
        "fuzz": ("the entries named fuzz-* are atom sequences drawn from "
                 "FUZZ_ATOMS with splitmix64(0x20261012); the alphabet is "
                 "fixed, so the selection is reproducible"),
        "pre_tokenizer": "llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2",
        "columns": ("ids_plain is Encode(text, add_bos=false, "
                    "parse_special=false) so only user-defined tokens resolve; "
                    "ids_parse_special is Encode(text, false, true) so control "
                    "tokens resolve too; pieces_hex is the split as `Encode` "
                    "sees it: user-defined tokens whole, every other span "
                    "split by the qwen2 expression, and merges never cross a "
                    "piece"),
        "add_bos": False,
        "entries": entries,
    }
    return vocabulary, golden


def find_specials(tokens: list, types: dict, kinds: tuple = (3, 4)) -> list:
    """(text, id) pairs, longest text first, id order for equal lengths, as
    `Tokenizer::Create` partitions them. `kinds` selects the token types that
    may match: `Encode` always matches user-defined (4) tokens and adds
    control (3) tokens only when special text is parsed."""
    specials = [(token, token_id) for token_id, token in enumerate(tokens)
                if types[token_id] in kinds and token]
    specials.sort(key=lambda item: -len(item[0]))
    return specials


def types_of(vocabulary_tokens: list, control_texts: list,
             user_texts: list) -> dict:
    types = {}
    for token_id, token in enumerate(vocabulary_tokens):
        if token in user_texts:
            types[token_id] = 4
        elif token in control_texts:
            types[token_id] = 3
        else:
            types[token_id] = 1
    return types


def split_fragments(raw: bytes, specials: list) -> list:
    """`raw` cut at every occurrence of a special text, longest special first;
    a special becomes a `(text, id)` tuple, everything else stays `bytes`."""
    fragments = [raw]
    for text, token_id in specials:
        needle = text.encode("utf-8")
        next_fragments = []
        for fragment in fragments:
            if isinstance(fragment, tuple):
                next_fragments.append(fragment)
                continue
            start = 0
            while True:
                index = fragment.find(needle, start)
                if index < 0:
                    break
                if index > start:
                    next_fragments.append(fragment[start:index])
                next_fragments.append((needle, token_id))
                start = index + len(needle)
            if start < len(fragment):
                next_fragments.append(fragment[start:])
        fragments = next_fragments
    return fragments


def encode_with_specials(raw: bytes, ranks: dict, token_ids: dict,
                         tokens: list, kinds: tuple = (3, 4)) -> list:
    """Partition specials longest-text-first, then encode the remaining
    spans; `kinds` is the set of token types allowed to match."""
    types = types_of(tokens, CONTROL_TEXTS, USER_DEFINED_TEXTS)
    out = []
    for fragment in split_fragments(raw, find_specials(tokens, types, kinds)):
        if isinstance(fragment, tuple):
            out.append(fragment[1])
        else:
            out.extend(encode(fragment, ranks, token_ids))
    return out


def pieces_of_text(raw: bytes, tokens: list) -> list:
    """The pre-tokenizer split the way `Encode(text, false, false)` sees it:
    user-defined tokens are partitioned out first and stay whole, every
    remaining span is split by the qwen2 expression on its own."""
    types = types_of(tokens, CONTROL_TEXTS, USER_DEFINED_TEXTS)
    pieces = []
    for fragment in split_fragments(raw, find_specials(tokens, types, (4,))):
        if isinstance(fragment, tuple):
            pieces.append(fragment[0])
        else:
            pieces.extend(raw_span[start:start + length]
                          for raw_span, start, length in
                          ((fragment, start, length)
                           for start, length in pieces_of(fragment)))
    return pieces


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "fixtures")
    os.makedirs(fixtures, exist_ok=True)
    vocabulary, golden = build_fixture()
    for name, payload in (("tokenizer_vocab.json", vocabulary),
                          ("tokenizer_golden.json", golden)):
        path = os.path.join(fixtures, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1,
                      sort_keys=False)
            handle.write("\n")
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    print("vocabulary %d tokens, %d merges" % (len(vocabulary["tokens"]),
                                               len(vocabulary["merges"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
