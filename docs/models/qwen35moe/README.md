# Qwen3.5-MoE (`qwen35moe`)

Ornith-1.5 family: hybrid Gated-DeltaNet / full-attention mixture-of-experts
text models. Bring-up vehicle is the 35B-A3B file; the 122B file shares the
graph at larger widths.

| File | Status |
| --- | --- |
| `Ornith-1.5-35B-Q5_K_M.gguf` (single file) | Config + weight binding passes (`qwen35moe.artifact`) |
| `Ornith-1.5-122B-UD-Q4_K_XL` (3 shards) | Config + weight binding passes (`qwen35moe.artifact`) |

## What is verified

- `src/models/qwen35moe` binds the released metadata and every tensor of both
  files: 40/48 trunk layers in a 3:1 linear-to-full pattern, 256 experts routed
  8 per token with a sigmoid-gated shared expert, one `nextn` predictor block,
  mRoPE sections `[11, 11, 10]`, zero-centered RMSNorms.
- `reference.cpp` is the scalar oracle of the same graph. It matches an
  independent NumPy implementation of the transformers graph bit-for-bit on a
  synthetic F32 artifact (`reference_test`, max_abs ~1e-5) and is chunk
  invariant. Parity of the oracle against served weights is open
  ([parity](#parity-against-served-weights)).
- The served engine, tokenizer/chat-template wiring and quality runs are open.

## Parity against served weights

`tools/qwen35moe/parity.cpp` drives the oracle over GGUF token ids and prints
the top-5 per frontier; `compare_parity.py` checks it against the
`/completion` top-logprobs of a llama.cpp server on the same file. Status:
12/12 frontiers mismatch on the 122B UD-Q4_K_XL (oracle argmax absent from the
reference top-5), while the same driver matches the NumPy fixture 6/6 — the
graph is right, a quantized weight path (Q8_0/Q4_K/IQ4_XS, absent from the F32
fixture) is the suspect. The 35B file (Q5_K/Q6_K/F32 only) is the control.

Note the checkpoint on the M1 box is not this file: ollama's
`ornith-1.5:122b` blob reports `general.name = Qwen3.5-122B-A10B-ck2100-...`
(`size_label 256x6.7B`, Q4_K/Q6_K recipe) while the Unsloth file reports
`Ornith 1.5 122B`. Same `qwen35moe` graph and dims — the binder accepts both —
but different weights, so cross-checks must name the file, never the tag.

## Sources

- **Architecture of record:** HuggingFace `transformers`
  `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (router: softmax, top-k,
  renormalise; shared expert gated; `attn_q` packs query and gate;
  `RMSNorm` is `(1 + w)` zero-centered).
- **Files:** `unsloth/Ornith-1.5-122B-GGUF` (`UD-Q4_K_XL`, architecture
  `qwen35moe`), `ornith-ai/Ornith-1.5-35B-A3B` (`Q5_K_M`, same key, no
  `attention.recurrent_layers` — the interval rule decides).
