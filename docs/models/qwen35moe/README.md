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
  mRoPE sections `[11, 11, 10]`. The file stores standard RMSNorm weights
  (means near 1, e.g. `attn_norm` 1.03/0.92) and negative SSM decay rates, so
  the oracle applies them directly with no `+1` or sign flip.
- `reference.cpp` is the scalar oracle of the same graph. It matches an
  independent NumPy implementation of the transformers graph bit-for-bit on a
  synthetic F32 artifact (`reference_test`, max_abs ~1e-5) and is chunk
  invariant. Both oracles agree layer by layer on real 35B weights through
  layer 3 (post-MoE residual, max_abs ~1e-8); the full 40-layer check is
  running. Parity of the oracle against served weights is open
  ([parity](#parity-against-served-weights)).
- The served engine, tokenizer/chat-template wiring and quality runs are open.

## Parity against served weights

`tools/qwen35moe/parity.cpp` drives the oracle over GGUF token ids and prints
the top-5 per frontier; `compare_parity.py` checks it against the
`/completion` top-logprobs of a llama.cpp server on the same file. Status:
the oracle (top-1 `3201` for prompt `[760]`) disagrees with both the ollama
fork (`:40111`, argmax `328`) and upstream llama.cpp master CPU (`:40112`,
argmax `328`, 9/10 shared top-10) on the same 35B file (md5-identical both
sides). Cache pollution is ruled out (fresh server, first request identical).

Root cause (source-level, no weights needed): the GDN k/v-head pairing for
unequal heads (16 k-heads, 32 v-heads). The training stack pairs adjacently —
FLA `fused_recurrent` uses `i_h = i_hv // (HV // H)`, transformers
`qwen3_5_moe`/`qwen3_next` use `repeat_interleave` — and the oracle follows
it. llama.cpp pairs by tiling instead (`iq1 = iv1 % neq1` in the fused
kernel, `ggml_repeat_4d` in the fallback), which is a different computation.
A temporary switch of the oracle to tiling moves its top-1 (`3201` →
`193925`) but matches neither implementation, confirming the pairing is load
bearing and that the oracle tracks the training stack. The native engine must
keep the adjacent pairing; parity against llama.cpp needs the upstream fix.

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
