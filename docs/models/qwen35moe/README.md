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
  (means near 1) and negative SSM decay rates, so the oracle applies them
  directly with no `+1` or sign flip. Two layouts the code must honor: the
  converter stores SSM value-side rows tiled, so K/V heads pair by tiling
  (`h % k_heads`, as ggml does), while the packed full-attention projection
  stays in HF per-head order `[q0, g0, q1, g1, ...]` and is deinterleaved per
  head — a flat halves split silently scrambles every full layer.
- `reference.cpp` is the scalar oracle of the same graph. It matches an
  independent NumPy implementation of the transformers graph bit-for-bit on a
  synthetic F32 artifact (`reference_test`, max_abs ~1e-5) and is chunk
  invariant. Both oracles agree on real 35B weights through all 40 layers
  (post-MoE residual, worst max_abs 4.77e-06). Parity against served weights
  holds on both files ([parity](#parity-against-served-weights)).
- The HIP engine (`kernels/rocm`, `engine.cpp`) runs the full graph on device:
  upload, grouped GEMV (Q4_K/Q5_K/Q6_K/Q8_0), softmax top-k router, gated
  delta-net recurrence, fused QK-norm/RoPE/split/cache write, online GQA
  attention and the MoE finish, each covered by a device unit test; the
  end-to-end `device_forward` check against the oracle is in progress.
  Tokenizer/chat-template wiring and quality runs are open.

## Parity against served weights

`tools/qwen35moe/parity.cpp` drives the oracle over GGUF token ids and prints
the top-5 per frontier; `compare_parity.py` checks it against the
`/completion` top-logprobs of a llama.cpp server on the same file. Status:
12/12 frontiers agree on the argmax on both the 35B file (top-pair gaps
within ±0.2 nats of float noise) and the 122B UD-Q4_K_XL shards (within
±0.24), and the C++ oracle matches the independent NumPy oracle bit-for-bit
through all 40 layers.

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
