# Qwen3.5-MoE (`qwen35moe`) bring-up

Native engine for the Qwen3.5 hybrid-MoE family on `gfx1151`: Ornith-1.5-122B
(122B total, 256 experts, 8 per token) and Ornith-1.5-35B-A3B (35B total, same
routing). GGUF architecture key `qwen35moe`, Unsloth dynamic quants plus the
`kaivoss` DFlash2 draft.

[Usage](../../../docs/models/qwen35moe/README.md) (planned)

## Source ownership

| Path | Responsibility |
|---|---|
| [`config.hpp`](config.hpp), [`config.cpp`](config.cpp) | Metadata of the `qwen35moe` file: trunk depth, the per-layer linear/full pattern, mRoPE sections, SSM dims, expert counts, and rejection of shapes this runtime does not run. |
| [`weights.hpp`](weights.hpp), [`weights.cpp`](weights.cpp) | Tensor-name binding and per-role format/shape checks over the mapped file; no weight is copied here. |

## Architecture (of record: transformers `modeling_qwen3_5_moe.py`)

- **Hybrid stack, 3:1:** `attention.recurrent_layers` marks each block; trunk
  layer `l` is full attention iff `(l + 1) % full_attention_interval == 0`.
  `block_count` includes one `nextn` MTP predictor block when
  `nextn_predict_layers == 1`.
- **Linear layers** are Gated DeltaNet: `attn_qkv` splits into
  `2 * key_dim + value_dim`, `attn_gate` is the output gate `z`,
  `ssm_alpha`/`ssm_beta`/`ssm_dt` drive the decay and rate,
  `ssm_norm` is the gated RMSNorm, `ssm_out` the output projection.
  `ssm_a` stores the *negative decay rates* (`-A`), not the checkpoint's
  `A_log`: the graph multiplies them with `softplus(a + dt)` directly, no
  `exp`, no negation. The converter bakes `-exp()` in.
- **Full layers** pack query and gate (`attn_q` is `[hidden, 2 * heads *
  head_dim]`), carry per-head-dim Q/K norms, rotate the first `rotary_dim`
  elements with mRoPE sections `[11, 11, 10]`, and gate the attention output
  with `sigmoid(gate)` before `attn_output`.
- **Mixture, every trunk layer:** `ffn_gate_inp` is the F32 router matrix,
  selection is `top-k(softmax(logits))` renormalised to sum 1 (no bias, no
  `expert_gating_func` metadata); the shared expert is
  `sigmoid(ffn_gate_inp_shexp . x) * MLP_shexp(x)`.
- **Norms are standard RMSNorms on file:** `y = rms(x) * w`. The transformers
  checkpoint centres them at 0 and reads `(1 + w)`, but the GGUF converter
  bakes that shift in: every norm tensor in both files is all-positive with
  mean near 1 (e.g. `attn_q_norm` in [0.61, 2.13]). A runtime that adds 1
  again amplifies every branch 2-4x and diverges. The shared-expert gate
  vector is not a norm (mean 0) and is used raw.
- **Vocabulary:** 248320 entries, the Qwen3.5/3.8 vocabulary.

## Status

Config + weight binding with host tests (`tests/models/qwen35moe/`). The HIP
engine, tokenizer/chat-template wiring, serve dispatch and quality runs are
open. Bring-up order: 35B-Q5_K_M first (single 24 GB file, same graph), then
the 122B UD-Q4_K_XL shards.
