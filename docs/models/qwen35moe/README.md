# Qwen3.5-MoE (`qwen35moe`)

Ornith-1.5 family: hybrid Gated-DeltaNet / full-attention mixture-of-experts
text models. Bring-up vehicle is the 35B-A3B file; the 122B file shares the
graph at larger widths.

| File | Status |
| --- | --- |
| `Ornith-1.5-35B-Q5_K_M.gguf` (single file) | Config + weight binding passes (`qwen35moe.artifact`) |
| `Ornith-1.5-122B-UD-Q4_K_XL` (3 shards) | Config + weight binding passes (`qwen35moe.artifact`) |

## What is verified

`src/models/qwen35moe` binds the released metadata and every tensor of both
files: 40/48 trunk layers in a 3:1 linear-to-full pattern, 256 experts routed
8 per token with a sigmoid-gated shared expert, one `nextn` predictor block,
mRoPE sections `[11, 11, 10]`, zero-centered RMSNorms. The served engine,
tokenizer/chat-template wiring and quality runs are open.

## Sources

- **Architecture of record:** HuggingFace `transformers`
  `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (router: softmax, top-k,
  renormalise; shared expert gated; `attn_q` packs query and gate;
  `RMSNorm` is `(1 + w)` zero-centered).
- **Files:** `unsloth/Ornith-1.5-122B-GGUF` (`UD-Q4_K_XL`, architecture
  `qwen35moe`), `ornith-ai/Ornith-1.5-35B-A3B` (`Q5_K_M`, same key, no
  `attention.recurrent_layers` — the interval rule decides).
