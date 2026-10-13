# Kolibri-1

Aleph Alpha's Kolibri-1: a 78-billion-parameter mixture-of-experts reasoning
model for German and English with 3.46 billion active parameters per token,
Apache-2.0. This card covers Hob-forge's independent GGUF conversion,
[`Hob-forge/Kolibri-1-GGUF`](https://huggingface.co/Hob-forge/Kolibri-1-GGUF),
whose GGUF architecture key is `kolibri1`. Stock llama.cpp does not run this
architecture; the conversion ships a patch for it. Gufo implements it natively.

**Status.** The engine, its device check against a scalar oracle, the chat
renderer and the throughput tool are in the tree and pass on `gfx1151`. The
`gufo chat` / `gufo serve` dispatch for this architecture is not wired yet;
until it is, run the tools under `tools/kolibri/` described below. The measured
numbers in [BENCHMARKS.md](BENCHMARKS.md) come from the engine, not from HTTP.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Architecture

- **Layers:** 50, hidden width 2560, 48 query heads, 4 KV heads, head dim 128.
  Q projection is 2560 → 6144, K/V are 2560 → 512.
- **Attention pattern:** four sliding layers, then one without any positional
  encoding — layers 4, 9, …, 49. Sliding layers attend over a 513-token window
  and rotate Q/K by RoPE (`rope.freq_base` from the file); the patternless
  layers see the whole past and are not rotated. RMSNorm eps 1e-6.
- **Per-head QK norm:** Q and K are normalized per head (a 128-wide weight each)
  before the scores.
- **Mixture:** 384 routed experts per layer, 6 per token, each a 512-wide SwiGLU
  (separate gate, up and down). Selection maximizes `logit + expert_bias`, the
  contribution weights are the plain sigmoids of the selected logits, **not**
  renormalized (`norm_topk_prob: false`), so the experts of a token can sum well
  past 1. One ungated shared expert of the same width is always added at weight 1.
- **Sandwich norms:** each of the two branches is normalized before (`attn_norm`,
  `ffn_norm`) and after (`post_attention_norm`, `post_ffw_norm`), and the residual
  is added after the second norm.
- **Vocabulary:** 128,000, SentencePiece Unigram with byte fallback. The released
  artifact ships an untied `output.weight`; a tied file binds the same view.
  Max context 262,144.

### Quantizations

The published Q4_K_M keeps attention and the routed/shared gate-up halves in
Q4_K, down projections in Q6_K, and every norm, the router matrix and the
expert biases in F32. The load binds each row by its GGUF type: Q4_K and Q6_K
use the sub-16 decoders of the core dequantizer, and any other type is rejected
at load rather than reinterpreted. A Q4_K_M load occupies **45,254 MiB** of
device memory.

| File | Status |
| --- | --- |
| `Kolibri-1-Q4_K_M.gguf` | Measured ([Benchmarks](BENCHMARKS.md), [Quality](QUALITY.md)) |
| `Kolibri-1-Q6_K.gguf`, `Q8_0`, `Q5_K_M` | Not run; same binder, types to confirm per tensor |

## Get the weights and check them

```sh
hf download Hob-forge/Kolibri-1-GGUF Kolibri-1-Q4_K_M.gguf \
  --local-dir models/kolibri-1
```

The engine refuses a file whose tensor names, shapes or row types disagree with
`kolibri1`, so a wrong revision fails at load with the offending tensor named.

## Tools

The engine is currently driven by `tools/kolibri/`; `tools/kolibri/m1.sh` builds
inside the ROCm container on the M1 box and ships the artifact back.

```sh
# scalar oracle logits for a prompt (CPU, no ROCm needed)
tools/kolibri/build_reference_logits.sh build/gpu-test <model.gguf>
tools/kolibri/reference_logits <model.gguf> --prompt "..." --json

# device logits against that oracle, plus the chat-template golden
tools/kolibri/m1.sh model-test kolibri1_device_forward

# single-user prefill and decode, no sampling: throughput <model> [pp] [tg] [chunk]
tools/kolibri/m1.sh build
tools/kolibri/build_throughput.sh
tools/kolibri/throughput <model.gguf> 2048 64 32
```

## How it runs

- **Decode** packs rows to `max_rows` slots in one launch: a slot holds the
  query projections (the 6144-wide block spans 12 warps so each head's norm has
  one owner), then attention reads the cache, then the router and the grouped
  expert projection, shared expert last. Attention, the shared expert and the
  down projections overlap on their own streams, and the host fetch of the logits
  waits on the event that closes all of them.
- **The grouped projection** bins rows by expert and runs the experts in passes
  of eight slots (`kSlots`). `bin[8]{expert,rows,base}` lives in constant memory.
  With the 6-of-384 routing of this model, a pass almost never repeats an
  expert, which is why the small bin count does not re-read weights.
- **Prefill** re-reads the expert weights per chunk. The chunk width is a
  parameter (`--prefill-chunk`), and 32 measured fastest of the widths tried.
- **Cache:** the 10 patternless layers keep the whole context; the 40 sliding
  layers keep a ring of the window plus one chunk. A 32,768 context costs about
  1.45 GiB of cache.
