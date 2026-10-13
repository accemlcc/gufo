# Kolibri-1

Aleph Alpha's Kolibri-1: a 78-billion-parameter mixture-of-experts reasoning
model for German and English with 3.46 billion active parameters per token,
Apache-2.0. This card covers Hob-forge's independent GGUF conversion,
[`Hob-forge/Kolibri-1-GGUF`](https://huggingface.co/Hob-forge/Kolibri-1-GGUF),
whose GGUF architecture key is `kolibri1`. Stock llama.cpp does not run this
architecture; the conversion ships a patch for it. Gufo implements it natively.

**Status.** The engine, its device check against a scalar oracle, the tokenizer,
the chat renderer and the throughput tool pass on `gfx1151`, and `gufo chat` /
`gufo prompt` dispatch to this architecture. The HTTP path
(`gufo serve llm`, and with it the continuation cache and concurrency) is not
wired yet; the numbers in [BENCHMARKS.md](BENCHMARKS.md) therefore come from the
engine, not from HTTP.

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

## Run it

```sh
nix build
./result/bin/gufo chat --model models/kolibri-1/Kolibri-1-Q4_K_M.gguf \
  --max-tokens 512
./result/bin/gufo prompt --model models/kolibri-1/Kolibri-1-Q4_K_M.gguf \
  --raw -p "Hauptstadt von Deutschland?"
```

`--system`, the shared sampling flags (`-n` / `--max-tokens` and friends) and
`--verbose` apply. The context is fixed at 8,192 by `kKolibriContext` in
`src/cli/prompt/kolibri1_prompt.cpp` and is not a flag yet, and reasoning effort
is not one either, so `chat` renders the template's default (`high`).

## Developer tools

`tools/kolibri/m1.sh` builds inside the ROCm container on the M1 box and ships
the artifact back; everything below runs there.

```sh
# scalar oracle logits (CPU, no ROCm needed). The prompts file holds token ids,
# one prompt per line; with no file they come from the command line.
tools/kolibri/m1.sh build gufo_kolibri1_reference
tools/kolibri/build_reference_logits.sh
artifacts/kolibri/reference_logits <model.gguf> out.json prompts.txt

# device logits against that oracle, plus the chunk-invariance check
tools/kolibri/m1.sh model-test kolibri1_device_forward

# oracle vs the patched llama.cpp on synthetic artifacts: window clipping, the
# patternless layer at first, middle and last position, one- and many-token prompts
LLAMA_DIR=~/kolibri-xcheck/llama.cpp tools/kolibri/run_parity.sh

# single-user prefill and decode, no sampling
tools/kolibri/build_throughput.sh
artifacts/kolibri/throughput <model.gguf> [pp] [tg] [chunk] [max_context]
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
- **Prefill** re-reads the expert weights per chunk, so the chunk width
  (`ModelOptions::prefill_chunk`) is a real knob; of the widths tried at 512
  tokens, 32 measured fastest ([Benchmarks](BENCHMARKS.md)). `gufo chat` and
  `gufo prompt` pass 16 today, which is why their prefill sits at ~250 tok/s.
- **Cache:** the 10 patternless layers keep the whole context; the 40 sliding
  layers keep a ring of the window plus one forward's rows, rounded up to a
  power of two — 1,024 slots at a chunk of 16. KV is binary16, so a session at
  32,768 costs about 720 MiB (640 MiB of patternless layers, 80 MiB of rings)
  and the full 262,144 about 5.1 GiB.
