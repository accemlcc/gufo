# Kolibri-1

Aleph Alpha's Kolibri-1: a 78-billion-parameter mixture-of-experts reasoning
model for German and English with 3.46 billion active parameters per token,
Apache-2.0. This card covers Hob-forge's independent GGUF conversion,
[`Hob-forge/Kolibri-1-GGUF`](https://huggingface.co/Hob-forge/Kolibri-1-GGUF),
whose GGUF architecture key is `kolibri1`. Stock llama.cpp does not run this
architecture; the conversion ships a patch for it. Gufo implements it natively.

**Status.** The engine, its device check against a scalar oracle, the tokenizer
and the chat renderer pass on `gfx1151`; `gufo chat`, `gufo prompt` and
`gufo serve llm` dispatch to this architecture, and the served path is verified
for chat, reasoning separation, tool calls, prefix reuse and session isolation
([Quality](QUALITY.md)). The numbers in [BENCHMARKS.md](BENCHMARKS.md) are
engine-direct except the "Served path" section. No speculative decoding (the
architecture has no draft head here) and no wide-batch prefill.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Architecture

- **Layers:** 50, hidden width 2560, 48 query heads, 4 KV heads, head dim 128.
  Q projection is 2560 → 6144, K/V are 2560 → 512.
- **Attention pattern:** four sliding layers, then one with no positional
  encoding (NoPE) — layers 4, 9, …, 49, read from
  `attention.sliding_window_pattern` (1 = sliding, 0 = patternless). Sliding
  layers attend over `attention.sliding_window` = **513 keys**: the query plus
  the 512 preceding tokens, which is the window the tech report sizes, and
  rotate Q/K by RoPE base 10,000 (`rope.freq_base`); the patternless layers see
  the whole past and are not rotated. RMSNorm eps 1e-6.
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
- **Vocabulary:** 128,000 **UniBPE** — merges built bottom-up like BPE, each
  one chosen by Unigram loss (report §2.1.2), so it ships as GPT-2-style BPE:
  `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = kolibri1` (the loader
  rejects any other `pre`, `tokenizer.cpp:317-320`), 127,644 ranked merges, no
  added-token array, `eos = 127906`, `pad = 127901`, `add_bos = add_eos = 0`.
  Unknown text is carried by every byte having a symbol through GPT-2's
  `bytes_to_unicode` map (`tokenizer.cpp:62-66`) — there is no SentencePiece
  model in the file. The released artifact ships an untied `output.weight`; a
  tied file binds the same view. Max context 262,144 (trained to 256k).

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
./result/bin/gufo serve llm --model models/kolibri-1/Kolibri-1-Q4_K_M.gguf \
  --served-model-name Kolibri-1 --context 131072 --sessions 1 --port 8102
```

`--system`, the shared sampling flags (`-n` / `--max-tokens` and friends) and
`--verbose` apply. `chat` and `prompt` fix the context at 8,192
(`kKolibriContext` in `src/cli/prompt/kolibri1_prompt.cpp`); `serve` takes
`--context` and holds it for the process, since the session cache is sized from
it at load. Sampling falls back to the upstream `generation_config` through
`sampling::TextModelPreset::kKolibri1` (temperature 1.0, top-p 0.97, top_k 128)
— the GGUF ships no `general.sampling.*`, so requests that send nothing get
those numbers.

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
- **Prefill** re-reads the expert weights per forward, so the forward's row
  count (`ModelOptions::prefill_chunk`) is a real knob — and it is 256 now, in
  three takings: 16 → 128 doubled served prefill (2× at 1,572 and at 13,030
  prompt tokens, decode unmoved), because since `17eada37` a wide block sends
  the seven dense projections to the matrix cores instead of stepping them
  through the batch-invariant GEMV; 128 with one group table over the forward
  took the routed stage to 1.96× of itself; and 128 → 256 added **12.9 % to
  18.4 %** served across 3.2 K to 97.9 K with decode and memory unmoved. That is
  where the bytes stop: the routed stage's bytes per token halve from 128 to
  256 and its time fell 11 % then, because `MoeRoute` re-read a row of the
  layer input once per expert and so was flat in the width. That was a shape
  and not a law — tiled, eight rows against eight experts per block
  (`kRouterTiledMinRows`), the same launch is 3.25× cheaper at 256 rows, the
  stage gives up 17 % more of itself and prefill **13.6 %** more
  engine-direct at 4 K — **10.8 %** of it served at 13 K — with decode unmoved
  and every logit bit-identical
  ([Benchmarks](BENCHMARKS.md#the-router-reads-each-operand-once)). `gufo chat`, `gufo
  prompt` and `gufo serve` pass this constant. The scheduler's `--prefill-chunk`
  is a different number — the budget of prompt tokens between decode rounds —
  and reaches the model as `max_input_tokens`.
- **Cache:** the 10 patternless layers keep the whole context; the 40 sliding
  layers keep a ring of the window plus one forward's rows, rounded up to a
  power of two — `next_pow2(513 + rows)`, so 1,024 slots for any forward up to
  511 rows. KV is binary16, so a session at 32,768 costs about 720 MiB
  (640 MiB of patternless layers, 80 MiB of rings) and the full 262,144 about
  5.1 GiB.

Measured on this box: `gufo serve llm --context 131072 --sessions 1` reports
**51,164 MiB** of device memory on its own `event=load_completed` line — 45,254
MiB of weights, the rest patternless-layer cache at 131,072 positions, rings and
launch scratch; the same line reads 47,324 MiB at `--context 32768`.
That is the number a co-resident engine has to leave room for.

`scripts/kolibri-gufo.sh start|stop|status|logs` drives that box: it refuses to
start below 16 GiB of available memory **and** while more than 8 GiB of GTT is
already resident, naming the holder (`amd-smi process`) instead of stopping a
container on a guess, and it waits for `/health` before reporting ready.

## Sources

- **Architecture of record:** Aleph Alpha, *Kolibri: A Sovereign European
  Model on the Pareto Frontier* (tech report, 3 October 2026), §2.1 and
  Table 2: 78.1 B total parameters, 3.46 B active per token excluding
  embeddings, 50 blocks with no leading dense layer, 10 patternless against
  40 sliding, window of 512 preceding tokens, RoPE θ = 10,000 on the sliding
  blocks only, per-head QK RMSNorm before the rotation, 6 routed experts of
  384 plus one shared per layer, weights not renormalised after selection.
- **Official inference path:**
  [`Aleph-Alpha/aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference)
  — a vLLM plugin (`aleph_alpha_inference/kolibri1.py`, `reasoning.py`, and the
  chat template its tests render). This build does not run it; it is the
  reference the engine's semantics were checked against, alongside the patched
  llama.cpp in `tools/kolibri/run_parity.sh`.
- **What actually ships** is the artifact's own metadata, readable with
  `tools/kolibri/gguf_meta.py`: `general.architecture = kolibri1`, 903 tensors,
  `attention.head_count = 48` / `head_count_kv = 4`, `key_length = 128`,
  `layer_norm_rms_epsilon = 1e-6`, `attention.sliding_window = 513`,
  `sliding_window_pattern` four 1s then a 0 across the 50 layers,
  `rope.freq_base = 10000.0`, and `tokenizer.chat_template` carrying the same
  template text the official plugin tests — which is what `chat_template.cpp`
  reproduces byte for byte.
