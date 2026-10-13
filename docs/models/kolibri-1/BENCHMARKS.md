# Kolibri-1 benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (MINIX Elite ER939-AI),
`Hob-forge/Kolibri-1-GGUF` **Kolibri-1-Q4_K_M.gguf** (45,254 MiB resident).
Measured October 13, 2026.

**These are engine numbers, not server numbers.** They come from
`tools/kolibri/throughput`, which drives the engine directly: no HTTP, no
sampler, no tokenizer in the timed region, greedy is irrelevant because no token
is sampled — the loop reads logits and picks the argmax itself. Single user,
`--prefill-chunk` as stated. Treat them as an upper bound on what the served
path will do once it exists. **TODO** means unmeasured.

## Prefill chunk

Prompt length against the chunk width the engine feeds the grouped projection at
once. The expert weights are re-read per chunk, so the width is a real knob;
512 tokens:

| Chunk | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 16 | 251.0 | 59.0 |
| 32 | **321.9** | 59.0 |
| 64 | 256.8 | 57.5 |

2,048 tokens, the two extremes again:

| Chunk | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 16 | 251.9 | 57.9 |
| 64 | 260.2 | 57.6 |

Decode does not depend on the chunk (one row per forward), which the tables
bear out: it moves with context depth, not with chunk width.

## Depth

Prefill and decode at the width that measured fastest above, decode measured
after the prompt is cached — depth is the cached prefix:

| Depth (tokens) | Chunk | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: | ---: |
| 512 | 32 | 321.9 | 59.0 |
| 2,048 | 64 | 260.2 | 57.6 |
| 8,192 | 64 | 254.4 | 54.9 |

Decode loses about 7% from a fresh context to 8K, which is the expected shape
for a model whose cache is only 10 layers deep at full context plus 40 sliding
rings. Prefill at 8,192 tokens is 254 tok/s at chunk 64 — the patternless layers
are quadratic and the sliding ones are not, and the 4:1 mix keeps the whole
curve nearly flat.

## Served path

`gufo serve llm`, one session, context 8,192, October 13, 2026. The HTTP API
returns no timing fields; these come from the server's own `event=completed`
line, which carries `prompt_tokens prefill_tokens cached_tokens prefill_tps
decode_tps ttft_ms cache_restore_ms queue_ms`. The methods agree: a 2,046-token
prompt measured 9.367 s from the client against `prefill_tps 218.6` in the log,
and a 64-token generation 59.2 tok/s against `decode_tps 60.5`.

| Metric | Value |
| --- | ---: |
| prefill_tps (log, several requests) | 217.7 – 244.4 |
| decode_tps (log) | 59.4 – 60.5 |
| ttft_ms on a warm prefix | 1.3 |
| cache_restore_ms | 1.1 – 3.5 |

Served prefill lands on the chunk-16 row of the first table, which is what the
geometry predicts: `--prefill-chunk` is the scheduler's budget of prompt tokens
between decode rounds — it arrives here as `max_input_tokens` — while
`kKolibriPrefillChunk` is the number of rows one forward actually carries. A
512-token step runs as 32 forwards of 16 rows. Widening the forward is the open
performance item the next section explains.

Loading over HTTP took 18.5 s for the 45 GiB file, against 12-35 s
engine-direct depending on page-cache state.

## Against the reference implementation

Same box, same artifact, October 13, 2026. The reference is the patched
llama.cpp the conversion ships, configured `-DGGML_HIP=ON` for `gfx1151`
against this ROCm, and run as `llama-bench -ngl 99 -p 512,2048 -n 64 -r 1`
with flash attention left at its default. The Gufo columns are the
engine-direct numbers above, best chunk.

| Test | Gufo (tok/s) | llama.cpp ROCm (tok/s) | Gain |
| --- | ---: | ---: | ---: |
| pp512 | 321.9 | 1217.2 | **−74%** |
| pp2048 | 260.2 | 1232.8 | **−79%** |
| tg64 | 59.0 | 64.8 | **−9%** |

Gufo loses on this architecture today, badly at prefill and slightly at decode.
Both sides ran on the device: `USED_GTT` was 51.6 GB during the llama-bench run
against a 47.5 GB file, and on this APU the processor and the matrix cores draw
from one LPDDR5X pool, so neither column is reading a faster chip.

The two tests fail for different reasons. Decode is bandwidth-bound on both:
3.46B active parameters spread over 6 of 384 experts per layer, the shared
expert, the attention projections and one untied 128,000-wide output row add up
to about 2 GB of weights per token, which puts Gufo at roughly 118 GB/s and
llama.cpp at 130 GB/s — the same regime, 10% apart. Prefill is not
bandwidth-bound: at 251 tok/s the engine would have to stream some 500 GB/s,
which the hardware does not have, so it is compute-bound instead, and 251 tok/s
of a 3.46B-active model is about 1.7 TFLOP/s against llama.cpp's 8.5. The
reference runs prefill through hipBLASLt's tensor-core GEMM; this engine uses
its own projection kernel and additionally re-reads the expert weights once per
prefill chunk, which is why a wider chunk moves prefill only from 251 to 322.
Closing that gap is the model's first performance item.

## Not measured

| Cell | Status |
| --- | --- |
| Concurrency (2/4/8 users) | TODO — one session per forward today; `SupportedPlans` is serial width 1 |
| Q6_K / Q8_0 / Q5_K_M | TODO |
| Contexts beyond 8,192 | In progress — 128 K recall probe running |
| Continuation cache, prefix reuse | Measured ([Served path](#served-path), [Quality](QUALITY.md)) |
| Tool-call round-trip | Measured ([Quality](QUALITY.md)) |
