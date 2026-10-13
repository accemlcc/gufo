# Kolibri-1 benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (MINIX Elite ER939-AI),
`Hob-forge/Kolibri-1-GGUF` **Kolibri-1-Q4_K_M.gguf** (45,254 MiB resident).
Measured October 13, 2026 on the box that runs it, host `kubuntu`.

## Read this before using a number

Decode is reproducible here; prefill is not. Nine repetitions of the same
engine-direct cell gave decode a median absolute deviation of 0.0-0.2 tok/s,
while prefill in one fixed cell (same width, same context, same binary) spanned
**222.6 to 309.3 tok/s** — and a cell whose weights took 41.7 s to map prefilled
faster than one that took 12.2 s, so page-cache state is not a usable explanation
either. Every prefill row below therefore carries its repetition count and, where
there are repetitions, the MAD; rows marked **n=1** are single runs and support a
direction, not a ranking. `tools/kolibri/throughput` prints the session width it
used, and `~/kolibri-stats.py` computes the medians from the logs — a width
comparison without both is how the first round of this card ended up claiming a
+34 % effect that did not replicate.

**These are engine numbers unless the section says served.** They come from
`tools/kolibri/throughput`, which drives the engine directly: no HTTP, no sampler,
no tokenizer in the timed region, greedy is irrelevant because no token is
sampled.

## Repeated: the two widths that matter

Five repetitions at each width, run interleaved `16, 32, 16, 32, 16` so drift
falls on both, pooled with four earlier repetitions of the same cells (n=9).
Prefill and decode in tok/s, median ± MAD.

| Prompt | Session width | Chunk 16 | Chunk 32 |
| ---: | ---: | ---: | ---: |
| 512 | 640 | 253.2 ± 6.3 | **317.8 ± 22.0** |
| 512 | 640 (decode) | 59.6 ± 0.0 | 59.6 ± 0.0 |
| 8,192 | 131,072 | 245.4 ± 21.2 | 250.3 ± 5.6 |
| 8,192 | 131,072 (decode) | 55.5 ± 0.1 | 55.6 ± 0.2 |

At chat length 32 measures about 25 % ahead of 16 and the gap (64.6) is more than
double the summed MAD (28.3). At 8,192 the two are not separable. Taken on its
own — the cleanest set, since nothing but these cells ran during it — the
interleaved five also reversed the long-prompt case: 265.4 ± 1.3 for 16 against
245.3 ± 2.4 for 32, which is the shape of the argument for not promoting 32 on
strength of the short-prompt row.

**The served constant is 16**, the width the engine was qualified at, and the
first performance item below is the honest way to take the short-prompt win
without giving up the long-prompt one.

## Single runs: directional only

Each of these is one process, one pass. They are kept because they bound where
the interesting region is, not because they rank it.

| Prompt | Session width | Chunk | pp | tg |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 640 | 128 | 269.4 | 59.6 |
| 8,192 | 8,320 | 16 | 225.2 | 55.9 |
| 8,192 | 8,320 | 32 | 278.7 | 55.6 |
| 8,192 | 8,320 | 64 | 254.4 | 55.4 |
| 8,192 | 8,320 | 128 | 297.8 | 55.4 |
| 8,192 | 131,072 | 64 | 253.8 and 324.9 | 55.7 |
| 8,192 | 131,072 | 128 | 324.2 | 55.6 |
| 2,048 | 131,072 | 64 | 298.8 | 58.3 |

The 8,192 / 131,072 / chunk-64 row is the noise stated as a number: the same cell
measured 253.8 and 324.9 tok/s, 28 % apart, one session width and one binary
apart from nothing. Any width claim that is smaller than that is a claim about
the order the processes ran in.

## Depth

Engine-direct, chunk 64, session width 131,072, decode after the prompt is
cached — one run each:

| Depth (tokens) | pp | tg |
| ---: | ---: | ---: |
| 512 | 276.0 | 59.6 |
| 2,048 | 298.8 | 58.3 |
| 8,192 | 324.9 | 55.8 |

Decode falls about 7 % from a fresh context to 8K, which is the expected shape
for a cache that is ten layers deep at full context plus forty sliding rings.
Prefill does not fall with depth here, and it should if the patternless layers
dominate: at 8,192 they attend the whole past quadratically. One run each cannot
separate that from the ±15 % above, so the flat curve is not a finding — the
served depth measurement below is the one that shows depth costing.

## Served path

`gufo serve llm --context 131072 --sessions 1`, forward width **16**, one
session. The HTTP API returns no timing fields; everything here is the server's
own `event=completed` line (`prompt_tokens prefill_tokens cached_tokens
prefill_tps decode_tps ttft_ms cache_restore_ms queue_ms`).

| Depth of prompt | pp (tok/s) | tg (tok/s) | ttft |
| ---: | ---: | ---: | ---: |
| 71 | 240.0 | 59.7 | 123.3 ms |
| 2,141 | 230.4 | 63.0 | 9.11 s |
| 39,292 cold | 158.8 | 51.1 | 247.5 s |
| 39,292 cold, second context | 157.7 | 50.8 | 248.9 s |
| 39,292, same context repeated | — | 51.9 | **180.7 ms** |
| 108,627 cold, after a 9,940-token calibration leg | 97.9 | 31.8 | **1,109.7 s** |
| 109,656, needle-free 9,845-token calibration before it | 93.4 | 31.7 | 1,086.4 s, 8,192 reused |

Depth costs: prefill falls from ~230 to ~159 tok/s between 2K and 39K, and decode
from ~60 to ~51 — the quadratic patternless layers, visible only where depth was
actually measured twice. Reuse, at the same depth: 42 tokens (the conversation
frame) reused and 39,250 prefilled for a same-length context with different
bytes; the whole 39,292 reused for the conversation that had been seen, in 180.7
ms against 247.5 s ([Quality](QUALITY.md)).

What depth actually costs is cache bytes rather than attention arithmetic, and
the three decode numbers put that in one regime. A generated token reads the ten
patternless layers' KV for every position in the context: 10 layers × 4 KV heads
× 128 dimensions, K and V, binary16 — 20,480 bytes per position, so 2.07 GiB at
108,627 positions against 0.76 GiB at 39,292, while the forty sliding rings
contribute their ~42 MB whatever the depth. The 3.46B active parameters add
roughly 2 GB of weights. Divide the sum by the measured rate — 59.6 tok/s at 512,
51.1 at 39,292, 31.8 at 108,627 — and it comes out at 131, 151 and 137 GB/s: one
bandwidth-bound regime, within the ±15 % any single measurement on this box
carries. Decode falls with depth because the cache has outgrown the weights, and
past ~60 K positions it is the larger half of every token.

Depth past 39K was predicted before it was measured, and the prediction is what
the 108,627 row above tests. `T(n) = a*n + b*n^2` — a linear term for the
bandwidth-bound pass, a quadratic one for the ten patternless layers attending
the whole past — fitted to cold points in two variants:

* two points, both served at forward width 16 and session width 131,072: 2,141
  tokens at 9.111 s and 39,292 at 247.7 s, giving `a=4.14e-3 s/token` and
  `b=5.51e-8 s/token^2`;
* three points, adding 8,192 tokens at 33.97 s measured **engine-direct**
  (`tools/kolibri/throughput`, chunk 16, width 131,072) — a different regime, no
  HTTP and no scheduler, so the two sources are not interchangeable, giving
  `a=3.63e-3` and `b=6.81e-8`.

They predict 110,000 tokens at 1,122 s to 1,223 s, 90 to 98 tok/s. At the depth
actually sent — 108,627 — they say 1,100 s and 1,197 s. Measured: 1,120.7 s at
97.9 tok/s; the two-point fit was 2 % low, the three-point fit 6 % high.
This is the only point beyond 39K, so every other point on that curve is
extrapolation, and one consequence it implies is arithmetic rather than
observation: the quadratic share grows from 34-42 % of the time at 39K to
59-67 % at 110K. Decode at that depth is observed, not fitted: 31.8 tok/s
against 59.6 at 512.

The graded leg carries a condition worth keeping with its number. `serve_check
recall` first sends a calibration prompt of its own — 200 filler lines asked with
`max_tokens=1` (9,940 tokens, 47.2 s at 210.8 tok/s) — and that prompt plants its
own needles at 10/50/90 % of *its* 200 lines. So the two texts part company at
line 20: calibration breaks into a needle there, the graded prompt keeps counting
filler until line 221. The log states exactly that distance —
`common_prefix_tokens=934`, which is nineteen lines at the probe's own 49.7
tokens per line — while the only checkpoint the calibration turn left sat at
9,940 (`nearest_checkpoint_tokens=9940`). A checkpoint longer than the shared
prefix cannot be applied to a new prompt, so
`cache=miss cached_tokens=0 prefill_tokens=108627` is what the geometry requires
here, not evidence about whether checkpoints survive 8.2 GB of snapshot. The
probe now calibrates without needles, which makes the calibration a real prefix of
the graded prompt, and the leg run on October 13 at 10:04 measured the answer:
**`cached_tokens=8192`, `prefill_tokens=101464` of a 109,656-token prompt** — the
checkpoint survived 108K of state and was applied. It landed on 8,192 rather than
the 9,845 the calibration actually shared, which is the 2,048-token checkpoint
grid (`text_model_runner.cpp:753`) snapped down to a window-safe boundary by
`PrefillStop`. Same mechanism, one row above, explains cache-depth's
`cached=20008`: 20,480 on the grid, snapped down 472. Decode reproduced to one
tenth — 31.7 tok/s here against 31.8 in the cold leg — while prefill moved 97.9
to 93.4, exactly the split between the two kinds of measurement this card keeps
insisting on.

Loading the model over HTTP took 18.5 s for the 45 GiB file, against 12-42 s
engine-direct depending on page-cache state.

## Against the reference implementation

Same box, same artifact. The reference is the patched llama.cpp the conversion
ships, configured `-DGGML_HIP=ON` for `gfx1151` against this ROCm, run as
`llama-bench -ngl 99 -p 512,2048 -n 64 -r 1`. **n=1 on both sides** — llama.cpp
was not repeated, and the Gufo prefill column is the n=9 median at width 640.

| Test | Gufo (tok/s) | llama.cpp ROCm (tok/s) | Ratio |
| --- | ---: | ---: | ---: |
| pp512 | 253.2 (n=9, chunk 16) | 1217.2 (n=1) | **−79 %** |
| pp2048 | 298.8 (n=1, chunk 64) | 1232.8 (n=1) | −76 % |
| tg64 at 512 | 59.6 (n=9) | 64.8 (n=1) | −8 % |

Decode is bandwidth-bound on both sides: 3.46B active parameters spread over 6 of
384 experts per layer, the shared expert, the attention projections and one
untied 128,000-wide output row come to about 2 GB of weights per token, which
puts Gufo near 118 GB/s and llama.cpp near 130 GB/s — one regime, 10 % apart.
Prefill is not bandwidth-bound: at 253 tok/s this model would need some 500 GB/s,
which the hardware does not have, so it is compute-bound, and 253 tok/s of a
3.46B-active model is about 1.7 TFLOP/s against llama.cpp's 8.5. The reference
runs prefill through hipBLASLt's tensor-core GEMM; this engine uses its own
projection kernel and re-reads the expert table once per forward. That is the
gap, and it is the model's first performance item.

## Not measured

| Cell | Status |
| --- | --- |
| Concurrency (2/4/8 users) | TODO — one session per forward today; `SupportedPlans` is serial width 1 |
| Q6_K / Q8_0 / Q5_K_M | TODO |
| Contexts beyond 39,292 | Measured once, at 108,627 ([Served path](#served-path)); nothing between 39K and it |
| Continuation cache, prefix reuse | Measured ([Served path](#served-path), [Quality](QUALITY.md)) |
| Tool-call round-trip | Measured ([Quality](QUALITY.md)) |
| Functional suites (13) | Recorded, 6 pass / 7 fail, every failure with a cause ([Quality](QUALITY.md#functional-qualification)) |

## Performance items

1. **Forward width per prompt length.** 32 is 25 % ahead at chat length, 16 holds
   the long-prompt case; one constant has to lose one of them. Nothing structural
   blocks the split: the sliding ring is `next_pow2(513 + rows)` = 1,024 slots for
   any width up to 511, so both widths reserve the same memory and only the
   executor's scratch changes.
2. **Prefill GEMM.** The reference's advantage is a tensor-core path, not a
   better schedule.
3. **Repeated-cell measurement protocol.** `~/kolibri-stats.py` exists; the
   engine-direct harness should run n repetitions itself rather than leave it to
   the caller's shell loop.
4. **The patternless layers' KV is now the bandwidth bill.** 2.07 GiB per
   generated token at 108,627 positions, against ~2 GB of weights — the reason
   decode falls 59.6 → 51.1 → 31.8 tok/s with depth. Binary8 KV, or sharing heads
   across the ten full-attention layers, is the only lever that keeps decode flat
   as context grows; widening the sliding window is not, since those layers are
   precisely the ones that cannot slide.
