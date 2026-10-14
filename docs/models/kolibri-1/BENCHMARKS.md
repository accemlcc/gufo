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

Those cells are the old dense path and one run each. Since `17eada37` a block of
32 rows or more goes to the matrix cores, and both dense paths were then measured
against each other inside a single window, three runs per cell
([Prefill width, both dense paths](#prefill-width-both-dense-paths)) — which also
confirms the 28 % above as the shape of the GEMV's variance, not as luck.

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

`gufo serve llm --context 131072 --sessions 1`, one session. Everything in this
table was measured at forward width **16**, the width the runner passed until
2026-10-13; the shipped constant is 128 now and the width's own served
comparison lives in [Served forward width](#served-forward-width). That older run
read the server's `event=completed` line (`prompt_tokens prefill_tokens
cached_tokens prefill_tps decode_tps ttft_ms cache_restore_ms queue_ms`); the
build measured against it answers with a `timings` object per request. For the
shipped width at depth, use
[Served depth, cold, at 128 rows](#served-depth-cold-at-128-rows).

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
the 9,845 the calibration actually shared, because that calibration is another
conversation and nothing was checkpointed where the two prompts part: the only
candidates are the 2,048-token checkpoint grid (`text_model_runner.cpp:753`) and
8,192 is the largest of its points below the shared prefix. cache-depth's
`cached=20008` is a different rule and a finer one — the stable prefix, which
ends before a trailing text-only user turn (`chat_template.hpp:47-51`) — and
20,606 − 20,008 = 598 is exactly that user block; the same probe measured 594 and
592 at shallower depths. Decode reproduced to one tenth — 31.7 tok/s here against
31.8 in the cold leg — while prefill moved 97.9 to 93.4, exactly the split
between the two kinds of measurement this card keeps insisting on.

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
Prefill changes regime with the forward width, and the shipped width is the
memory-bound one. Every matmul in this engine — q/k/v/o, the shared expert, all
384 routed experts, the output head — is one kernel, `GroupedGemvKernel`
(`kernels/rocm/gemv.hip.cpp:299`), a scalar dequant-then-FFMA dot product; the
engine's only WMMA is attention (`attention.hip.cpp:58`) and it calls no
hipBLASLt. A forward processes `kProjectRows = 16` rows at a time (`executor.hpp:23`,
stepping loops `executor.cpp:223` and `executor.cpp:259`), so every step re-reads
the ~30 MB of dense projection weights for its layer plus ~2.4 MB for each expert
its 96 slots actually chose — a group with no slot returns immediately
(`gemv.hip.cpp:309-311`), so traffic follows the *touched* experts per step, not
the table.

At width 16 that is 0.5-0.8 GB of weights per token, which at the measured ~250
tok/s is 150-200 GB/s: at the shipped width prefill sits on the memory ceiling,
not on arithmetic. From width 64 up the re-read amortises and the curve flattens,
but the ceiling behind that flattening belongs to the **routed experts**, not to
the dense projections — a 512-wide expert is 2.16 MB across gate, up and down, six
slots per token, so ~13 MB per token per layer and ~650 MB per token. That is now
measured rather than argued: lifting the seven dense projections off the GEMV
(item 2a) moved pp2048 from 257-331 tok/s to 359-367 while the experts stayed put,
and it collapsed the run-to-run spread from 29 % to 2.4 %
([Prefill width, both dense paths](#prefill-width-both-dense-paths)). What this
paragraph said before — that width alone cannot pass 325 — described the curve
correctly and the cause wrongly: 325 was a plateau that still contained the dense
GEMV.

That verdict is worth exactly its own range. Both engines ran at 512 and 2,048
prompt tokens; nothing on the reference side was ever run past 2,048, so −76 %
is a chat-length statement and not an architectural one.

Depth reverses the sign. This engine's served curve is 240 tok/s at 71 tokens,
158.8 at 39,292 and 97.9 at 108,627 — a factor of about 2.5 across three orders
of magnitude of prompt, which is a slope. The operator's standing observation
from the llama.cpp era of this model on this box is that its prefill falls to
roughly 20 tok/s near 100,000 tokens, which is a collapse. **That 20 is
remembered, not re-measured here**, and it is the reason the crossing point is
listed as unmeasured rather than asserted: if it holds, the curves cross
somewhere between 2,048 and 100,000 and the engine is several times ahead exactly
where long context is used. The cell that settles it is `llama-bench -ngl 99 -p
8192,32768,99000 -n 0 -r 1` on the patched build, on this box.

## Prefill width, both dense paths

Engine-direct `tools/kolibri/throughput`, same box, same artifact, one device
window, so no machine state separates the two builds: pp 2048, decode 32, context
2,144, three runs per cell. **GEMV** is `17eada37^`, **tiled** is `17eada37`, whose
dense path takes over at 32 rows (`kMatrixCoreRows`).

| Build | chunk | pp2048 tok/s, three runs | median |
| --- | ---: | --- | ---: |
| GEMV | 16 | 289.5 · 243.8 · 303.2 | 289.5 |
| GEMV | 128 | 331.3 · 265.3 · 257.2 | 265.3 |
| tiled | 16 | 262.0 · 252.9 · 251.4 | 252.9 |
| tiled | 128 | 366.9 · 367.3 · 358.7 | 367.3 |

Three readings, in order of how much they should be trusted. At width 128 the
tiled build is **1.11× the best** run of the GEMV build and **1.38× its median** —
that range is the claim, not the median. At width 16 the patch is inactive by
construction, and still the two medians differ by 13 % (289.5 against 252.9): that
is the drift between phases of one window on this box, and it is why the sentence
before gives a range. The spread is not noise to average away — 2,800 short GEMV
launches per 2,048-token prompt competing for DRAM with the desktop is a different
machine from 700 regular ones with a third of the traffic, and it shows as 29 %
against 2.4 %. Decode did not move: 58.2-58.5 tok/s on both builds, every cell.

The isolated-shape probe (`tools/kolibri/prefill_gemm_probe.cpp`) predicted
0.93 ms/token saved at width 128; the measured saving is 0.30 ms/token (3.02 →
2.72 ms per token). Its per-shape ratios hold, its absolute dense cost is about
three times too high — inside a real layer the dense re-reads are largely served
from cache, so the engine's dense share is ~0.6 ms/token, not 1.3. Read that probe
as a ratio, never as a cost model.

## Served forward width

The ladder above measures a forward; what ships is a request. Both builds —
`kKolibriPrefillChunk` 16 and 128, md5-verified apart — served the same artifact
inside one device window on m1: `--context 131072 --sessions 1`, no disk cache,
two runs per cell plus one generating leg.

| Prompt | width 16, two runs | width 128, two runs | ratio |
| --- | --- | --- | ---: |
| 1,572 tokens | 290.7 · 293.5 | 531.3 · 545.8 | **1.83×** |
| 13,030 tokens | 252.3 · 251.0 | 505.7 · 503.0 | **2.00×** |
| 13,030, ttft | 51.6 s | 25.8 s | **half** |
| 13,030 + 64 generated, decode | 53.27 | 53.15 | — |

Tokens per second are the server's own `timings.prompt_per_second`.
`cache_snapshot_ms` sits where it was on both widths (40-45 ms at 13 K), so the
wider forward did not move work onto the snapshot side, and memory is identical
because the ring is `next_pow2(513 + rows)`: 128 rows still lands on 1,024 slots.
Decode not moving is the check that matters most here — a wider forward that
stole decode would show up in that last row.

What it costs is one field of the identity. `prefill_chunk_rows=` keys the
disk-cache tree, so every Kolibri tree stored under 16 re-prefills once after the
change; the weights are untouched, the cost is per tree and one-time, and this
model serves no production traffic today.

Numerics at the new width are the same gate as at the old.
`kolibri1_device_forward_test` against the scalar reference, run at
`KOLIBRI_TEST_PREFILL_CHUNK` 32 and 128 as well as 16: prefill max_abs 3.3e-3,
mean 4.6e-4, argmax equal, and the chunked case that crosses the width boundary
exact at 0 — the same numbers the old width produced.

**Served absolutes of this section read high against a later window.** The same
artifact, the same width-128 build and a 13 K prompt measured 336.9 tok/s on
`--context 16384` where this section recorded 505.7 on `--context 131072`; the
1.83×/2.00× width ratios are ratios inside one window and stand, the absolute
tok/s do not carry across the two configurations. See
([The same change on the model, device to itself](#the-same-change-on-the-model-device-to-itself)).


## Served depth, cold, at 128 rows

`tools/kolibri/serve_check.py sweep` walks one cold prompt per depth against a
live server: m1, 2026-10-14, `--context 131072 --sessions 1`, forward width 128
(the shipped constant), 64 generated tokens per leg, artifact
`Kolibri-1-Q4_K_M.gguf` at 47,454,113,472 bytes, which is the byte size the
Hob-forge repository reports for that file. Both rates are the server's own
`timings`, and the depth is what the server counted, not what was estimated
from the text.

Every prompt begins with a fresh random marker, because prefix reuse here is
byte-exact from the first token. The `cached` column is the coldness proof: 42
tokens in every leg, which is the conversation frame, identical bytes in every
request, and nothing of the filler.

| Prompt | pp (tok/s) | tg (tok/s) | prefill | cache hit |
| ---: | ---: | ---: | ---: | ---: |
| 3,203 | 571.8 | 59.2 | 5.5 s | 42 |
| 4,135 | 545.3 | 58.3 | 7.5 s | 42 |
| 16,336 | 480.2 | 53.1 | 33.9 s | 42 |
| 32,654 | 420.1 | 47.5 | 77.6 s | 42 |
| 65,240 | 335.9 | 39.5 | 194.1 s | 42 |
| 97,873 | 279.3 | 33.9 | 350.4 s | 42 |

Five of those legs fit `T(n) = a*n + b*n^2` with almost nothing left over:
`a=1.781e-3 s/token`, `b=1.835e-8 s/token^2`, and the largest error on any one
leg is 0.36 s, on the shortest leg of 5.5 s. The sixth leg was measured after
the fit, kept out of it, and the fit said 350.1 s against the 350.4 s it took:
0.10 %. The quadratic share of a prompt's time — the ten patternless layers
attending the whole past — is 3.4 %
at 3,203, 25.2 % at 32,654 and 50.2 % at 97,873: past about 87 K positions,
prefill is mostly attention rather than weights.

That is worth putting against the older numbers on this page, because the
comparison says two things the single deep legs of the [Served
path](#served-path) section could not.

* **Width, on the same artifact.** The width-16 run read 158.8 tok/s at 39,292
  and 97.9 at 108,627; this curve interpolates to 399.7 and 264.9 there, so
  **2.52× and 2.71×**. [Served forward width](#served-forward-width) measured
  **1.83× and 2.00×** at 1,572 and 13,030: the wider forward keeps, and a
  little grows, its gain at depths where the work is attention rather than the
  routed experts.
* **Fit quality.** The old coefficients came from two points, and read
  `a=4.14e-3`, `b=5.51e-8`. Forward width cannot change how many token pairs
  the patternless layers attend, so `b` three times larger is not a kernel
  difference: two points pin a curve by their endpoints alone, and every depth
  beyond 39 K in the older section is that fit talking. `b` here is constrained
  by a 30× span of depths with a 0.4 s residual.

Decode follows the same one-regime bandwidth model as before, which is also the
check that the wider forward did not move work onto the decode side: weights
~1.95 GB plus 20,480 bytes of patternless cache per position, over the measured
rate, gives 119 GB/s at 3,203, 124 at 32,654 and 134 at 97,873 (1.87 GiB of
that cache at the deepest leg). Device state through the run: 84 GiB of GTT
resident, 51,188 MiB of it from the load and the rest the snapshot cache's
configured 32 GiB, flat with depth.

## Forward width 256, served

`kKolibriPrefillChunk` and `kMaxGroupSlots` were 128; both are 256 now. It is
the cheapest experiment the engine has — two constants, no kernel — and the byte
model said what it should be worth: a forward reads the distinct experts of its
own rows, so 128 rows touch 384 × (1 − (1 − 6/384)<sup>128</sup>) ≈ 333 of them
per layer and 256 rows touch ≈ 377. Per token that is 287 MB of routed expert
weights at 128 rows against 163 MB at 256 (2.21 MB per expert, the same figure
the decode column of [Where the forward's time
sits](#where-the-forwards-time-sits) is built from), and the dense projections
and the shared expert halve by construction. Predicted, with the routed stage at
64 % of a 128-row forward: prefill near 1.7×.

Measured: **13 % to 18 %**. One window per build on m1, device to itself,
`--context 131072 --sessions 1`, no disk cache, `serve_check.py sweep` with a
fresh marker per leg (49.25 tokens per line on both legs), 64 generated tokens,
the same artifact. Both binaries are branch `HEAD` out of one tree, constants
apart (`md5 4a99c68c` at 128, `088e3969` at 256). The control leg ran *after*
the candidate leg, and windows on this box differ by about 2 % — the width-128
curve above read 545.3 tok/s at 4,135 where this one's control read 539.9 — so
the gains below are six to nine times the drift they sit in.

| Prompt | width 128 pp | width 256 pp | gain | tg 128 | tg 256 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,205 | 542.6 | 639.1 | **+17.8 %** | 58.1 | 58.8 |
| 4,135 | 539.9 | 639.4 | **+18.4 %** | 58.3 | 58.3 |
| 16,338 | 478.1 | 560.5 | **+17.2 %** | 53.2 | 53.2 |
| 32,653 | 417.1 | 484.8 | **+16.2 %** | 47.3 | 47.7 |
| 65,239 | 332.4 | 381.1 | **+14.7 %** | 39.5 | 39.4 |
| 97,874 | 277.3 | 313.2 | **+12.9 %** | 33.9 | 33.9 |

`T(n) = a*n + b*n^2` per width, five legs fitted and the deepest held back:

| Width | a (s/token) | b (s/token²) | held-out leg | quadratic share at 97,874 |
| --- | ---: | ---: | --- | ---: |
| 128 | 1.7849e-3 | 1.8756e-8 | 354.4 s against 353.0 s (+0.40 %) | 50.7 % |
| 256 | 1.5013e-3 | 1.7208e-8 | 311.8 s against 312.5 s (−0.23 %) | 52.9 % |

All of the gain is in `a`: **−15.9 %**, which is 1/0.853 — the stage total below
and the served curve agree to the decimal. `b` sits 8 % lower in the second
window, and a forward width cannot change the number of pairs the ten
patternless layers attend, so that 8 % is window drift and not attention; the
same reading the section above reached from another pair of windows. The
quadratic share therefore *rises* with the cheaper width. Decode is unchanged —
five of six legs identical to the decimal, the sixth +1.2 % — and so is memory:
the ring is `next_pow2(513 + rows)`, and 513 + 256 = 769 still lands on 1,024
slots; served GTT 53.7 GB on both.

Numerics at the new width are the same gate as at the old one, on the device and
against the scalar reference: `kolibri1_device_forward_test` at
`KOLIBRI_TEST_PREFILL_CHUNK=256` gives prefill `max_abs 3.28e-3`, `mean_abs
4.57e-4`, argmax equal, a decode step `max_abs 1.94e-3`, argmax equal, and a
517-token prompt across a chunk of 256 exact at 0 — the same three numbers the
128 build produces in the same window. `kolibri1_route_test` covers a row
against a batch of 256 (ids, weights and all 384 logits bit-identical alone and
inside the batch), a sub-epsilon boundary at 256 rows, and the refusals the new
bound makes reachable; `kolibri1_gemv_test` is unchanged and passes, its
forty-eight-slot group included.

### The 15 %, and the 43 % that did not arrive

Both switches on, the same two legs (4,096 and 16,384 tokens, `--gen 32`),
medians over the *full* forwards of each run — 232 forwards at 128 rows, 114 at
256 — milliseconds per token:

| Stage | 128 rows | 256 rows | ratio |
| --- | ---: | ---: | ---: |
| routed experts | 1.279 | 1.132 | 0.885 |
| dense qkv | 0.213 | 0.167 | 0.784 |
| attention | 0.291 | 0.265 | 0.911 |
| output projection | 0.108 | 0.082 | 0.759 |
| shared expert | 0.105 | 0.062 | **0.590** |
| **total** | **1.998** | **1.704** | **0.853** |

The second switch — `GUFO_KOLIBRI_ROCM_ROUTED_STAGE_PROFILE`, added with the
five-stage one and never read until this window — splits the routed stage into
its six launches, with `MoeFinish` carried as the difference to the stage line:

| Launch | 128 rows | 256 rows | ratio | share at 128 |
| --- | ---: | ---: | ---: | ---: |
| counter `memset` | 0.002 | 0.001 | 0.50 | 0 % |
| `MoeRoute` | 0.269 | 0.261 | 0.970 | 21 % |
| gate | 0.335 | 0.294 | 0.878 | 26 % |
| up | 0.325 | 0.288 | 0.886 | 25 % |
| `GateUpAct` | 0.006 | 0.006 | 1.00 | 0 % |
| down | 0.320 | 0.263 | 0.822 | 25 % |
| `MoeFinish` | 0.018 | 0.019 | 1.06 | 1 % |

Three readings.

**Routing does not amortise, and this page understated it by six times.** 0.269
ms per token at 128 rows, 0.261 at 256: the selection walks every row, so its
cost per token is flat in the width, and it is **13.5 % of the forward** at 128
rows and **15.3 %** at 256. [One group table over the
forward](#one-group-table-over-the-forward) recorded it as "under two percent of
the forward" — 31.6 ms per forward divided by a per-token figure. The probe and
the ring agree with each other (631.6 µs per layer at 128 rows is 4.9 µs per
token; fifty layers of that is the 0.247 ms the ring prints), so the number was
never the problem, only the denominator.

**That reading held until the shape changed.** The flatness was not a property
of routing; it was this kernel's decomposition — a wave per expert, every row
re-read under it. Tiled, the same launch is 3.25× cheaper at 256 rows and 0.085
ms per token inside the forward ([The router reads each operand
once](#the-router-reads-each-operand-once)).

**The expert reads are not what limits the stage.** The three GEMV launches go
0.980 → 0.845 ms per token, ratio 0.862, where the bytes promised 0.57. At face
value that is 293 GB/s at 128 rows and 193 GB/s at 256, and 293 is above what
this box delivers from its widest projection (225 GB/s), so the byte model is
wrong somewhere rather than the kernel impossible. Two candidates, and one
counter that chooses between them:

* the router is not uniform: a hot tail touches fewer than the 333 distinct
  experts the uniform model predicts at 128 rows, the stage already moves nearer
  220 GB/s there, and there is correspondingly less to save;
* the launches are bound per slot rather than per byte, in which case time per
  forward follows slots (768 → 1,536) and not experts (333 → 377). Per forward
  the three take 125 ms → 216 ms — between the two readings and nearer the slots.

`MoeRoute` gathers the table anyway and could publish its group count beside it;
the same two legs with that counter decide it in one window. Until then the
claim on the record is the measured one: **doubling the rows bought 14 % of the
routed stage and 17 % of the forward**, and that stage is now 66 % of it.

**The control inside the measurement is the shared expert.** One fixed matrix,
read once per forward, so its bytes per token halve exactly — and its time
follows: 0.105 → 0.062, ratio 0.590, the closest any stage comes to the byte
model. Every stage that gathers falls less than that. Which is also why 256 ends
this lever instead of being a step on it: `kMaxGroupSlots` is the bound, past it
`Experts` steps the forward and a step re-reads, and the milliseconds left are
in the grouped GEMV and the router ([Performance
items](#performance-items)).

What the width costs is one field of the identity: `prefill_chunk_rows=256` keys
the disk-cache tree, so a tree stored under 128 re-prefills once and the weights
are untouched. The group table grows with the bound (`kGroupInts` is 258, so
`GroupTableInts(384)` is about 99 KB) and the gather now runs over sixteen
`kSelectRows` blocks rather than eight.

## Where the forward's time sits

`GUFO_KOLIBRI_ROCM_PREFILL_STAGE_PROFILE=1` brackets the embedding, five stages of
every layer, the output norm and the logits with HIP events and prints one line per
forward (`Executor::Forward`, named after the deepseek runtime's
`GUFO_DEEPSEEK_ROCM_*_STAGE_PROFILE` switches). It exists because a stage's cost
derived from an isolated kernel probe was off by a factor of three — the section
above is what that mistake cost. These are the stages of a live forward,
milliseconds per token averaged over every prefill forward of the run; the totals
agree with the tool's own rate within 1 % (4.183 ms = 239 pp against the run's
238.3).

| Stage | pp2048 @16 | pp2048 @128 | pp8192 @128 | decode step |
| --- | ---: | ---: | ---: | ---: |
| routed experts | 2.212 (52.9 %) | 2.218 (81.3 %) | 2.213 (77.9 %) | 7.55 (41.8 %) |
| dense qkv + o_proj | 1.353 (32.3 %) | 0.302 (11.1 %) | 0.304 (10.7 %) | 6.30 (34.9 %) |
| attention | 0.356 (8.5 %) | 0.108 (4.0 %) | 0.223 (7.8 %) | 1.44 (8.0 %) |
| shared expert | 0.260 (6.2 %) | 0.098 (3.6 %) | 0.099 (3.5 %) | 1.54 (8.5 %) |
| head + norms + embed | 0.002 | 0.002 | 0.001 | 1.19 (6.6 %) |
| **total** | **4.183** | **2.726** | **2.840** | **18.05** |

**The routed experts did not amortise — and the reason was not the experts.**
Their cost is 2.212 ms per token at width 16 and 2.218 at width 128, flat to a
third decimal while every other stage falls. That was read as the grouped table
doing its job, and so as the end of width as a lever. It was the mixture count:
the table held sixteen rows, the wide forward ran eight of them, and a step's
traffic is constant in its own width by construction. The measurement stands;
the reading did not — see [One group table over the
forward](#one-group-table-over-the-forward).

**The dense switch did what it claimed**: 1.353 → 0.302 ms/token at width 128, a
1.05 ms saving inside a 1.457 ms total change, the remainder being attention and
the shared expert amortising too.

**Attention is a depth term, not a width term**: 0.108 ms at pp2048 and 0.223 at
pp8192 with the width identical — the ten patternless layers doubling over four
times the depth while the forty windowed ones sit at their 513-slot cap.

**The decode row calibrates the machine.** The output head moves a 268 MB Q6_K
table in 1.19 ms — **225 GB/s**, which is what this box's memory actually
delivers, from the engine's own widest projection. Against that anchor: routed
experts 660 MB in 7.55 ms = 87 GB/s (39 % of the ceiling) and dense projections
1.07 GB in 6.30 ms = 170 GB/s (76 %). Decode's headroom is therefore the routed
GEMV's — the same item b, this time with split-K across the 512 output rows of a
512-wide expert — and past ~40 K positions attention joins it, which is where the
depth table above says a 39 K session spends most of its step.

## One group table over the forward

`kMaxGroupSlots` was sixteen, so a 128-row forward walked `Executor::Experts`
through eight mixtures and each mixture read the experts of its own rows. The
bound was 128 rows then — one table per forward — and past one block's rows the
table is gathered by a second launch from the ids and counts the selection
leaves behind (`MoeRoute`, `kSelectRows`). It is 256 rows now, which is a
served-width change rather than a new mechanism, measured in [Forward width 256,
served](#forward-width-256-served).

**This corrects the section above.** "The routed experts do not amortise" was
read from 2.212 ms per token at width 16 against 2.218 at 128 and recorded as
the grouped table doing its job. A step's traffic is constant in its own width
by construction, so a stage that steps is flat by construction too: the
flatness was the mixture count, not the ceiling. Width was not spent — it went
to 256 rows and bought 17 % more
([Forward width 256, served](#forward-width-256-served)), which is also where
it stops.

Kernel numbers, synthetic 384-expert tables at the artifact's own shapes, m1.
**Those runs shared the device with a Qwen forward**, so their absolute times
are inflated and only the ratios and the correctness are claims; the model's own
numbers below are from a window with the device to itself. Both columns cover
the same seven hundred and sixty-eight slots; "eight tables" is what a 128-row
forward used to issue.

| Shape | eight tables of 96 groups | one table of 384 groups | ratio |
| --- | ---: | ---: | ---: |
| Q4_K gate 512×2560 | 8 × 351.7 = 2,813 µs | 1,491 µs | **1.89** |
| Q4_K down 2560×512 | 8 × 347.2 = 2,778 µs | 1,496 µs | **1.86** |
| Q6_K gate 512×2560 | 8 × 456.5 = 3,652 µs | 1,853 µs | **1.97** |
| Q6_K down 2560×512 | 8 × 460.6 = 3,685 µs | 1,889 µs | **1.95** |

The eight-table column reads a 96-expert table eight times (768 matrix reads)
and the one-table column reads 384 experts twice apiece (384 reads), and both
run at 190-226 GB/s of the bytes they touch: the gain is fewer reads, not a
faster kernel, which is why it lands close to the byte ratio of two and not
above it.

Routing itself, per layer, same shared device: 30.2 µs at one row, 109.6 at
sixteen, **631.6 µs at 128** — the wide number carries the counts memset and the
gather launch the narrow ones never pay. Across fifty layers that is 31.6 ms of
a 128-row step — 0.25 ms per token, which is **12 % of that step and not the
"under two percent" this sentence used to say**: it divided a per-forward figure
by a per-token one. The ring agrees with the probe from inside a live forward
(0.269 ms per token at 128 rows, 0.261 at 256, flat because the selection walks
every row), so what was wrong was the denominator and not the measurement
([Forward width 256, served](#forward-width-256-served)). The table is not free:
it is the routed stage's second item, and it is why that stage did not halve
when its bytes did.

The three numbers at the start of that sentence were the narrow shape's, and
the last of them was flat for the reason above: tiled, `MoeRoute` is 0.087 ms
per token at 256 rows, and the stage's second item is the three expert
launches again.

Correctness on the device: `kolibri1_gemv_test` matches the FP64 dot on every
case, now including a group of forty-eight slots compared bit for bit at pass
widths one, four, eight and sixteen, and a lone slot held against a table whose
other slots belong to other experts; `kolibri1_route_test` covers tables at nine
and forty-eight rows (widest group one slot per row), a row's ids, weights and
logits identical alone and inside a batch of 128, and the new refusal past
`kMaxGroupSlots`. 13/13 CPU checks.

### The same change on the model, device to itself

A window on m1 with nothing else on the GPU: the two builds — `HEAD` and
`HEAD~1` of this change, checked out and rebuilt between the runs *inside the
one window*, so no machine state separates them — against the same artifact.

`kolibri1_device_forward` against the scalar reference, on the real weights, at
both widths: prefill of 37 tokens at chunk 16 and of 261 tokens at chunk 128
give `max_abs 3.28e-3`, `mean_abs 4.57e-4`, argmax equal; decode step
`max_abs 1.94e-3`, argmax equal; the chunked case across the width boundary is
exact at 0 in both. The gathered table does not move the numerics.

Stages of a live prefill, ms per token, `GUFO_KOLIBRI_ROCM_PREFILL_STAGE_PROFILE`,
pp2048 at chunk 128, ranges over the forwards of a run:

| Stage | old build | new build | ratio |
| --- | ---: | ---: | ---: |
| routed experts | 2.225 – 2.253 | 1.103 – 1.189 | **1.96×** |
| dense qkv | 0.196 – 0.283 | 0.201 – 0.289 | — |
| attention | 0.030 – 0.099 | 0.030 – 0.095 | — |
| o_proj | 0.098 – 0.099 | 0.102 – 0.106 | — |
| shared expert | 0.097 – 0.098 | 0.100 – 0.106 | — |
| **total** | **2.678 – 2.761** | **1.601 – 1.712** | **1.67×** |

The routed stage lands where the arithmetic said it would: 6-of-384 over 128
rows touches 384 × (1 − (1 − 6/384)<sup>128</sup>) ≈ 331 distinct experts
against the ≈ 84 per mixture × 8 mixtures = 672 the stepped version reads, so
2.0× fewer bytes and 1.96× measured. Every other stage sits where it was, which
is the control that says the window is comparing one change and not one machine.

| Rate | old build | new build | ratio |
| --- | ---: | ---: | ---: |
| pp2048, engine-direct, context 2,144 | 2.713 ms/token stage total | 1.601 ms/token stage total, 601.6 tok/s measured | **1.70×** on the stage total |
| pp8192, engine-direct, context 8,300 | 351.6 tok/s | **563.4 tok/s** | **1.60×** |
| decode, context 8,300 | 52.90 tok/s | 52.80 tok/s | — |
| pp13,057, served, one session, context 16,384 | 336.85 · 336.90 tok/s | **539.03 · 537.45** | **1.60×** |
| pp10,438, served, same server | — | 557.82 tok/s | |

Served rows are `tools/kolibri/served_pp.py`: one marker-prefixed prompt per
run — a repeated prompt would be a byte-exact prefix hit rather than a prefill —
`max_tokens: 1`, and the server's own `timings.prompt_per_second`.

The old build's engine-direct 351.6 tok/s at pp8192 is within 0.2 % of the
352.1 recorded before the change ([Where the forward's time
sits](#where-the-forwards-time-sits)), so the two windows agree on the machine.
Served, the identical prompt sent twice came back `cached=10438` in 0.01 s —
the continuation cache is untouched by the new gather.

What does not carry over: the served pair recorded under
[Served forward width](#served-forward-width) — 505.7 · 503.0 tok/s at 13,030
prompt tokens — is not reproduced by the old build in this window at 13,057
tokens, which reads 336.9. The runs differ in `--context` (131,072 there
against 16,384 here) and the earlier harness left no request-side record to
replay, so no cause is claimed; what is claimed here is the ratio, measured
old-against-new with one set of flags and one prompt builder. Read the served
absolutes of this section as the current configuration's, and the 1.60× as the
change.

Decode is untouched by construction — one row's table never needed more than
one block — and measured so: 52.9 against 52.8 tok/s at the same depth.

## Decode's routed stage, apart from the memory it never touches

Two numbers had been read as one claim. The live stage is
`660 MB in 7.55 ms = 87 GB/s`
([Where the forward's time sits](#where-the-forwards-time-sits));
`kolibri1_gemv_test`'s TIME cases reported 305-348 GB/s for the same six-group
decode table. Both were right, and neither was the decode step: the test's twenty
rounds read the same 4.42 MB of expert weights twenty times, and 4.42 MB fits
this generation's L2, while a decode step reads that block once and not again for
fifty layers.

The helper now prints a second line wherever one is possible, giving each round
experts no earlier round touched (`gemv_test.cpp`, `Timed`). On this box — a
served Qwen holding its 95 GB of GTT, nothing computing on it — the Q4_K gate
launch reads **276 GB/s over one table and 167 over sixty disjoint ones**, its
down-shape twin 275 and 157, and the Q6_K pair 341/181 and 404/189. Binding the
launch to all 384 experts is still the loss it was warm (26.5 µs against 41.2 on
the cold six-group table), so the bound `Experts` chooses was never an artefact
of the warm measurement.

`tools/kolibri/decode_gemv_probe.cpp` is the same experiment with more control
and nothing else on the device: 32 rotations over disjoint experts, 135-270 MB
per round, minimum over four interleaved rounds, m1 with no other kernel
running. Its numbers run a little higher than the in-tree ones for the same
reason it exists — one launch per variant, not one variant per launch — and it is
what the launcher should be steered by.

| One launch | Engine's policy | Rotating | Best variant in the probe |
| --- | --- | ---: | ---: |
| dense qkv, m=6144 k=2560 Q4_K | split 4 | 207 GB/s | 209 (split 2, 128 threads) |
| routed gate or up, m=512 k=2560 Q4_K | split 4 | 187 GB/s | 193 (split 2, 128 threads) |
| routed down, m=2560 k=512 Q4_K | split 1 | 184 GB/s | 191 (split 2, 128 threads) |
| routed down, m=2560 k=512 Q6_K | split 1 | 183 GB/s | 206 (split 2) |
| gate and up together, `blockIdx.z` over the two tensors | — | 206 GB/s | — |

**The split policy is right where `k` gives it room and 4-12 % short where it
does not.** `kSplitWaves` chooses 1 below four super-blocks
(`gemv.hip.cpp:424-431`), which is the down projection's case at `k=512`, and a
split of 2 across its two units is where the difference sits — 12 % on the Q6_K
shape. That difference is not free: splitting a row's reduction is a different
summation order, and the probe reads `rel 9.2e-5` to `6.1e-3` between a split and
the reference's split on random weights (cancellation, not the artifact's
weights), where a variant that keeps the split reads `0.00e+00` in zero rows. A
split change is a numerics change and owes the golden forward a re-check.

**The two halves of a routed row belong in one launch.** Gate and up read the
same row of `h` and write two buffers, so a launch carrying both on
`blockIdx.z` changes neither tensor's arithmetic — it reads `0.00e+00` against
the reference. 8.44 MB in 42.9 µs is 206 GB/s where two launches of 4.42 MB run
at 187: **9.3 % of the stage's two widest launches, and bit-safe.**

**Prefetching the next block loses at every shape** — 152-184 GB/s against the
183-207 of the same geometry without it, worst on the gate shape at 23 %. That
retires the theory that the unit walk is bound by a dependent load chain: the
registers a prefetch needs cost more than the latency it hides.

**The router is the stage's second item and had never been measured.**
`KOLIBRI_TIME=1 ./kolibri1_route_test` gives 31.4 µs per layer at `rows=1` on
this box while a served Qwen forward held the device, with the 3.93 MB F32 gate
matrix hot in cache across its 200 rounds: **≥1.57 ms per token**, the order of a
whole gate projection, for two launches — `RouterKernel` and a single-block
`SelectKernel`. The same test's wide rows (130.5 µs at 16, 698.6 at 128) are the
prefill shape, where `GatherGroupsKernel` joins them.

So the decode step's 7.55 ms of routed stage was read as roughly 3.8 ms of
grouped GEMV at the rate a cold launch actually gets, at least 1.6 ms of
routing, and about 2.2 ms the stage bucket could not attribute to any measured
kernel. That last piece is why `GUFO_KOLIBRI_ROCM_ROUTED_STAGE_PROFILE` exists:
set together with the five-stage switch, `Forward` prints a second line
splitting the stage into `memset`, `route`, `gate`, `up`, `act`, `down` and the
`finish` left over (`Executor::Experts`, `Executor::Forward`). Markers are
recorded only for a single-piece forward, and each costs about a microsecond, so
its decode line reads high: it answers *which launch* the time sits in.

**The window came, and the 2.2 ms has a name.** The two profile legs of
[Forward width 256, served](#forward-width-256-served) carry decode rows too —
62 steps per leg behind 4 K and 16 K prompts, nothing else on the device, and
the two builds' medians identical to the third decimal, which is itself the
control that width does not touch decode. One step, ms per token: `route` 2.740,
`down` 1.643, `gate` 1.363, `up` 1.360, `finish` 0.586, `memset` 0.265, `act`
0.249 — 8.21 of a 19.96 ms step, with qkv 3.55, o_proj 2.62, attention 2.53,
shared expert 1.52 and head 1.16 around it. The routed stage is therefore 4.37 ms
of grouped GEMV, 2.74 ms of routing — 1.7× what the isolated probe charges,
because a live step does not run its gate matrix hot across 200 rounds the way
the probe does — and **1.10 ms of three launches that move almost nothing**: the
384-counter memset at 5.3 µs per layer, `GateUpAct` at 5.0, `MoeFinish` at 11.7.
Launch latency, 5.5 % of a decode step, and the first thing a fusion removes.

## The router reads each operand once

`RouterKernel` gave one wave one expert and let it walk every row of the
forward, so a row of `x` was read once per expert: 256 rows over 384 experts is
two gigabytes through the caches for half a gigaflop. That is the shape that
made the routed stage flat per token from sixteen rows to two hundred and
fifty-six, and it is now chosen per launch.

The tiled shape covers eight rows against eight experts in a block — four waves
of two rows held in registers, the eight gate rows staged through 8 KB of
shared memory, the hidden dimension in tiles of 256 floats. Timed both ways over
a fixed synthetic input — the same source built with `kRouterTiledMinRows` at
one and then past every width the tool sweeps, one session, logits hashed, µs
per launch (`tools/kolibri/router_hash.cpp`, built by `build_router_hash.sh`,
device to itself):

| Rows | one wave per expert | tiled | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 29.4 | 51.0 | 0.58 |
| 2 | 33.6 | 45.5 | 0.74 |
| 4 | 41.9 | 45.6 | 0.92 |
| 8 | 61.9 | 48.8 | **1.27** |
| 16 | 106.7 | 66.5 | 1.60 |
| 32 | 177.3 | 91.3 | 1.94 |
| 64 | 316.0 | 140.3 | 2.25 |
| 128 | 625.8 | 209.0 | 3.00 |
| 256 | 1,217.2 | 374.5 | **3.25** |

An earlier window of the same sweep agreed within 8 % at every width above one
row — the tiled launch at 256 rows repeated to 0.1 % — while the one-row
launches moved by up to 14 %, which is the noise of a 30 µs launch and not a
difference between the shapes.

**The logits of both shapes are equal at every width.** Both reduce a row
against an expert in the same order — lane `l` over the quads l, l+32, l+64 …,
four float chains, the narrowing to double, the four-way add, the butterfly —
and a shared-memory tile only cuts that sequence at a fixed point, so
re-tiling cannot move a logit: the sweep hashes all 384 logits of every width
above and both columns agree bit for bit. The golden forward reads it the same
way afterwards: prefill `max_abs 3.28e-3` with argmax equal, decode step
`1.94e-3`, a 37-token prompt across a 16-row chunk exact at 0 — the same
figures the page carried before the change.

**Below eight rows the tiles lose, and occupancy is why.** At one row the
narrow launch spreads its 3.9 MB of gate matrix over 96 blocks (384 waves,
about nine per compute unit) and the tiled one over 48 (192 waves, about five),
with nothing for a tile to reuse in between. The two lines cross between four
and eight rows, so `kRouterTiledMinRows = 8`.

Inside the engine — the same source built at both settings of the threshold,
both profile switches on, 4,096 tokens at chunk 256, device to itself, mean over
the forwards, ms per token:

| | narrow | tiled | |
| --- | ---: | ---: | --- |
| `route` of the routed stage | 0.251 | **0.085** | 2.9× less |
| routed stage | 1.014 | **0.845** | −17 % |
| forward | 1.464 | **1.288** | −12 % |
| prefill 4,096 tokens | 682 tok/s | **775 tok/s** | **+13.6 %** |
| decode 32 tokens | 52.1 tok/s | 52.1 tok/s | unmoved |

Decode is unmoved by construction: one row is below the threshold, and the two
legs print `route` 2.66 and 2.71 ms per step for it, which is the same launch
either way. That is also the counter the
[decode section](#decodes-routed-stage-apart-from-the-memory-it-never-touches)
said the ring could not give. `SelectKernel`, the gather and the counters are
the same launches in both legs, so the 0.166 ms per token that left `route` was
the router's own read, and what stays at 0.085 — 0.44 ms of a 256-row layer,
against the 373 µs the kernel takes by itself — is the selection around it.

## Not measured

| Cell | Status |
| --- | --- |
| Concurrency (2/4/8 users) | TODO — one session per forward today; `SupportedPlans` is serial width 1 |
| Q6_K / Q8_0 / Q5_K_M | TODO |
| Contexts beyond 97,873 | The cold curve runs to 97,874 at both shipped widths, 128 and 256 ([Served depth](#served-depth-cold-at-128-rows), [Forward width 256](#forward-width-256-served)); the only deeper leg, 108,627 in [Served path](#served-path), is a width-16 build, so nothing above 97,873 is measured on what ships |
| llama.cpp prefill beyond 2,048 | TODO — the table above stops at pp2048, so its ratio says nothing about depth; the reference build has never been run at 8K/32K/99K |
| Continuation cache, prefix reuse | Measured ([Served path](#served-path), [Quality](QUALITY.md)) |
| Tool-call round-trip | Measured ([Quality](QUALITY.md)) |
| Functional suites (13) | Recorded across three runs — 7 pass / 6 fail, every failure with a cause, four harness-side fixes landed but not re-run ([Quality](QUALITY.md#functional-qualification)) |
| Whole-model prefill on the matrix-core path | Measured — 1.11× best, 1.38× median at width 128, spread 29 % → 2.4 % ([Prefill width, both dense paths](#prefill-width-both-dense-paths)) |
| Served forward width | Measured — 1.83× at 1,572 and 2.00× at 13,030 prompt tokens, decode flat, ttft halved ([Served forward width](#served-forward-width)) |
| Whole model on one group table | Measured in one window, both builds: routed stage 2.225-2.253 → 1.103-1.189 ms/token (**1.96×**), prefill total 2.72 → 1.61 ms/token, pp8192 351.6 → 563.4 tok/s (**1.60×**), served pp13k 336.9 → 538.3 tok/s (**1.60×**), decode unchanged, golden forward exact against the scalar reference at chunk 16 and 128 ([The same change on the model, device to itself](#the-same-change-on-the-model-device-to-itself)) |
| Routed stage split by launch | Measured, device to itself, both switches. Prefill at 128 and 256 rows — `MoeRoute` 0.269/0.261, gate 0.335/0.294, up 0.325/0.288, down 0.320/0.263, `memset` + `act` + `finish` under 0.03 ms per token — and decode, routing 2.74, three GEMVs 4.37, three small launches 1.10, of an 8.21 ms stage ([Forward width 256, served](#forward-width-256-served), [Decode's routed stage](#decodes-routed-stage-apart-from-the-memory-it-never-touches)) |
| Router shape | Measured on this device, the same source built at both settings of `kRouterTiledMinRows`: 29.4 µs against 51.0 at one row, 61.9 against 48.8 at eight, 1,217.2 against 374.5 at 256 (**3.25×**), logits equal at every width and across the builds; in the engine `route` 0.251 → 0.085 ms per token, forward 1.464 → 1.288, prefill 682 → 775 tok/s (**+13.6 %**), decode 52.1 unmoved; golden forward `max_abs 3.28e-3` with argmax equal ([The router reads each operand once](#the-router-reads-each-operand-once)) |

## Performance items

1. **Forward width — taken three times, and closed.** The runner passed 128:
   served, 1.83× at 1,572 prompt tokens and 2.00× at 13,030, decode unmoved
   ([Served forward width](#served-forward-width)). What made that width pay the
   first time was the dense matrix-core path underneath it, without which extra
   rows are mostly re-reads. What capped it afterwards was `kMaxGroupSlots` at
   sixteen rows: `Experts` stepped the wide forward through eight mixtures, each
   reading its own experts, which is why the routed stage read as finished at
   2.21 ms per token. The bound is the forward now, and the second taking is
   measured: **1.60× prefill served and engine-direct, decode unmoved**
   ([One group table over the forward](#one-group-table-over-the-forward),
   [The same change on the model, device to
   itself](#the-same-change-on-the-model-device-to-itself)). The third, 128 to
   256, is measured too: **+12.9 % to +18.4 %** served across 3.2 K to 97.9 K,
   decode and memory unmoved, numerics identical to the 128 build's
   ([Forward width 256, served](#forward-width-256-served)) — and it is the last
   one. The routed stage gave up 11 % of itself on bytes that halved, because
   0.269 ms per token of it was `MoeRoute`, which walked every row and so never
   amortised — since re-tiled, and the stage falls with the width again
   ([The router reads each operand once](#the-router-reads-each-operand-once));
   past `kMaxGroupSlots` a forward is stepped and a step re-reads.
   The cost that stays on the record: `prefill_chunk_rows` is part of the
   disk-cache identity (`inference_backend.cpp:348`), so a width change
   re-prefills every stored tree once.
2. **The prefill GEMM: tensor cores in two steps; a is built, b is next.**
   The reference's advantage is a tensor-core path, not a better schedule, and the
   ladder is measurable in front of us — `gemma4` was cut from the same GEMV and
   already switched above 16 rows (`gemma4/kernels/rocm/executor.cpp:476-540`),
   and `qwen38_flash_next` runs dense on F16/int8 WMMA and hipBLASLt with routed
   experts on compacted buckets (`qwen38/kernels/rocm/executor.cpp:1543-1621`,
   `:2504-2575`).
  - **a) Dense projections and the shared expert — built.** `Executor::Project`
    hands a block of `kMatrixCoreRows = 32` rows or more to
    `gufo::hip::LaunchBatchedQuantGEMMPreQuantized` (`qwen/hip/ops/gemm.hpp:279`,
    dispatch `qwen/hip/kernels/prefill_quant_gemm.hip:1517-1537`), the entry
    `gemma4` uses for the same job; under 32 rows, and for every decode row, the
    batch-invariant GEMV is untouched. The block's input is staged once into Q8_1
    tiles (`QuantizedActivationBytes`), one buffer per executor, sized by the
    widest projection input — and no new link edge, since those kernels compile
    into `gufo_core` (`qwen/CMakeLists.txt:34`), which this engine already links.
    Measured per shape at the artifact's own widths with random weights
    (`tools/kolibri/prefill_gemm_probe.cpp`, so without the artifact and without a
    device window): over the seven projections of one layer, the tiled path runs
    0.76 of the GEMV at 16 rows — the reason the threshold is 32 and not 17 — then
    1.46 at 32, 2.95 at 64, 3.51 at 128 and 4.97 at 256. Per token that is
    0.81 ms of projection at 32 rows, 0.42 at 64, 0.37 at 128 and 0.26 at 256,
    against 1.30 ms for the GEMV at any width. That last number is about three
    times the truth in a live layer: measured end to end at width 128 the saving is
    0.30 ms/token, 1.11× the best and 1.38× the median run of the GEMV build, and
    the expert GEMV is what remains on the curve (item b, ~650 MB of the ~900 MB a
    prefill token reads). What the wide path trades is rounding — its activations
    are int8 per 32 values, and against the GEMV on the same input the largest
    deviation is 3.2e-3 to 4.6e-3 of a projection's largest output, the same trade
    Qwen and gemma4 already make.
  - **b) Routed experts** (2.4 of 6.9 GFLOP, ~90 % of the bytes) as a WMMA tile
    over per-expert padded buckets — `gemma4/kernels/rocm/moe_prefill.hip.cpp`
    is Q4_K/Q6_K → binary16 already, and `MoeRoute` emits the `counts` such a
    `RoutedCompact` needs (`route.hip.cpp:358-365`). With every expert read once
    per ≥256-row chunk, traffic drops to 93-184 MB/token, a memory floor of
    800-1600 tok/s, and the tensor core supplies the FLOPs. This is the step that
    reaches the reference's 1,217 pp, and it inherits the accuracy review that
    gemma4 already accepted: Q4_K products rounded through binary16.
    It will not be bit-identical to the decode GEMV either; that kernel's
    batch-invariance contract (`gemv.hip.cpp:11-27`, compiled
    `-ffp-contract=off`) stays in force for decode, and prompt and decode
    rounding already differ in Qwen and gemma4.
3. **Two grid and allocation wastes to sweep up while in there.**
   `AttentionQueryRowsPerBlock()` returns 32 while the kernel advances 64 rows per
   block (`attention.hpp:59-61` against `attention.hip.cpp:131-133`), so at width
   16 three of four in-block row groups are masked-idle and up to half the launched
   blocks do nothing (coverage is still complete). And `MoeRoute` does a
   `hipMallocAsync`/`hipFreeAsync` pair plus a single-block top-6 kernel **per
   16-row step** (`route.hip.cpp:546`, `:576`) — ~25,600 of both per 8,192-token
   prompt. Effect sizes not measured.
4. **Repeated-cell measurement protocol.** `~/kolibri-stats.py` exists; the
   engine-direct harness should run n repetitions itself rather than leave it to
   the caller's shell loop.
5. **The patternless layers' KV is now the bandwidth bill.** 2.07 GiB per
   generated token at 108,627 positions, against ~2 GB of weights — the reason
   decode falls 59.6 → 51.1 → 31.8 tok/s with depth. Binary8 KV, or sharing heads
   across the ten full-attention layers, is the only lever that keeps decode flat
   as context grows; widening the sliding window is not, since those layers are
   precisely the ones that cannot slide.
   The vendor agrees that this is the lever — the released stack is served with
   an FP8 KV cache, and report §2.1.3 with Appendix I.4.1 argues that per-head
   QK normalisation bounds every query and key element inside the FP8 range so
   the cache cannot overflow. That is a quality argument on their FP8
   checkpoint and Hopper hardware, not a measurement of a narrower cache on
   Q4_K_M here; the recall probe is still what would have to pass.

6. **Decode's routed stage: attributed now, and ranked by that.** One quiet
   window with both switches closed the last piece: of an 8.21 ms routed stage
   inside a 19.96 ms step, 4.37 ms is the grouped GEMV, 2.74 ms is routing and
   **1.10 ms is three launches that move almost nothing** — counter `memset`
   0.265, `GateUpAct` 0.249, `MoeFinish` 0.586
   ([Decode's routed stage, apart from the memory it never touches](#decodes-routed-stage-apart-from-the-memory-it-never-touches)).
  - **Fuse the tail of a routed row.** Those three are 5.0-11.7 µs per layer
    apiece on payloads of a few kilobytes: launch latency, 5.5 % of a decode
    step. `act` belongs in the down GEMV's input read, `finish` in its epilogue,
    and the counters in neither. This is now the best-evidenced item on the list
    and the only one whose size is measured rather than modelled.
  - **Fused gate and up**, one launch over `blockIdx.z`: 206 GB/s against 187 for
    the same bytes in two launches, arithmetic unchanged (`0.00e+00` against the
    reference). 4.4 µs per layer, ~0.22 ms per token, and it needs no re-baseline.
  - **No counters on the decode path.** `Experts` memsets 384 counters every
    layer and `SelectKernel` `atomicAdd`s them
    (`route.hip.cpp:362-365`), but only `GatherGroupsKernel` reads them and that
    runs only past `kSelectRows` (`route.hip.cpp:510`, `:576`). Passing
    `counts = nullptr` for a forward of at most `kSelectRows` rows deletes an
    atomic per slot and one serialized launch per layer. The ring prices it now:
    `memset=` is 0.265 ms per token of decode, 1.3 % of the step for zeroing 1.5
    KB — small, and the same launch the item above wants deleted anyway. A wide
    forward still zeroes before it gathers, since the memset lives in the same
    place for it.
  - **The split at `k=512`**: 4 % on the shipped Q4_K down, 12 % on a Q6_K one.
    The artifact's routed down is Q4_K — 13.2 MB per layer only adds up as three
    launches of 4.42 MB — so the larger number belongs to a build still on the
    list above. Both sit behind a golden re-check, because a split reorders a
    reduction.
  - **The router was second, at 2.74 ms per token** — 1.7× what the isolated
    probe charges it, because a live step does not keep the gate matrix hot
    across 200 rounds the way the probe does. On the prefill side the same
    segment was 0.269 ms per token at 128 rows and flat at 256, which is the
    reason a wider forward stopped paying. Both halves of that have an answer
    now: the counter came as a second shape of the same kernel rather than a
    new instrument, and it says the read was the cost — tiled, prefill `route`
    is 0.087 ms per token, and the ~0.07 ms per 256-row layer left over the
    kernel's own time is the selection and its gather
    ([The router reads each operand once](#the-router-reads-each-operand-once)).
    Decode still runs the narrow shape, so its 2.74 ms stands: 3.93 MB of F32
    gate matrix is ~20 µs at the machine's rate, and the gap is the single-block
    `SelectKernel` plus logits nobody reads.
