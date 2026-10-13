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

## Not measured

| Cell | Status |
| --- | --- |
| HTTP serve, `gufo chat` / `/v1/chat/completions` | TODO — architecture not wired to the server yet |
| llama.cpp reference comparison | TODO — needs the conversion's patched build as the baseline |
| Concurrency (2/4/8 users), continuation cache | TODO |
| Q6_K / Q8_0 / Q5_K_M | TODO |
| Contexts beyond 8,192 | TODO |
| Long-context recall, tool-call round-trip | TODO |

Weight load is 12–35 s for the 45 GiB file depending on page-cache state; it is
not part of any timed region above.
