# Kolibri-1 quality

Two separate questions, kept apart: does this engine compute the architecture
that the weights describe, and does the device agree with the host.

## Provenance chain

```
patched llama.cpp  ↔  scalar reference  ↔  device forward
   (run_parity.sh)      (device_forward_test.cpp)
```

**Reference against an independent implementation.** `kolibri1` is not stock
llama.cpp; the GGUF conversion ships a patch for it, and that patched build is
the second implementation here. `tools/kolibri/run_parity.sh` runs both engines
over the same synthetic artifacts and token ids and compares full-vocabulary
logits at every position, with `MAX_ABS` 1e-2 and `MAX_KL` 1e-5: single-token
prompts (where a RoPE bug is invisible), multi-token prompts (where it is not),
positions past the sliding window, and the patternless layer at first, middle
and last position. Synthetic artifacts come from
`tools/kolibri/make_tiny_artifact.py`, so the shapes are cheap to vary while the
quant mix matches the released file.

**Device against that reference.** `tests/models/kolibri1/device_forward_test.cpp`
loads the released `Kolibri-1-Q4_K_M.gguf` (45,254 MiB) on `gfx1151` and
compares the device's logits row against the host oracle — October 13, 2026:

| Check | max_abs | mean_abs | argmax |
| --- | ---: | ---: | --- |
| gathered embedding row vs the host table | 0 | 0 | — |
| prefill, 37-token prompt | 0.0033 | 0.00046 | agree (5513) |
| one decode step after it | 0.0019 | 0.00028 | agree (46) |
| prefill of the same prompt split across a chunk of 16 | 0 | 0 | agree (5513) |

The oracle is a single FP32 pass per row; the device splits the projections over
K, gathers embedding rows before the projection, and adds expert contributions in
an order that follows the batch. So bit-equality is not the bar — the last line
of the table is exact because the split happened to reduce identically there,
not because it is guaranteed. Tolerances: 0.5 / 0.02 (max/mean) for the checks
against the oracle, 0.02 / 0.002 for chunk invariance; both would fail by orders
of magnitude on a mis-bound tensor, a wrong head count, or window clipping lost
across a chunk boundary.

Below the whole-forward check, the pieces have their own tests against the same
oracle: `gemv_test.cpp` (quantized projections and the grouped expert launch, in
max-abs and mean-relative terms, including the launch-bound case where more rows
than experts are routed), `route_test.cpp` (bias selection and the
un-renormalized sigmoid weights), `qkv_post_test.cpp` (per-head QK norm and
RoPE), and `attention_test.cpp` (window edge, ring wrap, the no-position layer,
and 128K decode). The chat renderer and tokenizer are pinned by byte-exact
goldens in `chat_template_test.cpp`, `tokenizer_test.cpp` and
`tokenizer_model_test.cpp`.

## Not verified

- **Text quality.** No benchmark suite has been run against the original FP8
  model, and the conversion itself warns that quantization can reduce accuracy.
  Nothing here says how good this model is; only that the engine computes it.
- **Long context.** The device check runs at a few dozen tokens. The cache
  geometry supports the full 262,144, but recall or perplexity at depth is untested.
- **Tool calling on the device.** The tool grammar is produced and tested as
  text; no measurement that a served tool call round-trips.
- **Other quants.** Q6_K, Q8_0 and Q5_K_M have not been loaded. The binder
  accepts Q4_K, Q6_K and F32 rows and names the tensor it rejects, so an
  unexpected per-tensor layout fails loudly rather than silently — but that is
  a load-time guarantee, not a correctness measurement.
- **Concurrency.** Everything measured is one user. Multi-session rings and the
  continuation cache are not yet exercised for this architecture.
