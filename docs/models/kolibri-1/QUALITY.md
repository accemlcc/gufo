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

## Served path

`gufo serve llm` was driven over HTTP on October 13, 2026 with
`tools/kolibri/serve_check.py` and the server's own `event=completed` line. A
chat turn returns; reasoning and answer arrive on separate channels
(`reasoning_content`, `content`) with neither polluted by the other;
`tool_choice: "required"` returns a structured call; and two conversations
interleaved as A1 B1 A2 B2 A3 answer from their own state — each reply carries
its own planted code and person, never the other's.

Cache behaviour was measured in both directions, at 92 tokens and at 39,292:

- A prompt the server has never seen pays for everything except the conversation
  frame — the system line plus the opened assistant tag, 42 tokens on this
  template, which the runner checkpoints after the first request:
  `prompt=93 prefill=51 cached=42`. A same-length context with different bytes
  is treated the same way at depth: 39,292 tokens reused 42 and prefilled 39,250.
- A conversation repeated is reused whole: `prompt=39,292 prefill=0
  cached=39,292` in 180.7 ms against 247.7 s cold, answer text identical.
- On every leg, hit or miss, `prefill_tokens + cached_tokens == prompt_tokens`.
  That invariant is what tells a taken cache apart from a usage field counting
  something else; the probe asserts it rather than reading one number.
- A prefix shared with a finished turn is reused at depth, not just at chat
  length: a 109,656-token prompt sharing its first 9,845 tokens with the
  calibration turn that preceded it came back `cached_tokens=8192`,
  `prefill_tokens=101464`. It stopped at `4 × 2,048`, below the end of the
  shared text, because that calibration was a *different conversation*: nothing
  was checkpointed at the byte where the two prompts part, so the only
  candidates left were the grid points of `text_model_runner.cpp:753`.
- Inside one conversation the boundary is a turn, not a rounding: reuse ends at
  the end of the **second-to-last closed turn**, its four-token assistant frame
  included, so every continuation lags one turn of its own growth behind the
  prompt end. Measured on a conversation whose turns grow by 598 tokens
  (`cache-depth.json`, preset `kolibri1`, temperature 0, seed 31, rc=0): 20,606
  → 0 cold, 21,204 → 20,602, 21,802 → 20,606, three branches at 21,825 →
  21,204, resume 21,820 → 21,204, and the same request once more 21,820 →
  21,820 with `prefill_tokens=0`. `21204 − 20602 = 598 + 4` and
  `21802 − 20606 = 2 × 598` are the two halves of that sentence;
  `cached + prefill == prompt` holds in all sixteen rows and every cold control
  caches 0 with the full prompt as prefill.
- The first turn after a leg that ended without a trailing user turn pays the
  grid as well, and not because of a foreign prefix: such a leg's stable prefix
  is its own prompt end, so the checkpoint it keeps contains the opened
  assistant tag, which a real turn does not send — the best remaining match is
  again a grid point. Measured on a 128-line block: block 6,328 → 0 cold, first
  turn 6,331 → 6,144 (3 × 2,048), next turn 6,351 → 6,308 (43 tokens refilled,
  which is the previous turn's stable prefix), sent again 6,351 → 6,351.
- Recall at 108,627 and 109,656 scored all three planted facts (10/50/90 % of the
  prompt). At the second depth the budget ran out mid-reasoning
  (`generated_tokens=512`, `finish=length`, empty `content`), so the answer was
  read from the reasoning channel — stated because a scorer that only looked at
  `content` would have reported a miss the model did not have.

`tools/kolibri/serve_check.py` re-runs all of it: `sessions` (interleaving),
`recall --tokens N` (three facts planted at 10/50/90% depth), `prefix` (cold,
warm, and a same-size different-bytes control), `coldpair` (two prompts the
server has never seen, so a hit cannot be explained by an earlier run). Give it
`--log` with the server's log file and the accounting comes off the server
instead of off `usage`.

`tools/kolibri/prefix_probe.py` walks that ladder deliberately: `off` and
`thinking` families, block / first turn / continuation / repeat, and a nonce in
the block header so a second run against the same server cannot read itself
warm. It asserts the accounting identity on every leg, not one cached number.

## Functional qualification

`tests/functional/run.py --sampling-preset kolibri1`, thirteen suites against
`gufo serve llm --sessions 1 --context 32768` on October 13, 2026. Three runs are
in this record and every row says which one it stands on: the full sweep
`functional-kolibri-20261013-092324`, then `serveprobe9-20261013-122921` and
`functional-kolibri-20261013-131254` after the harness was told what this template
actually promises. **Seven pass, six fail, and none of the six is unexplained.**

| Suite | Stand | What was actually there |
| --- | --- | --- |
| discovery, tools, structured, long-context, sampling-defaults, progress | passed | 092324. `discovery` asserts text-only modalities, which is what `/v1/models` reports. |
| cache-depth | **passed** | Re-run in `serveprobe9-…-122921`, 13 checks. `cache_depth.py` had demanded a frontier no template promises; with the preset's own floor (`previous − growth − 96`) each leg lands where the rule above says, and a fall back to the 2048 capture grid — or to zero — still fails. |
| cache-shared-prefix | not decided | `failed (900.019 s)` in 131254 with **no FAIL line in the suite log**. `run.py:256` defaults `--suite-timeout` to 900 s and `run.py:359` enforces it, while the four uncached 21k controls need ~1500-1700 s; it got through eight checks. A frame, not a behaviour — `--suite-timeout 1800` covers it and the assertion is untouched. |
| tool-reasoning | 67 of 68 checks | 433.071 s in 131254. The single failure is dataset 68, `unfinished_inline_then_call_filename_buffered`: the call parsed perfectly (`terminal`, `{"command": "pwd"}`, `finish=tool_calls`) but `text` and `reasoning` came back empty with `completion_tokens: 14` where the case expects 92 characters of prose plus the call — fourteen tokens cannot have carried that text and lost it, so the model never wrote it. `literal_protocol_token_word_buffered` and `_stream` ran and passed (`text` = `EOS = "`, no tools). |
| stops | failed at 092324 | The control asserts the *unstopped* answer is longer than twelve characters; the model answered `LANTERN` — seven characters, `finish=stop`, three tokens. A precondition written for talkative models. `openai_sdk.py` now picks a marker a one-word answer can contain (`4c5370e7`), **not re-run since**. |
| conversation | failed at 092324 | `body()` always sent `chat_template_kwargs={enable_thinking: false, reasoning_effort: "low"}`, and an explicit effort outranks `enabled` in this template — the documented Jinja rule. Effort won: 48 prompt tokens without the kwargs, 56 with. Fix landed in `4c5370e7`, not re-run. |
| reasoning-separator | failed at 092324 | Asserts a continued conversation reuses the previous turn minus a margin. `preserve_thinking` defaults to false, so earlier thoughts are not in the continued prompt at all. |
| tool-streaming | failed at 092324 | The boundary the suite asserts belongs to a parser Kolibri does not use: argument deltas follow completed values, not the chunk grid. See Known behaviour. |

Four of the six were the harness asserting its own frame — a twelve-character
marker, a client deadline, a reasoning effort it asked for twice, an answer-shape
table — and `4c5370e7` replaced each with a preset-bound expectation while
`83d0a1dc` added the silent-call gate for dataset 68. **None of them is recorded
green here, because none has been re-run since**: a suite counts when a run says
so, not when the diff looks right.

The cache-depth boundary behind the second row is a rule, not a fault: the engine
declares the prefix the next request is expected to share, and for a prompt
ending in a text-only user turn that is the end of the *previous* turn, because
agent clients replace the trailing turn per request (`chat_template.hpp:47-51`).
`text_model_runner.cpp:714` cuts the stored checkpoint to exactly that offset,
so paying again for the previous user block is the designed cost. The standing
suspect was measured out rather than argued: `preserve_thinking` neither helps
nor hurts reuse, it only adds the generated part — at chat length a dropped
thought reused 6,466 of 6,485 and a preserved one all 6,485, a difference of 19
tokens, which is the thought block; at 4,803 tokens the same pair is 4,211
against 4,258, 47. Also excluded: `CheckpointPosition` granularity, which is
`session().position()` here exactly as in the Gemma 4 runner, and snapshot size,
the snapshot in question being 1.53 GB against a ceiling of host RAM minus
headroom.

## Known behaviour

- **A thinking budget can empty `content`.** With `reasoning_effort: high` a
  small `max_tokens` is spent entirely inside `
` and `content` comes back
  `""` with `finish_reason: length`. This is the model spending its budget, not a
  parser failure; `serve_check.py` therefore scores the reasoning channel too and
  re-asks once at `low` rather than reporting a miss it did not observe.
- **`parallel_tool_calls: true` admits a second envelope.** The constrained
  grammar allows another call when the request permits parallel calls, and the
  model uses it, so the same call can arrive twice. `parallel_tool_calls: false`
  yields exactly one. Upstream grammar behaviour, not this adapter.
- **`tool_choice: "required"` leaves the reasoning shell in `content`.** The
  `tool_only` branch of `ParseStructuredGeneration` splits reasoning off only when
  the generation begins with the closing marker, and this template surrounds the
  marker with newlines, so the returned `content` starts with a newline, the
  closing marker and two more newlines before the call text. `tool_choice: "auto"`
  is unaffected. The fix belongs in that shared splitter (whitespace around the
  marker), not in this model's adapter; until it lands, trim client-side or use
  `auto`.
- **Tool arguments stream per completed value, not per token.** At a tool marker
  `StreamingTextFilter` stops the text channel and keeps the rest of the
  generation for the parser (`openai_chat.cpp:2473`, `hidden_.append(piece)`),
  which is easy to mistake for buffering of the call. The argument channel is
  separate: `StreamingToolCalls::Update` (`openai_chat.cpp:2778`) publishes only
  the newly parsed suffix of a call, and a new suffix exists when the JSON
  parser holds a completed value (`partial_arguments`, `openai_chat.cpp:75`,
  written by `AddPartialArgument`, `openai_chat.cpp:1472`), refreshed per piece
  at `openai_chat.cpp:3287`. A call whose arguments are one long string
  therefore arrives as one or two deltas. Nothing waits for the generation to
  end; the granularity is the parser's. Progressive string prefixes would mean
  changing the shared parser and with it the wire behaviour of every marker
  model on this engine, so it is left alone here on purpose — clients get the
  call complete and parseable. `tool_streaming.py` asserts a fragment count only
  where a parser promises one: the typed call syntax of Gemma 4, while marker
  presets assert that deltas arrive at all and before the terminal event
  (`RELEASES_ARGUMENT_ON_VALUE` in `tool_streaming.py`).

## Not verified

- **Text quality.** No benchmark suite has been run against the original FP8
  model, and the conversion itself warns that quantization can reduce accuracy.
  Nothing here says how good this model is; only that the engine computes it.
- **Other quants.** Q6_K, Q8_0 and Q5_K_M have not been loaded. The binder
  accepts Q4_K, Q6_K and F32 rows and names the tensor it rejects, so an
  unexpected per-tensor layout fails loudly rather than silently — but that is
  a load-time guarantee, not a correctness measurement.
- **Depth beyond 109,656 served.** Recall has been scored cold at 39,292, 108,627
  and 109,656 prompt tokens — every planted fact at 10/50/90 % of a prompt twenty
  times the sliding window, so all forty rings sit behind the facts and only the
  ten patternless layers can carry them. The cache geometry supports 262,144;
  nothing has been scored between 109,656 and that.
- **A narrower KV cache.** The cache here is binary16. Aleph Alpha serves the
  FP8 checkpoint with an FP8 KV cache (their plugin's serve command; drop the
  flag on `Kolibri-1-BF16`) and argues in report §2.1.3 with Appendix I.4.1
  that per-head QK normalisation bounds every query and key element inside the
  FP8 range, so the cache cannot overflow. That is their checkpoint, their
  hardware and their accuracy run: nothing here has scored FP8-or-narrower KV
  on the Q4_K_M artifact, so the KV-width proposal in
  [BENCHMARKS](BENCHMARKS.md) stays a proposal, gated on the same recall probe
  (`tools/kolibri/serve_check.py recall`) that graded binary16 at depth.
