# Kolibri-1 experiments

Retained and rejected decisions, newest first. Numbers are on `gfx1151` with
the released Q4_K_M unless stated.

## Retained

- **A depth curve needs cold legs at one build.** The served depth numbers on
  this page were two points at forward width 16, fitted as `T(n) = a*n + b*n^2`
  and then quoted at every depth beyond them. `serve_check.py sweep` now walks
  cold legs (fresh marker per prompt, so no byte-exact prefix reuse, and the
  reported `cached` is the proof) across 3.2 K to 97.9 K on the shipped width:
  `a=1.781e-3`, `b=1.835e-8`, worst leg off by 0.36 s, and a leg held back from
  the fit landing 0.10 % off. Two things came of it — `b` three times the old
  value was a two-point artifact and not a kernel change, since forward width
  cannot change the number of token pairs the patternless layers attend; and
  the width gain measured at 13 K (1.83×, 2.00×) holds at 39 K and 108 K
  (2.52×, 2.71×), where attention rather than the routed experts is the work
  ([BENCHMARKS](BENCHMARKS.md#served-depth-cold-at-128-rows)).
- **A timing helper that repeats one table measures the L2.** `gemv_test`'s TIME
  cases reported 305-348 GB/s for the six-group decode launch — 4.42 MB read
  twenty times out of a cache that holds it whole — which is the number anyone
  comparing it against the live stage's 87 GB/s would have had to explain away.
  `Timed` now prints a second line wherever one is possible, giving each round
  experts no earlier round touched: the same gate launch is 167 GB/s and its
  down-shape twin 157, on a box that also holds another model's 95 GB of GTT.
  The cold rate is what a decode step pays, because it reads an expert once and
  not again for fifty layers; the wide shapes, which already stream 70-283 MB a
  launch, print one line and did not move. A standalone probe with the device to
  itself puts the shipped split at 187 GB/s, gate-and-up-in-one-launch at 206
  with the arithmetic unchanged, and every prefetch variant below plain —
  [BENCHMARKS](BENCHMARKS.md#decodes-routed-stage-apart-from-the-memory-it-never-touches).
- **One group table over the forward, not one per sixteen rows.** The routed
  stage measured flat at 2.21 ms per token from width 16 to width 128, and that
  flatness was recorded as the grouped table doing its job. It was doing
  nothing of the sort: `kMaxGroupSlots` was 16, so `Executor::Experts` cut a
  128-row forward into eight mixtures and each read the experts of its own
  rows — traffic per step is constant in the step's width by construction, so
  a stepped stage cannot fall no matter how wide the forward gets. The bound is
  the table's, and it is 128 now. Past the sixteen rows one block of the
  selection covers, a block cannot see the slots of the rows it does not own, so
  the selection publishes ids and counts and a second launch gathers the whole
  table from them; `counts` is required whenever a table beyond `kSelectRows` is
  asked for, which is a contract the callers and the timing helper had to learn
  (the helper had no counts buffer and aborted the wide case on first touch).
  On synthetic tables at the artifact's shapes the same 768 slots cost 1.86-1.97×
  more as eight 96-group tables than as one 384-group table, at the same
  190-226 GB/s in both columns — fewer reads, not a faster kernel
  ([BENCHMARKS](BENCHMARKS.md#one-group-table-over-the-forward)).
- **Staged weight upload.** A GGUF tensor is one host row-major block. The old
  upload copied every tensor whole into one device buffer and mapped it
  column-major. The 3-D expert tensors are the largest rows in the file
  (2,359 MiB each for `ffn_gate_exps`), so the copy needed a host block of that
  size on top of 45 GiB of weights and died in `std::bad_alloc`. Now a 2-D
  tensor maps directly and a 3-D one uploads in windows of at most 5 GiB,
  writing one matrix slab per pass. Same final layout, same byte order.
- **Row-major view of the file.** GGUF stores columns, so a tensor's
  dimensions are `[columns, rows]` — `attn_q` is `[2560, 6144]` because
  `y = xᵀ · C`, not because the file is transposed. An analysis that read them
  the other way around concluded the artifact was packed transposed; it is not,
  and the binder's shapes are correct.
- **The expert half folded into the block, not the lane.** A half (gate/up)
  block of a shared expert is 40 warp-tiles (512 columns / 128). Keyed by
  `(row, warp)` with `warp < 5`, both halves would map to the same m-block and
  the second copy would overwrite the first; keyed by `(row, warp, half)` with
  `warp < 10` it is distinct. The earlier shape — each lane computing both
  halves for its own column block — re-read a partner block per row and per
  lane, six redundant reads for a shared expert, and produced zeros.
- **`kMaxRows` as the gate/up pass bound.** The launcher used to be called with
  the number of *used* slots for the gate/up half and `max_rows` for the down
  half. `used` counts experts with at least one row, and a slot is routed to
  exactly one expert, so with fewer used slots than rows the high slots were
  never filled — and the down half silently read zeros for the experts
  scheduled after them. Both passes now run the same bound and read their own
  row count from the bin table by slot. Pinned by
  `gemv_test --launch-bound`, which runs eight slots over three distinct expert
  ids.
- **Warp-aggregated expert atomics.** The down projection writes with
  `atomicAdd` because six experts contribute to one row. Per lane, this was 64
  million 128-bit atomic requests against 19 million global reads in a
  four-row launch, and it sat on the loop's dependency chain: 44% of the
  kernel's time in a launch where the experts should be pure streaming. Now the
  contribution is written to shared memory, reduced across the warp, and one
  lane per 32-column group adds. Same order of accumulation, so the oracle
  agreement is unchanged; a microbenchmark on the real table dropped from 1880
  to 381 µs per launch.
- **The logits copy runs on the launch stream.** The MoE leaves the epi stream,
  and the host copy of the logits used to run on the default stream. Under the
  captured-and-replayed launch graph that is not a dependency, so the copy could
  read before the experts finished. It now runs on the stream the graph was
  replayed on, after the event that closes the other two.
- **A chunk's logit row is numbered inside its own forward.** The session asked
  for row `done + n - 1` of a forward that only contains `n` rows, so any prompt
  longer than the prefill chunk threw `logit row outside the forward`. Short
  prompts never reached it, which is why it survived the end-to-end checks.
  `Session::Append` now asks for row `n - 1` and only takes logits on the last
  chunk; the device test prefill 37 tokens both in one append and across a chunk
  of 16 and requires the same answer.
- **Serving's prefill knob is not the forward's knob.** Served prefill measured
  218 tok/s against the engine's 251 at chunk 16, which read as `--prefill-chunk`
  being ignored. It is not: that flag is the scheduler's budget of prompt tokens
  between decode rounds and arrives at the model as `max_input_tokens`, while
  `kKolibriPrefillChunk` is the number of rows one forward carries — a 512-token
  step is 32 forwards of 16. Passing the budget through as the row count would
  silently resize the sliding ring (`next_pow2(513 + rows)`: 1,024 up to 511 rows,
  2,048 from 512) and the scratch with it. The row count stays a model constant,
  varied by measurement rather than by flag.
- **No reasoning format for a model that matches the default.** The first adapter
  set the markup for reasoning by hand and wrapped each marker in newlines,
  because the template writes the marker surrounded by newlines. The splitter in
  the serve layer looks for the marker where the generation begins, so a wrapped
  marker is never found and the closing marker leaked into `content`. The server's
  default markup is already the bare marker pair this model writes, so `Markup()`
  keeps its default and the model adds nothing. Rule: an adapter overrides only
  what actually differs.
- **Timing came from the log line, not a new API field.** Measuring pp/tg over
  HTTP first needed SSE assembly client-side, and the obvious fix — a `timings`
  object in the response — would put fork-only fields into a contract the
  community clients share. The `event=completed` line already carries
  `prompt_tokens prefill_tokens cached_tokens prefill_tps decode_tps ttft_ms
  cache_restore_ms queue_ms`, and it agrees with the client-side measurement
  (2,046 tokens: 9.367 s against `prefill_tps 218.6`; 64 tokens: 59.2 against
  `decode_tps 60.5`), so the card quotes the log and the API stays upstream.
- **Prefix reuse proven by control, not by speed.** "The second request was
  faster" cannot distinguish reuse from a warm page cache, and "a different
  conversation hit the cache" cannot distinguish a bug from a client that built
  its second conversation out of the first one's messages. The probe pair is now
  two prompts the server has never seen, same length, different bytes, so any
  cold hit is a bug (`coldpair`), and a same-size context with a shifted line
  numbering is the negative control inside the reuse probe (`prefix`).

- **Reading a cache hit as cross-talk.** Two conversations answered identically
  while the log reported `cached_tokens` equal to the full prompt, and the
  conclusion on the table was that prefix reuse matched by length. The
  conversations had been sent twice by the harness, and the second pair was
  byte-identical to the first: reuse of identical tokens is the design, and
  `cache_miss_reason=prefix_changed` with `common_prefix_tokens=13` on a request
  that differed only in one text field shows the comparison is byte-exact. What
  looked like contamination was the harness replaying itself.

## Rejected

- **Splitting attention into a scores pass and a values pass.** Proposed to cut
  per-tile LDS↔register traffic. Not measured: with the V half being 512 of the
  6,144 columns read per block, it cannot outweigh the atomics problem it was
  meant to sit next to, and the atomics turned out to be the larger cost by far.
- **Column-parallel decode.** A block per fixed column set, looping over the
  batch. Rejected on analysis before coding: it removes the padding waste for
  ragged batches, which is not what decode is missing here, and it would split
  the per-row softmax across blocks.
- **A 16-slot grouped launch.** Fewer passes over the expert table per
  token-batch, but 16 accumulator sets of 64 registers spill at 128 threads on
  this target. Eight is the largest count that keeps the accumulators in
  registers; with 6-of-384 routing, eight slots collide on an expert so rarely
  that the extra passes cost nothing measurable.
  The premise was decode-shaped, where a row's six experts almost never share a
  group. It stopped holding the moment the routed stage ran one table over a
  hundred and twenty-eight rows, where two slots per expert is the average and
  a hot one carries a row each. The check drives a group to forty-eight slots
  and demands it come out bit-identical at every pass width, which is what the
  pass-width handling is for. The rejection of *sixteen accumulators* stands;
  what is gone is the assumption that a group is short.
