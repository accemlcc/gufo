# Kolibri-1 experiments

Retained and rejected decisions, newest first. Numbers are on `gfx1151` with
the released Q4_K_M unless stated.

## Retained

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
