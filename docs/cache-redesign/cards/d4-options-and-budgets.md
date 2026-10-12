# D4 · Options and budgets

**Type:** decision · **Blocks:** 08, 10, 19 · **Status:** done

## Question

The new cache needs memory the current one does not reserve (committed spill
backing), and the meaning of the staging budget changes. How do existing
options map, and do we add any?

## Current options

From `src/cli/serve/serve.cpp` and `docs/SERVER.md`:

| Option | Today |
| --- | --- |
| `--cache-ram-bytes N` | 0 (default) picks an automatic snapshot budget. |
| `--cache-disk DIR` | Enables restart-safe reuse. Several processes may share a directory. |
| `--cache-disk-bytes N` | Disk budget, 8 GiB by default. |
| `--cache-disk-staging-bytes N` | 0 picks an automatic size; limits the largest checkpoint that can be written or restored. |
| (internal) | 128 checkpoint records. |

## Proposal

1. **Spill backing comes out of `--cache-ram-bytes`.** One RAM budget; the
   ledger (card 03) splits it into committed backing, private state, tails and
   metadata. No new option. Backing is committed at startup. At the measured
   ~25 GB/s page-commit rate, 8 GiB adds roughly 0.34 s to startup (estimate).
2. **`--cache-disk-staging-bytes` keeps its name; its meaning narrows.** It
   bounds the transfer buffer only and no longer limits checkpoint size. Card 12
   selects a 1 MiB default from target-disk piece measurements; it is a
   transfer-piece budget, independent of checkpoint size.
3. **`--cache-disk-bytes` keeps its meaning.** Referenced, temporary and orphan
   bytes all count against it.
4. **`--cache-disk DIR` becomes single-process.** A second process gets a clear
   ownership error at startup, not a silent fallback.
5. **The record limit stays an internal constant (128).** Private-state bytes
   already bound checkpoint density.

## Alternatives considered

- A separate `--cache-spill-bytes`: rejected, one more knob with no clear user
  decision behind it.
- Allocating spill backing lazily: rejected by the RFC because page commits
  block other HIP calls ([PR #445](https://github.com/gufo-org/gufo/pull/445)
  measured an ~8 ms decode stall).

## Also decide

- Is the automatic `--cache-ram-bytes` value still right when the memory is
  committed at startup instead of growing on demand? It becomes visible as
  resident memory immediately.
- Is the single-process directory change a breaking change for the release
  policy (`!` in the PR title)? Recommendation: yes, mark card 19 as breaking.

## Review notes

## Results (2026-10-12)

Implemented for Flash-Next, the only model the component cache serves on the
card 19 branch. Qwen 27B and DeepSeek keep the legacy cache and its option
meanings until their adapters land (cards 15–17); `--help`, `docs/SERVER.md`,
`docs/KV-CACHE.md` and `docs/CLI.md` describe both.

1. **RAM.** Card 19 already commits the budget at startup: serving allocates
   the execution slots, observes the remaining host RAM, lowers the ledger
   limits, then commits the pool. The Flash-Next provider turns nine tenths of
   the budget after 4 MiB into 64 KiB pages; the remainder holds index,
   history, catalog and queue metadata. The startup line
   (`schema=component-cache-v1 event=configured`) now also reports the
   post-slot `automatic_bytes` and `max_bytes`, as the legacy line does, and
   `disk_capacity_bytes`.
2. **Staging.** The 1 MiB default came with card 12. A staging budget smaller
   than the largest append row (1,024 bytes on Flash-Next) could
   never publish or restore: every transfer piece holds whole rows, so each
   publication failed as `persistence_incomplete`. Startup now rejects it
   with the required size. The old advice to raise staging to the checkpoint
   size for 262K Flash-Next contexts is removed: the buffer is pinned host
   memory allocated at startup, so a large value only costs RAM.
3. **Disk bytes.** The store already counted referenced, temporary and orphan
   files, but refused to open a directory above the budget, while the legacy
   store evicts to fit. A lowered `--cache-disk-bytes` now opens: serving
   reclaims orphans and temporaries, then evicts the oldest checkpoints until
   the directory fits, and logs `event=disk_evicted reason=budget`. The
   publication check also refuses while the directory is above budget; it
   previously subtracted the managed bytes from the budget unsigned, which
   would have admitted any write.
4. **Single process.** Card 10's `LOCK` file and `flock` already apply. A
   second server fails at startup with `Error loading model '...': cache
   directory ownership unavailable (another process may own it): DIR` and
   exit code 1, after loading its weights. The check stays where the store
   opens rather than before weight loading: a restarted server would
   otherwise fail immediately while its predecessor still drains accepted
   disk writes.
5. **Records.** `RetentionPolicy::kRecordLimit` stays 128.

Decisions on the open questions:

- **Automatic RAM budget:** unchanged (half of the post-slot available RAM, at
  most 32 GiB). Retention reaches the same footprint under the old cache once
  enough checkpoints accumulate; committing at startup makes it visible
  earlier, which is safer for co-resident servers that size themselves at
  their own startup. `docs/KV-CACHE.md` says so next to the #556 advice to set
  an explicit budget. Co-resident image, ASR and TTS capacity stays an open
  resource gate in [todo.md](todo.md).
- **Breaking change:** yes. PR #543 merged into `fedeizzo/cache-staging`
  without `!`; the pull request that brings the branch into `main` must carry
  `!` (recorded in the todo's final row).

Checks: `component_text_cache_test` adds startup eviction under a lowered
budget, the one-row staging floor and directory ownership through
`TextRunnerPool`; `cache_disk_test` covers opening above budget, refusing
publication until retirement, and publishing again. Each new assertion fails
on the preceding code (startup throws `cache directory exceeds disk budget`;
the staging check and the over-budget publication refusal were each removed
to confirm their assertions fail). All 57 hosted contracts and the shared
formatter pass.

A production-build smoke run (Flash-Next AR, one session, context 16384,
release `gufo serve llm`) checks the startup paths on gfx1151. The first
server published two unrelated 4,741-token prompts as 251,663,008-byte
checkpoints (503,415,538 managed bytes) and reported `capacity_bytes` and
`automatic_bytes` 16,198,375,424 and `max_bytes` 28,101,783,552. A restart
with `--cache-disk-bytes 503415537` logged `event=disk_evicted reason=budget
entries=1 managed_bytes=251707769`. The newer prompt then restored 4,736 tokens
from disk and prefilled 5, and the evicted one prefilled all 4,741. A start with
`--cache-disk-staging-bytes 1` exited 1 with `--cache-disk-staging-bytes 1 is
smaller than one 1024-byte cache row`. Model load took 21 s. These are
startup and correctness observations, not timing qualification. No request
path changes.
