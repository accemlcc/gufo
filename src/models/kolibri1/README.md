# Kolibri 1 implementation

The Kolibri 1 runtime answers Aleph Alpha's 78B MoE (3.46B active, `kolibri1`
architecture) on `gfx1151`. Nothing here is shared with another model: the
graph is 50 layers of grouped attention and mixture, and every kernel is
written for the shapes of that one artifact.

[Usage](../../../docs/models/kolibri-1/README.md) ·
[Quality](../../../docs/models/kolibri-1/QUALITY.md)

## Source ownership

| Path | Responsibility |
|---|---|
| [`config.hpp`](config.hpp), [`config.cpp`](config.cpp) | Metadata of the `kolibri1` file: sizes, the per-layer sliding/global kind list, the 513 window, and rejection of a file this engine cannot answer. |
| [`weights.hpp`](weights.hpp), [`weights.cpp`](weights.cpp) | Tensor-name binding and per-role format/shape checks over the mapped file; no weight is copied here. |
| [`tokenizer.hpp`](tokenizer.hpp), [`tokenizer.cpp`](tokenizer.cpp) | BPE vocabulary, merges, the pre-tokenizer split and control tokens of this vocabulary. |
| [`chat_template.hpp`](chat_template.hpp), [`chat_template.cpp`](chat_template.cpp) | Renderer of the qualified chat template (`TemplateId()` names the revision); the file's template must hash to it. |
| [`reference.hpp`](reference.hpp), [`reference.cpp`](reference.cpp) | Scalar oracle: the same graph in double, weights dequantized on the fly, full K/V for every position. |
| [`engine.hpp`](engine.hpp), [`engine.cpp`](engine.cpp) | One loaded artifact and one conversation: load order, session creation, prefill chunking, and the frontier logits. |
| [`kernels/rocm/device_model.hpp`](kernels/rocm/device_model.hpp), [`device_model.cpp`](kernels/rocm/device_model.cpp) | The upload: every tensor keeps its GGUF encoding on device, plus the per-layer view the kernels are handed. |
| [`kernels/rocm/executor.hpp`](kernels/rocm/executor.hpp), [`executor.cpp`](kernels/rocm/executor.cpp) | Scratch arena, group tables, the cache allocation, and the order of a layer. |
| [`kernels/rocm/attention.hpp`](kernels/rocm/attention.hpp), [`attention.hip.cpp`](kernels/rocm/attention.hip.cpp) | WMMA attention over the binary16 cache, split-K for one-row queries, window and ring addressing. |
| [`kernels/rocm/qkv_post.hpp`](kernels/rocm/qkv_post.hpp), [`qkv_post.hip.cpp`](kernels/rocm/qkv_post.hip.cpp) | Per-head query and key RMS norms, RoPE in the sliding layers only, and the cache write. |
| [`kernels/rocm/moe.hpp`](kernels/rocm/moe.hpp) | The contract the mixture kernels are written against, including the group-table layout. |
| [`kernels/rocm/gemv.hip.cpp`](kernels/rocm/gemv.hip.cpp) | Grouped Q4_K and Q6_K projection (dense projections are the one-group case) and the embedding gather. |
| [`kernels/rocm/route.hip.cpp`](kernels/rocm/route.hip.cpp) | Router logits, selection by `top_k(logits + bias)`, weights by unbiased `sigmoid(logits)`, expert counts and the group table. |
| [`kernels/rocm/blocks.hip.cpp`](kernels/rocm/blocks.hip.cpp) | Norms, the residual add, the expert activation and the end of a mixture. |
| [`CMakeLists.txt`](CMakeLists.txt) | Explicit source registration; strict floating point is set per file, not per target. |

## Arithmetic

A row is what must not move. `gemv.hip.cpp` and `blocks.hip.cpp` compile with
`-fno-fast-math -ffp-contract=off`, so a slot's value does not depend on which
other slots share its launch and a batch boundary does not change a row. Both
files assert that on the device: the same row alone, in a full batch, under a
different group split and at a different pass width has to come out bit-identical.

The mixture is not DeepSeek's. Selection reads `logits + bias`, the weights are
the unbiased `sigmoid(logits)` of the chosen experts with no renormalisation,
and one ungated shared expert is added with weight 1. Norms sit on the branch
outputs, not on the residual.

Weights never leave their encoding: `device_model.cpp` uploads Q4_K and Q6_K
blocks as they lie in the file, and the GEMV decodes them inside the dot. A
launch that cannot serve a format or shape returns false instead of guessing,
and the executor throws rather than mixing stale activations into a residual.

## State

Ten of the fifty layers have no positional encoding and attend over every
position, so their cache is `max_context` deep. The rest are sliding with a
513 window and hold a ring of `window + rows` slots. One allocation carries both
shapes per layer; a session's cost is therefore the ten full caches plus a small
ring, not the context times fifty.

A session owns its cache and nothing else. Scratch, the group tables and the
stream belong to the executor and are shared by every session, so forwards are
serialized on one stream. `Reset()` returns a session to position zero, which is
what a client that re-renders a transcript per turn needs.

## Checks

`kolibri1_device_forward_test` is the end-to-end one: it loads the real artifact
on the device, runs a short prefix and one decode step, and compares the
frontier logits with `Reference`. The kernel tests build their own weights and
compare against an FP64 dot; `tools/kolibri/m1.sh` carries a target to the GPU
box. See [docs/TESTING.md](../../../docs/TESTING.md) for the presets.
