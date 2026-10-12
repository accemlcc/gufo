#!/usr/bin/env python3
"""Writes a tiny but complete Kolibri 1 GGUF for cross-checking forwards.

A parity run against patched llama.cpp must not depend on the 44 GiB artifact:
both engines loading a few megabytes of F32 weights is what pins the graph, and
it keeps the run inside a normal CPU budget. The metadata mirrors the released
`Kolibri-1-Q4_K_M.gguf` key set (same keys, tiny values), so the reference
implementation takes its kolibri1 path for the right reasons:

  - `expert_gating_func = 5` (select top-k on logits + bias, weight by the
    unbiased sigmoid), `expert_weights_norm = false`, one shared expert
  - `sliding_window` plus a per-layer `sliding_window_pattern`
  - q/k head norms, no v norm, sandwich norms on the branch outputs

The weights are deterministic: every tensor comes from one seeded generator, so
two runs write byte-identical files and a mismatch cannot be blamed on input.

Usage: make_tiny_artifact.py <out.gguf>
"""

from __future__ import annotations

import argparse
import struct

GGUF_MAGIC = b"GGUF"
VERSION = 3

TYPE_UINT8, TYPE_INT8, TYPE_UINT16, TYPE_INT16 = 0, 1, 2, 3
TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32, TYPE_BOOL = 4, 5, 6, 7
TYPE_STRING, TYPE_ARRAY, TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64 = 8, 9, 10, 11, 12

F32 = 0  # ggml dtype f32

# A miniature of the released model: 3 layers so the 4:1 pattern still shows one
# global layer, 4 heads of 16 dims, 2 kv heads, and 8 routed experts so that
# selecting 3 of them exercises the router without a large table.
HIDDEN = 64
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 32
LAYERS = 3
GLOBAL_LAYER = 2  # 4 sliding then 1 global in the real model; 2 then 1 here
WINDOW = 5
VOCAB = 96
CONTEXT = 64
EXPERTS = 8
EXPERTS_USED = 3
EXPERT_FFN = 32
SHEXPERT_FFN = 32
RMS_EPS = 1e-5
ROPE_BASE = 10000.0
EOS_ID = 0
PAD_ID = 1


def string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def array(element_type: int, items: list) -> bytes:
    body = b"".join(encode_value(element_type, item) for item in items)
    return struct.pack("<I", element_type) + struct.pack("<Q", len(items)) + body


def encode_value(kind: int, value) -> bytes:
    if kind == TYPE_UINT32:
        return struct.pack("<I", value)
    if kind == TYPE_INT32:
        return struct.pack("<i", value)
    if kind == TYPE_FLOAT32:
        return struct.pack("<f", value)
    if kind == TYPE_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if kind == TYPE_STRING:
        return string(value)
    if kind == TYPE_ARRAY:
        element_type, items = value
        return array(element_type, items)
    raise ValueError(f"unsupported GGUF type {kind}")


def metadata(pairs: list[tuple[str, int, object]]) -> bytes:
    body = struct.pack("<Q", len(pairs))
    for key, kind, value in pairs:
        body += string(key) + struct.pack("<I", kind) + encode_value(kind, value)
    return body


def tensors(specs: list[tuple[str, list[int], list[float]]]) -> tuple[bytes, bytes, int]:
    """Returns (info, data, alignment). The tensor count is a header field; the
    info section is exactly one entry per tensor. Gaps between tensors are not
    padding the reader may skip - llama.cpp recomputes the expected offset and
    rejects a file where one tensor does not end exactly where the next
    begins, so only the final tensor may be followed by alignment padding."""
    alignment = 32
    info = b""
    payload = b""
    for name, dims, values in specs:
        info += string(name)
        info += struct.pack("<I", len(dims))
        for dim in dims:
            info += struct.pack("<Q", dim)
        info += struct.pack("<I", F32)
        info += struct.pack("<Q", len(payload))
        payload += struct.pack(f"<{len(values)}f", *values)
    return info, payload, alignment


class Random:
    """Small deterministic generator - no numpy dependency on the build host."""

    def __init__(self, seed: int = 0x5EED) -> None:
        self.state = seed & 0xFFFFFFFF or 1

    def uniform(self, count: int, scale: float = 0.05) -> list[float]:
        out = []
        for _ in range(count):
            # xorshift32, mapped to [-scale, scale).
            x = self.state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            self.state = x & 0xFFFFFFFF
            out.append((self.state / 0xFFFFFFFF - 0.5) * 2.0 * scale)
        return out


def build_specs(rng: Random, layers: int) -> list[tuple[str, list[int], list[float]]]:
    kv_dim = KV_HEADS * HEAD_DIM
    specs: list[tuple[str, list[int], list[float]]] = [
        # ggml stores matrices as [columns, rows]; the LM head is tied, so no
        # output.weight is written and both engines duplicate the embedding.
        ("token_embd.weight", [HIDDEN, VOCAB], rng.uniform(VOCAB * HIDDEN, 0.3)),
        ("output_norm.weight", [HIDDEN], [1.0 + 0.1 * v for v in rng.uniform(HIDDEN)]),
    ]
    for layer in range(layers):
        prefix = f"blk.{layer}"
        specs.extend(
            [
                (f"{prefix}.attn_norm.weight", [HIDDEN], _norm(rng, HIDDEN)),
                (f"{prefix}.attn_q.weight", [HIDDEN, HEADS * HEAD_DIM], rng.uniform(HIDDEN * HEADS * HEAD_DIM)),
                (f"{prefix}.attn_k.weight", [HIDDEN, kv_dim], rng.uniform(HIDDEN * kv_dim)),
                (f"{prefix}.attn_v.weight", [HIDDEN, kv_dim], rng.uniform(HIDDEN * kv_dim)),
                (f"{prefix}.attn_output.weight", [HEADS * HEAD_DIM, HIDDEN], rng.uniform(HIDDEN * HEADS * HEAD_DIM)),
                # One weight per head kind: every head of a kind shares it.
                (f"{prefix}.attn_q_norm.weight", [HEAD_DIM], _norm(rng, HEAD_DIM)),
                (f"{prefix}.attn_k_norm.weight", [HEAD_DIM], _norm(rng, HEAD_DIM)),
                (f"{prefix}.post_attention_norm.weight", [HIDDEN], _norm(rng, HIDDEN)),
                (f"{prefix}.ffn_norm.weight", [HIDDEN], _norm(rng, HIDDEN)),
                (f"{prefix}.ffn_gate_inp.weight", [HIDDEN, EXPERTS], rng.uniform(HIDDEN * EXPERTS, 0.2)),
                # Biases big enough that selection and weighting disagree, which
                # is the point of sigmoid_logit_add.
                (f"{prefix}.exp_probs_b.bias", [EXPERTS], rng.uniform(EXPERTS, 0.5)),
                (f"{prefix}.ffn_gate_exps.weight", [HIDDEN, EXPERT_FFN, EXPERTS], rng.uniform(EXPERT_FFN * HIDDEN * EXPERTS)),
                (f"{prefix}.ffn_up_exps.weight", [HIDDEN, EXPERT_FFN, EXPERTS], rng.uniform(EXPERT_FFN * HIDDEN * EXPERTS)),
                (f"{prefix}.ffn_down_exps.weight", [EXPERT_FFN, HIDDEN, EXPERTS], rng.uniform(HIDDEN * EXPERT_FFN * EXPERTS)),
                (f"{prefix}.ffn_gate_shexp.weight", [HIDDEN, SHEXPERT_FFN], rng.uniform(HIDDEN * SHEXPERT_FFN)),
                (f"{prefix}.ffn_up_shexp.weight", [HIDDEN, SHEXPERT_FFN], rng.uniform(HIDDEN * SHEXPERT_FFN)),
                (f"{prefix}.ffn_down_shexp.weight", [SHEXPERT_FFN, HIDDEN], rng.uniform(HIDDEN * SHEXPERT_FFN)),
                (f"{prefix}.post_ffw_norm.weight", [HIDDEN], _norm(rng, HIDDEN)),
            ]
        )
    return specs


def _norm(rng: Random, count: int) -> list[float]:
    """A norm vector near 1.0 - the identity-ish scale a trained model has."""
    return [1.0 + 0.05 * v for v in rng.uniform(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument(
        "--mode",
        choices=("mixed", "sliding", "global"),
        default="mixed",
        help="which attention kind the layers use; 'global' drops both the "
        "window and RoPE, 'sliding' keeps both in every layer",
    )
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument(
        "--global-layer",
        type=int,
        default=GLOBAL_LAYER,
        help="index of the single no-positional-encoding layer; an index "
        "outside the model makes every layer a windowed one",
    )
    parser.add_argument(
        "--zero",
        action="append",
        default=[],
        metavar="NEEDLE",
        help="write zeros into every tensor whose name contains NEEDLE; "
        "repeated. Ablates one half of a block so a parity difference can be "
        "attributed to attention (--zero .attn_) or to the experts "
        "(--zero exps --zero shexp).",
    )
    args = parser.parse_args()
    window = args.window

    rng = Random()
    specs = build_specs(rng, args.layers)
    for needle in args.zero:
        specs = [(name, shape, [0.0] * len(values) if needle in name else values)
                 for name, shape, values in specs]
    tensor_info, payload, alignment = tensors(specs)

    # Lengths are the GGUF string-array element type; the sliding-window pattern
    # is a bool array, true where the layer attends inside the window.
    if args.mode == "sliding":
        pattern = [True] * args.layers
    elif args.mode == "global":
        pattern = [False] * args.layers
    else:
        pattern = [layer != args.global_layer for layer in range(args.layers)]
    tokens = ["<|eos|>", "<|pad|>", "<|im_start|>", "<|/role|>", "<|im_end|>", "<think>", "</think>",
              "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"] + [
        chr(ord("a") + i % 26) * (1 + i // 26) for i in range(VOCAB - 11)
    ]
    token_types = [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4] + [1] * (VOCAB - 11)

    pairs = [
        ("general.architecture", TYPE_STRING, "kolibri1"),
        ("general.name", TYPE_STRING, "Kolibri 1 tiny"),
        ("general.quantization_version", TYPE_UINT32, 2),
        ("kolibri1.embedding_length", TYPE_UINT32, HIDDEN),
        ("kolibri1.block_count", TYPE_UINT32, args.layers),
        ("kolibri1.context_length", TYPE_UINT32, CONTEXT),
        ("kolibri1.attention.head_count", TYPE_UINT32, HEADS),
        ("kolibri1.attention.head_count_kv", TYPE_UINT32, KV_HEADS),
        ("kolibri1.attention.key_length", TYPE_UINT32, HEAD_DIM),
        ("kolibri1.attention.value_length", TYPE_UINT32, HEAD_DIM),
        ("kolibri1.attention.layer_norm_rms_epsilon", TYPE_FLOAT32, RMS_EPS),
        ("kolibri1.attention.sliding_window", TYPE_UINT32, window),
        ("kolibri1.attention.sliding_window_pattern", TYPE_ARRAY, (TYPE_BOOL, pattern)),
        ("kolibri1.rope.freq_base", TYPE_FLOAT32, ROPE_BASE),
        ("kolibri1.expert_count", TYPE_UINT32, EXPERTS),
        ("kolibri1.expert_used_count", TYPE_UINT32, EXPERTS_USED),
        ("kolibri1.expert_feed_forward_length", TYPE_UINT32, EXPERT_FFN),
        ("kolibri1.expert_shared_count", TYPE_UINT32, 1),
        ("kolibri1.expert_shared_feed_forward_length", TYPE_UINT32, SHEXPERT_FFN),
        ("kolibri1.expert_weights_norm", TYPE_BOOL, False),
        ("kolibri1.expert_gating_func", TYPE_UINT32, 5),
        ("tokenizer.ggml.model", TYPE_STRING, "gpt2"),
        ("tokenizer.ggml.pre", TYPE_STRING, "kolibri1"),
        ("tokenizer.ggml.tokens", TYPE_ARRAY, (TYPE_STRING, tokens)),
        ("tokenizer.ggml.token_type", TYPE_ARRAY, (TYPE_INT32, token_types)),
        ("tokenizer.ggml.scores", TYPE_ARRAY, (TYPE_FLOAT32, [0.0] * VOCAB)),
        ("tokenizer.ggml.merges", TYPE_ARRAY, (TYPE_STRING, [])),
        ("tokenizer.ggml.eos_token_id", TYPE_UINT32, EOS_ID),
        ("tokenizer.ggml.padding_token_id", TYPE_UINT32, PAD_ID),
        ("tokenizer.ggml.bos_token_id", TYPE_UINT32, EOS_ID),
        ("tokenizer.ggml.add_bos_token", TYPE_BOOL, False),
        ("tokenizer.ggml.add_eos_token", TYPE_BOOL, False),
    ]

    header = GGUF_MAGIC + struct.pack("<I", VERSION)
    header += struct.pack("<Q", len(specs))
    header += metadata(pairs) + tensor_info
    header += b"\0" * ((-len(header)) % alignment)

    with open(args.output, "wb") as handle:
        handle.write(header)
        handle.write(payload)
    print(f"{args.output}: {len(specs)} tensors, {len(header) + len(payload)} bytes")


if __name__ == "__main__":
    main()
