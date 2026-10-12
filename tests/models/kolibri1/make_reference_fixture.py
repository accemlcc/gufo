#!/usr/bin/env python3
"""Write the Kolibri 1 reference fixture: a tiny random artifact plus the
expected logits of one Forward call.

The NumPy forward below implements the Kolibri 1 graph of the Hob-forge
llama.cpp patch independently of the C++ oracle in src/models/kolibri1, so the
two agree only if both got the layout, the window, the norms and the routing
right. Regenerate with:

    python3 tests/models/kolibri1/make_reference_fixture.py

Outputs, in fixtures/ next to this script:
  kolibri1_tiny.gguf   F32 artifact: 3 layers, two sliding and one global
  kolibri1_logits.bin  u32 n_tokens, u32 vocab, i32 tokens[n],
                       f32 logits[n * vocab] (little endian)
"""
import os

import numpy as np

SEED = 20261012
HIDDEN = 8
LAYERS = 3
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 32
WINDOW = 5
CONTEXT = 64
EPS = 1e-6
THETA = 10000.0
EXPERTS = 4
EXPERTS_USED = 2
EXPERT_FFN = 6
SHARED_FFN = 6
VOCAB = 7
TOKENS = [3, 5, 1, 6, 0, 4]  # longer than the window, so a key falls out

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

U32, F32, BOOL, STRING, ARRAY = 4, 6, 7, 8, 9
F32_TENSOR = 0


def build(path, metadata, tensors):
    """Serialize a GGUF v3 file: header, KV pairs, tensor infos, payload."""
    out = bytearray()

    def raw(b):
        out.extend(b)

    def pod(value, dtype):
        raw(np.array([value], dtype=dtype).tobytes())

    def text(s):
        b = s.encode("utf-8")
        pod(len(b), "<u8")
        raw(b)

    raw(b"GGUF")
    pod(3, "<u4")
    pod(len(tensors), "<u8")
    pod(len(metadata), "<u8")
    for key, (kind, value) in metadata.items():
        text(key)
        pod(kind, "<u4")
        if kind == ARRAY:
            items, item_kind = value
            pod(item_kind, "<u4")
            pod(len(items), "<u8")
            for item in items:
                if item_kind == STRING:
                    text(item)
                else:
                    pod(item, {BOOL: "<?", U32: "<u4", F32: "<f4"}[item_kind])
        elif kind == STRING:
            text(value)
        elif kind == BOOL:
            pod(1 if value else 0, "<?")
        elif kind == U32:
            pod(value, "<u4")
        else:
            pod(value, "<f4")

    offset = 0
    for name, dims, data in tensors:
        text(name)
        pod(len(dims), "<u4")
        for dimension in dims:
            pod(dimension, "<u8")
        pod(F32_TENSOR, "<u4")
        pod(offset, "<u8")
        offset += (data.nbytes + 31) // 32 * 32
    while len(out) % 32:
        raw(b"\0")
    for _, _, data in tensors:
        raw(data.tobytes())
        while len(out) % 32:
            raw(b"\0")
    with open(path, "wb") as handle:
        handle.write(out)


def main():
    rs = np.random.default_rng(SEED)

    def normal(*shape, scale=0.3):
        return (rs.standard_normal(shape) * scale).astype(np.float32)

    def ones(*shape, jitter=0.05):
        return (1.0 + rs.standard_normal(shape) * jitter).astype(np.float32)

    q_dim, kv_dim = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    tensors = [
        ("token_embd.weight", [HIDDEN, VOCAB], normal(VOCAB, HIDDEN)),
        ("output.weight", [HIDDEN, VOCAB], normal(VOCAB, HIDDEN)),
        ("output_norm.weight", [HIDDEN], ones(HIDDEN)),
    ]
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        tensors += [
            (p + "attn_norm.weight", [HIDDEN], ones(HIDDEN)),
            (p + "attn_q.weight", [HIDDEN, q_dim], normal(q_dim, HIDDEN)),
            (p + "attn_k.weight", [HIDDEN, kv_dim], normal(kv_dim, HIDDEN)),
            (p + "attn_v.weight", [HIDDEN, kv_dim], normal(kv_dim, HIDDEN)),
            (p + "attn_q_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
            (p + "attn_k_norm.weight", [HEAD_DIM], ones(HEAD_DIM)),
            (p + "attn_output.weight", [q_dim, HIDDEN], normal(HIDDEN, q_dim)),
            (p + "post_attention_norm.weight", [HIDDEN], ones(HIDDEN)),
            (p + "ffn_norm.weight", [HIDDEN], ones(HIDDEN)),
            (p + "ffn_gate_inp.weight", [HIDDEN, EXPERTS],
             normal(EXPERTS, HIDDEN, scale=0.6)),
            (p + "exp_probs_b.bias", [EXPERTS], normal(EXPERTS, scale=0.4)),
            (p + "ffn_gate_exps.weight", [HIDDEN, EXPERT_FFN, EXPERTS],
             normal(EXPERTS, EXPERT_FFN, HIDDEN)),
            (p + "ffn_up_exps.weight", [HIDDEN, EXPERT_FFN, EXPERTS],
             normal(EXPERTS, EXPERT_FFN, HIDDEN)),
            (p + "ffn_down_exps.weight", [EXPERT_FFN, HIDDEN, EXPERTS],
             normal(EXPERTS, HIDDEN, EXPERT_FFN)),
            (p + "ffn_gate_shexp.weight", [HIDDEN, SHARED_FFN],
             normal(SHARED_FFN, HIDDEN)),
            (p + "ffn_up_shexp.weight", [HIDDEN, SHARED_FFN],
             normal(SHARED_FFN, HIDDEN)),
            (p + "ffn_down_shexp.weight", [SHARED_FFN, HIDDEN],
             normal(HIDDEN, SHARED_FFN)),
            (p + "post_ffw_norm.weight", [HIDDEN], ones(HIDDEN)),
        ]
    # The third layer is global: no window and no positional encoding.
    pattern = [True, True, False]

    metadata = {
        "general.architecture": (STRING, "kolibri1"),
        "general.name": (STRING, "kolibri1-tiny"),
        "kolibri1.block_count": (U32, LAYERS),
        "kolibri1.embedding_length": (U32, HIDDEN),
        "kolibri1.context_length": (U32, CONTEXT),
        "kolibri1.attention.head_count": (U32, HEADS),
        "kolibri1.attention.head_count_kv": (U32, KV_HEADS),
        "kolibri1.attention.key_length": (U32, HEAD_DIM),
        "kolibri1.attention.value_length": (U32, HEAD_DIM),
        "kolibri1.attention.sliding_window": (U32, WINDOW),
        "kolibri1.attention.layer_norm_rms_epsilon": (F32, EPS),
        "kolibri1.attention.sliding_window_pattern": (ARRAY, (pattern, BOOL)),
        "kolibri1.rope.freq_base": (F32, THETA),
        "kolibri1.expert_count": (U32, EXPERTS),
        "kolibri1.expert_used_count": (U32, EXPERTS_USED),
        "kolibri1.expert_feed_forward_length": (U32, EXPERT_FFN),
        "kolibri1.expert_shared_count": (U32, 1),
        "kolibri1.expert_shared_feed_forward_length": (U32, SHARED_FFN),
        "kolibri1.expert_weights_norm": (BOOL, False),
        "kolibri1.expert_gating_func": (U32, 5),
    }

    os.makedirs(FIXTURES, exist_ok=True)
    build(os.path.join(FIXTURES, "kolibri1_tiny.gguf"), metadata, tensors)
    logits = forward(tensors, pattern, TOKENS)

    path = os.path.join(FIXTURES, "kolibri1_logits.bin")
    with open(path, "wb") as handle:
        handle.write(np.uint32(len(TOKENS)).tobytes())
        handle.write(np.uint32(VOCAB).tobytes())
        handle.write(np.asarray(TOKENS, dtype=np.int32).tobytes())
        handle.write(logits.astype(np.float32).tobytes())
    print(f"wrote {len(TOKENS)} x {VOCAB} logits, max "
          f"{np.abs(logits).max():.4f}, argmax "
          f"{logits.argmax(axis=1).tolist()}")


def tensor(tensors, name):
    for found, _, data in tensors:
        if found == name:
            return data.astype(np.float64)
    raise KeyError(name)


def rms(x, weight):
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x * scale * weight


def rope(x, positions):
    # x: [rows, heads, dim]; NEOX pairs (i, i + dim/2) turn by
    # position * theta^(-2i/dim).
    half = HEAD_DIM // 2
    inv = THETA ** (-2.0 * np.arange(half, dtype=np.float64) / HEAD_DIM)
    angle = positions[:, None].astype(np.float64) * inv[None, :]
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


def forward(tensors, pattern, tokens):
    n = len(tokens)
    positions = np.arange(n)
    keys = [[] for _ in range(LAYERS)]
    values = [[] for _ in range(LAYERS)]
    x = tensor(tensors, "token_embd.weight")[tokens].astype(np.float64)

    for layer in range(LAYERS):
        p = f"blk.{layer}."
        sliding = pattern[layer]
        h = rms(x, tensor(tensors, p + "attn_norm.weight"))
        q = h @ tensor(tensors, p + "attn_q.weight").T
        k = h @ tensor(tensors, p + "attn_k.weight").T
        v = (h @ tensor(tensors, p + "attn_v.weight").T).reshape(
            n, KV_HEADS, HEAD_DIM)
        q = rms(q.reshape(n, HEADS, HEAD_DIM),
                tensor(tensors, p + "attn_q_norm.weight"))
        k = rms(k.reshape(n, KV_HEADS, HEAD_DIM),
                tensor(tensors, p + "attn_k_norm.weight"))
        if sliding:
            q, k = rope(q, positions), rope(k, positions)
        keys[layer].append(k)
        values[layer].append(v)
        k_all = np.concatenate(keys[layer], axis=0)  # [position, kv, dim]
        v_all = np.concatenate(values[layer], axis=0)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        attn = np.zeros((n, HEADS, HEAD_DIM))
        for r in range(n):
            # A sliding layer holds WINDOW keys including its own query.
            lo = max(0, r + 1 - WINDOW) if sliding else 0
            for head in range(HEADS):
                kvh = head // (HEADS // KV_HEADS)
                s = q[r, head] @ k_all[lo:r + 1, kvh].T * scale
                s -= s.max()
                s = np.exp(s)
                s /= s.sum()
                attn[r, head] = s @ v_all[lo:r + 1, kvh]
        o = attn.reshape(n, HEADS * HEAD_DIM) @ \
            tensor(tensors, p + "attn_output.weight").T
        o = rms(o, tensor(tensors, p + "post_attention_norm.weight"))
        x = x + o

        h = rms(x, tensor(tensors, p + "ffn_norm.weight"))
        logits = h @ tensor(tensors, p + "ffn_gate_inp.weight").T
        bias = tensor(tensors, p + "exp_probs_b.bias")
        gate_sh = tensor(tensors, p + "ffn_gate_shexp.weight")
        up_sh = tensor(tensors, p + "ffn_up_shexp.weight")
        down_sh = tensor(tensors, p + "ffn_down_shexp.weight")
        mix = np.zeros_like(x)
        for r in range(n):
            order = np.argsort(-(logits[r] + bias),
                               kind="stable")[:EXPERTS_USED]
            for e in order:
                g = h[r] @ tensor(tensors, p + "ffn_gate_exps.weight")[e].T
                u = h[r] @ tensor(tensors, p + "ffn_up_exps.weight")[e].T
                d = (g / (1.0 + np.exp(-g)) * u) @ \
                    tensor(tensors, p + "ffn_down_exps.weight")[e].T
                # Selected on the biased logits, weighted by the raw sigmoid.
                mix[r] += (1.0 / (1.0 + np.exp(-logits[r][e]))) * d
            g = h[r] @ gate_sh.T
            u = h[r] @ up_sh.T
            mix[r] += (g / (1.0 + np.exp(-g)) * u) @ down_sh.T
        mix = rms(mix, tensor(tensors, p + "post_ffw_norm.weight"))
        x = x + mix

    hidden = rms(x, tensor(tensors, "output_norm.weight"))
    return hidden @ tensor(tensors, "output.weight").T


if __name__ == "__main__":
    main()
