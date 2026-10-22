#!/usr/bin/env python3
"""Write the qwen35moe reference fixture: a tiny random F32 artifact plus the
expected logits of one Forward call.

The NumPy forward below implements the qwen35moe graph of the transformers
`modeling_qwen3_5_moe.py` independently of the C++ oracle in
src/models/qwen35moe, so the two agree only if both got the layout, the
pattern, the zero-centered norms, the DeltaNet recurrence, the packed q|gate,
the mRoPE prefix and the softmax top-k routing right. Regenerate with:

    python3 tests/models/qwen35moe/make_reference_fixture.py

Outputs, in fixtures/ next to this script:
  qwen35moe_tiny.gguf   F32 artifact: 4 trunk layers (3 linear, 1 full)
  qwen35moe_logits.bin  u32 n_tokens, u32 vocab, i32 tokens[n],
                        f32 logits[n * vocab] (little endian)
"""
import os
import struct

import numpy as np

SEED = 20261023
HIDDEN = 16
LAYERS = 4
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
ROTARY = 8
SECTIONS = [2, 1, 1, 0]
THETA = 100.0
CONTEXT = 64
EPS = 1e-6
SSM_CONV = 2
SSM_STATE = 8
SSM_GROUP = 1
SSM_RANK = 2
SSM_INNER = 16
EXPERTS = 4
EXPERTS_USED = 2
EXPERT_FFN = 6
SHARED_FFN = 6
VOCAB = 16
TOKENS = [3, 5, 1, 6, 0, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

KEY_DIM = SSM_GROUP * SSM_STATE
VALUE_DIM = SSM_INNER
QKV_DIM = 2 * KEY_DIM + VALUE_DIM
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM

def rms_norm(x, w):
    # Standard RMSNorm on file weights: the converter bakes the checkpoint's
    # (1 + w) shift in, so the stored weights sit near 1, not 0.
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(x, pos):
    half = ROTARY // 2
    f = THETA ** (-2.0 * np.arange(half) / ROTARY)
    ang = np.asarray(pos)[:, None] * f[None, :]
    c, s = np.cos(ang), np.sin(ang)
    xr = x[..., :ROTARY]
    a, b = xr[..., :half], xr[..., half:]
    y = np.empty_like(xr)
    y[..., :half] = a * c - b * s
    y[..., half:] = a * s + b * c
    out = x.copy()
    out[..., :ROTARY] = y
    return out

def main():
    rng = np.random.default_rng(SEED)
    T = {}
    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.5
    T["token_embd.weight"] = rand(VOCAB, HIDDEN)
    T["output.weight"] = rand(VOCAB, HIDDEN)
    T["output_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1
    for l in range(LAYERS):
        p = f"blk.{l}."
        full = (l + 1) % 4 == 0
        T[p + "attn_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1
        T[p + "post_attention_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1
        if full:
            T[p + "attn_q.weight"] = rand(2 * Q_DIM, HIDDEN)
            T[p + "attn_k.weight"] = rand(KV_DIM, HIDDEN)
            T[p + "attn_v.weight"] = rand(KV_DIM, HIDDEN)
            T[p + "attn_output.weight"] = rand(HIDDEN, Q_DIM)
            T[p + "attn_q_norm.weight"] = 1.0 + rand(HEAD_DIM) * 0.1
            T[p + "attn_k_norm.weight"] = 1.0 + rand(HEAD_DIM) * 0.1
        else:
            T[p + "attn_qkv.weight"] = rand(QKV_DIM, HIDDEN)
            T[p + "attn_gate.weight"] = rand(VALUE_DIM, HIDDEN)
            T[p + "ssm_a"] = -(rng.random(SSM_RANK).astype(np.float32) + 0.1)
            T[p + "ssm_conv1d.weight"] = rand(QKV_DIM, SSM_CONV)
            T[p + "ssm_dt.bias"] = rand(SSM_RANK) * 0.1
            T[p + "ssm_alpha.weight"] = rand(SSM_RANK, HIDDEN)
            T[p + "ssm_beta.weight"] = rand(SSM_RANK, HIDDEN)
            T[p + "ssm_norm.weight"] = 1.0 + rand(SSM_INNER // SSM_RANK) * 0.1
            T[p + "ssm_out.weight"] = rand(HIDDEN, VALUE_DIM)
        T[p + "ffn_gate_inp.weight"] = rand(EXPERTS, HIDDEN)
        T[p + "ffn_gate_inp_shexp.weight"] = rand(HIDDEN) * 0.1
        T[p + "ffn_gate_shexp.weight"] = rand(SHARED_FFN, HIDDEN)
        T[p + "ffn_up_shexp.weight"] = rand(SHARED_FFN, HIDDEN)
        T[p + "ffn_down_shexp.weight"] = rand(HIDDEN, SHARED_FFN)
        T[p + "ffn_gate_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
        T[p + "ffn_up_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
        T[p + "ffn_down_exps.weight"] = rand(EXPERTS, HIDDEN, EXPERT_FFN)
    l = LAYERS
    p = f"blk.{l}."
    T[p + "attn_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1
    T[p + "post_attention_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1
    T[p + "attn_q.weight"] = rand(2 * Q_DIM, HIDDEN)
    T[p + "attn_k.weight"] = rand(KV_DIM, HIDDEN)
    T[p + "attn_v.weight"] = rand(KV_DIM, HIDDEN)
    T[p + "attn_output.weight"] = rand(HIDDEN, Q_DIM)
    T[p + "attn_q_norm.weight"] = 1.0 + rand(HEAD_DIM) * 0.1
    T[p + "attn_k_norm.weight"] = 1.0 + rand(HEAD_DIM) * 0.1
    T[p + "ffn_gate_inp.weight"] = rand(EXPERTS, HIDDEN)
    T[p + "ffn_gate_inp_shexp.weight"] = rand(HIDDEN) * 0.1
    T[p + "ffn_gate_shexp.weight"] = rand(SHARED_FFN, HIDDEN)
    T[p + "ffn_up_shexp.weight"] = rand(SHARED_FFN, HIDDEN)
    T[p + "ffn_down_shexp.weight"] = rand(HIDDEN, SHARED_FFN)
    T[p + "ffn_gate_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
    T[p + "ffn_up_exps.weight"] = rand(EXPERTS, EXPERT_FFN, HIDDEN)
    T[p + "ffn_down_exps.weight"] = rand(EXPERTS, HIDDEN, EXPERT_FFN)
    T[p + "nextn.eh_proj.weight"] = rand(HIDDEN, 2 * HIDDEN)
    T[p + "nextn.enorm.weight"] = 1.0 + rand(HIDDEN) * 0.1
    T[p + "nextn.hnorm.weight"] = 1.0 + rand(HIDDEN) * 0.1
    T[p + "nextn.shared_head_norm.weight"] = 1.0 + rand(HIDDEN) * 0.1

    # --- forward ---
    h = T["token_embd.weight"][TOKENS].astype(np.float32)
    n = len(TOKENS)
    kvcache = [[] for _ in range(LAYERS + 1)]
    vcache = [[] for _ in range(LAYERS + 1)]
    ssm_s = [np.zeros((SSM_RANK, KEY_DIM, VALUE_DIM // SSM_RANK),
                      dtype=np.float32) for _ in range(LAYERS + 1)]
    ssm_c = [np.zeros((QKV_DIM, SSM_CONV - 1), dtype=np.float32)
             for _ in range(LAYERS + 1)]
    for l in range(LAYERS):
        p = f"blk.{l}."
        full = (l + 1) % 4 == 0
        x = rms_norm(h, T[p + "attn_norm.weight"])
        if full:
            qg = x @ T[p + "attn_q.weight"].T
            q, gate = qg[:, :Q_DIM], qg[:, Q_DIM:]
            q = q.reshape(n, HEADS, HEAD_DIM)
            q = rms_norm(q, T[p + "attn_q_norm.weight"])
            q = rope(q.reshape(n * HEADS, HEAD_DIM),
                     np.repeat(np.arange(n), HEADS)).reshape(
                     n, HEADS, HEAD_DIM)
            k = (x @ T[p + "attn_k.weight"].T).reshape(n, KV_HEADS, HEAD_DIM)
            k = rms_norm(k, T[p + "attn_k_norm.weight"])
            k = rope(k.reshape(n * KV_HEADS, HEAD_DIM),
                     np.repeat(np.arange(n), KV_HEADS)).reshape(
                     n, KV_HEADS, HEAD_DIM)
            v = (x @ T[p + "attn_v.weight"].T).reshape(n, KV_HEADS, HEAD_DIM)
            kvcache[l].append(k)
            vcache[l].append(v)
            K = np.concatenate(kvcache[l], 0)
            V = np.concatenate(vcache[l], 0)
            kk = np.repeat(K, HEADS // KV_HEADS, 1)
            vv = np.repeat(V, HEADS // KV_HEADS, 1)
            kkt = np.transpose(kk, (1, 0, 2))[None]
            vvt = np.transpose(vv, (1, 0, 2))[None]
            s = (q[:, :, None, :] * kkt).sum(-1) / np.sqrt(HEAD_DIM)
            # Causal: a row attends its own past only, like the
            # chunked reference and the device kernels.
            causal = np.arange(n)[None, None, :] <= np.arange(n)[:, None, None]
            s = np.where(causal, s, -1e30)
            s = s - s.max(-1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(-1, keepdims=True)
            o = (w[..., None] * vvt).sum(2)
            o = o * (1.0 / (1.0 + np.exp(-gate.reshape(n, HEADS, HEAD_DIM))))
            mix = o.reshape(n, Q_DIM) @ T[p + "attn_output.weight"].T
        else:
            qkv = (x @ T[p + "attn_qkv.weight"].T)
            # depthwise causal conv + silu
            yc = np.empty_like(qkv)
            for t in range(n):
                win = np.concatenate([ssm_c[l], qkv[t:t + 1].T], 1)
                ssm_c[l] = win[:, 1:]
                yc[t] = silu((win * T[p + "ssm_conv1d.weight"]).sum(1))
            q, k, v = np.split(
                yc, [KEY_DIM, 2 * KEY_DIM], -1)
            q = q.reshape(n, SSM_GROUP, SSM_STATE)
            k = k.reshape(n, SSM_GROUP, SSM_STATE)
            v = v.reshape(n, SSM_RANK, VALUE_DIM // SSM_RANK)
            # L2 norm with the epsilon inside the sum, then the 1/sqrt(head)
            # query scale — matches transformers l2norm plus scaling.
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
            q = q / np.sqrt(SSM_STATE)
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
            qr = np.repeat(q, SSM_RANK // SSM_GROUP, 1)
            kr = np.repeat(k, SSM_RANK // SSM_GROUP, 1)
            b = 1.0 / (1.0 + np.exp(-(x @ T[p + "ssm_beta.weight"].T)))
            a = x @ T[p + "ssm_alpha.weight"].T
            dt = T[p + "ssm_dt.bias"]
            # The file stores the negative decay rates (-A) directly.
            g = T[p + "ssm_a"] * np.log1p(np.exp(a + dt))
            z = (x @ T[p + "attn_gate.weight"].T).reshape(
                n, SSM_RANK, VALUE_DIM // SSM_RANK)
            S = ssm_s[l]
            outs = []
            for t in range(n):
                S = S * np.exp(g[t])[:, None, None]
                kv = (S * kr[t][:, :, None]).sum(1)
                S = S + kr[t][:, :, None] * (
                    v[t] - kv)[:, None, :] * b[t][:, None, None]
                outs.append((S * qr[t][:, :, None]).sum(1))
            o = np.stack(outs)
            o = rms_norm(o, np.tile(
                T[p + "ssm_norm.weight"], SSM_RANK).reshape(
                1, SSM_RANK, SSM_INNER // SSM_RANK)) * silu(z)
            mix = o.reshape(n, VALUE_DIM) @ T[p + "ssm_out.weight"].T
        h = h + mix
        x = rms_norm(h, T[p + "post_attention_norm.weight"])
        rl = x @ T[p + "ffn_gate_inp.weight"].T
        rl = rl - rl.max(-1, keepdims=True)
        pr = np.exp(rl)
        pr = pr / pr.sum(-1, keepdims=True)
        idx = np.argsort(-pr, axis=1, kind="stable")[:, :EXPERTS_USED]
        tot = np.take_along_axis(pr, idx, 1).sum(1, keepdims=True)
        wt = np.take_along_axis(pr, idx, 1) / tot
        moe = np.zeros_like(x)
        for t in range(n):
            for j in range(EXPERTS_USED):
                e = idx[t, j]
                act = silu(x[t] @ T[p + "ffn_gate_exps.weight"][e].T) * (
                    x[t] @ T[p + "ffn_up_exps.weight"][e].T)
                moe[t] += wt[t, j] * (
                    act @ T[p + "ffn_down_exps.weight"][e].T)
        gs = 1.0 / (1.0 + np.exp(-(x * T[p + "ffn_gate_inp_shexp.weight"]
                                   ).sum(-1, keepdims=True)))
        sh = silu(x @ T[p + "ffn_gate_shexp.weight"].T) * (
            x @ T[p + "ffn_up_shexp.weight"].T)
        moe = moe + gs * (sh @ T[p + "ffn_down_shexp.weight"].T)
        h = h + moe
    h = rms_norm(h, T["output_norm.weight"])
    logits = (h @ T["output.weight"].T).astype(np.float32)

    # --- write gguf ---
    os.makedirs(FIXTURES, exist_ok=True)
    meta = [
        ("general.architecture", 8, "qwen35moe"),
        ("qwen35moe.block_count", 4, LAYERS + 1),
        ("qwen35moe.nextn_predict_layers", 4, 1),
        ("qwen35moe.embedding_length", 4, HIDDEN),
        ("qwen35moe.context_length", 4, CONTEXT),
        ("qwen35moe.attention.layer_norm_rms_epsilon", 6, EPS),
        ("qwen35moe.attention.head_count", 4, HEADS),
        ("qwen35moe.attention.head_count_kv", 4, KV_HEADS),
        ("qwen35moe.attention.key_length", 4, HEAD_DIM),
        ("qwen35moe.attention.value_length", 4, HEAD_DIM),
        ("qwen35moe.rope.dimension_count", 4, ROTARY),
        ("qwen35moe.rope.freq_base", 6, float(THETA)),
        ("qwen35moe.full_attention_interval", 4, 4),
        ("qwen35moe.ssm.conv_kernel", 4, SSM_CONV),
        ("qwen35moe.ssm.state_size", 4, SSM_STATE),
        ("qwen35moe.ssm.group_count", 4, SSM_GROUP),
        ("qwen35moe.ssm.time_step_rank", 4, SSM_RANK),
        ("qwen35moe.ssm.inner_size", 4, SSM_INNER),
        ("qwen35moe.expert_count", 4, EXPERTS),
        ("qwen35moe.expert_used_count", 4, EXPERTS_USED),
        ("qwen35moe.expert_feed_forward_length", 4, EXPERT_FFN),
        ("qwen35moe.expert_shared_feed_forward_length", 4, SHARED_FFN),
    ]
    # rope sections (i32 array) and recurrent pattern (bool array)
    out = bytearray()
    out += b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(T)) + struct.pack("<Q", len(meta) + 2 + 10)

    def wstr(s):
        nonlocal out
        b = s.encode()
        out += struct.pack("<Q", len(b)) + b

    for key, kind, val in meta:
        wstr(key)
        out += struct.pack("<I", kind)
        if kind == 8:
            wstr(val)
        elif kind == 4:
            out += struct.pack("<I", val)
        elif kind == 6:
            out += struct.pack("<f", val)
    wstr("qwen35moe.rope.dimension_sections")
    out += struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", 4) + struct.pack("<4i", *SECTIONS)
    wstr("qwen35moe.attention.recurrent_layers")
    out += struct.pack("<I", 9) + struct.pack("<I", 7)
    out += struct.pack("<Q", LAYERS + 1)
    out += bytes([1, 1, 1, 0, 0])

    # Minimal tokenizer metadata so llama.cpp loads the fixture as a second
    # opinion. The tokens are synthetic; only the layout matters.
    out += struct.pack("<Q", len("tokenizer.ggml.model")) + b"tokenizer.ggml.model"
    out += struct.pack("<I", 8)
    wstr("gpt2")
    out += struct.pack("<Q", len("tokenizer.ggml.pre")) + b"tokenizer.ggml.pre"
    out += struct.pack("<I", 8)
    wstr("qwen35")
    for key, vals in (("tokenizer.ggml.tokens",
                       [f"tok{i}".encode() for i in range(VOCAB)]),
                      ("tokenizer.ggml.merges", [])):
        wstr(key)
        out += struct.pack("<I", 9) + struct.pack("<I", 8)
        out += struct.pack("<Q", len(vals))
        for v in vals:
            wstr(v.decode() if isinstance(v, bytes) else v)
    wstr("tokenizer.ggml.token_type")
    out += struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", VOCAB) + struct.pack(f"<{VOCAB}i", *([1] * VOCAB))
    wstr("tokenizer.ggml.scores")
    out += struct.pack("<I", 9) + struct.pack("<I", 6)
    out += struct.pack("<Q", VOCAB) + struct.pack(f"<{VOCAB}f", *([0.0] * VOCAB))
    for key, val in (("tokenizer.ggml.eos_token_id", VOCAB - 1),
                     ("tokenizer.ggml.bos_token_id", 0)):
        wstr(key)
        out += struct.pack("<I", 4) + struct.pack("<I", val)
    for key in ("tokenizer.ggml.add_bos_token",
                "tokenizer.ggml.add_eos_token"):
        wstr(key)
        out += struct.pack("<I", 7) + struct.pack("B", 0)

    names = sorted(T)
    off = 0
    infos = []
    for name in names:
        arr = np.ascontiguousarray(T[name], dtype=np.float32)
        # GGUF dims: columns first; expert stacks gain the expert axis.
        if name.endswith("_exps.weight"):
            f, hh, ee = arr.shape[1], arr.shape[2], arr.shape[0]
            dims = [hh, f, ee] if "down" in name else [hh, f, ee]
            if "down" in name:
                dims = [EXPERT_FFN, HIDDEN, EXPERTS]
            else:
                dims = [HIDDEN, EXPERT_FFN, EXPERTS]
            flat = arr.reshape(-1)
        else:
            flat = arr.reshape(-1)
            if arr.ndim == 2:
                dims = [arr.shape[1], arr.shape[0]]
            else:
                dims = [arr.shape[0]]
        infos.append((name, dims, flat))
        wstr(name)
        out += struct.pack("<I", len(dims))
        for dd in dims:
            out += struct.pack("<Q", dd)
        out += struct.pack("<I", 0) + struct.pack("<Q", off)
        size = flat.nbytes
        off += (size + 31) // 32 * 32
    while len(out) % 32:
        out += b"\x00"
    for _, _, flat in infos:
        out += flat.tobytes()
        while len(out) % 32:
            out += b"\x00"
    with open(os.path.join(FIXTURES, "qwen35moe_tiny.gguf"), "wb") as f:
        f.write(bytes(out))
    with open(os.path.join(FIXTURES, "qwen35moe_logits.bin"), "wb") as f:
        f.write(struct.pack("<II", n, VOCAB))
        f.write(struct.pack(f"<{n}i", *TOKENS))
        f.write(logits.astype("<f4").tobytes())
    print(f"wrote fixture: n={n} vocab={VOCAB}")

if __name__ == "__main__":
    main()
