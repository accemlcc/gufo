#!/usr/bin/env python3
"""Single-token per-layer reference on REAL qwen35moe weights.

Decodes tensors on demand (one layer at a time) and runs the transformers
graph in float32, dumping the residual stream after every trunk layer to
<outdir>/np_layer<LL>.bin. Compare against the C++ oracle's QWEN35MOE_DUMP
output to find the first divergent layer.
"""
import os
import struct
import sys

import gguf
import numpy as np

def qtype(t):
    return {
        0: None,
        1: gguf.GGMLQuantizationType.F16,
        8: gguf.GGMLQuantizationType.Q8_0,
        12: gguf.GGMLQuantizationType.Q4_K,
        13: gguf.GGMLQuantizationType.Q5_K,
        14: gguf.GGMLQuantizationType.Q6_K,
        30: gguf.GGMLQuantizationType.BF16,
    }[t]


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            head = f.read(64 << 20)
        assert head[:4] == b"GGUF"
        off = 8
        n_t = struct.unpack("<Q", head[off:off + 8])[0]
        off += 8
        n_kv = struct.unpack("<Q", head[off:off + 8])[0]
        off += 8

        def rs():
            nonlocal off
            ln = struct.unpack("<Q", head[off:off + 8])[0]
            off += 8
            s = head[off:off + ln].decode()
            off += ln
            return s

        self.meta = {}
        for _ in range(n_kv):
            k = rs()
            kind = struct.unpack("<I", head[off:off + 4])[0]
            off += 4
            if kind == 8:
                self.meta[k] = rs()
            elif kind in (2, 3, 4, 5, 10, 11, 12):
                sz = {2: 2, 3: 2, 4: 4, 5: 4, 10: 8, 11: 8, 12: 8}[kind]
                fmt = {2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q",
                       12: "d"}[kind]
                self.meta[k] = struct.unpack("<" + fmt, head[off:off + sz])[0]
                off += sz
            elif kind == 6:
                self.meta[k] = struct.unpack("<f", head[off:off + 4])[0]
                off += 4
            elif kind == 7:
                self.meta[k] = bool(head[off])
                off += 1
            elif kind == 9:
                it = struct.unpack("<I", head[off:off + 4])[0]
                off += 4
                c = struct.unpack("<Q", head[off:off + 8])[0]
                off += 8
                if it == 8:
                    self.meta[k] = [rs() for _ in range(c)]
                else:
                    sz = {4: 4, 5: 4, 2: 2, 3: 2, 10: 8, 11: 8}[it]
                    fmt = {4: "I", 5: "i", 2: "H", 3: "h", 10: "Q",
                           11: "q"}[it]
                    self.meta[k] = list(struct.unpack(
                        f"<{c}{fmt}", head[off:off + sz * c]))
                    off += sz * c
            else:
                raise ValueError(kind)
        self.tensors = {}
        for _ in range(n_t):
            name = rs()
            rank = struct.unpack("<I", head[off:off + 4])[0]
            off += 4
            dims = struct.unpack(f"<{rank}Q", head[off:off + 8 * rank])
            off += 8 * rank
            typ = struct.unpack("<I", head[off:off + 4])[0]
            off += 4
            toff = struct.unpack("<Q", head[off:off + 8])[0]
            off += 8
            self.tensors[name] = (typ, dims, toff)
        while off % 32:
            off += 1
        self.base = off
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        p = "qwen35moe."
        g = lambda s: self.meta[p + s]  # noqa: E731
        self.nl = g("block_count") - g("nextn_predict_layers")
        self.h = g("embedding_length")
        self.heads = g("attention.head_count")
        self.kv = g("attention.head_count_kv")
        self.hd = g("attention.key_length")
        self.rot = g("rope.dimension_count")
        self.theta = g("rope.freq_base")
        self.eps = g("attention.layer_norm_rms_epsilon")
        self.interval = g("full_attention_interval")
        self.ck = g("ssm.conv_kernel")
        self.ss = g("ssm.state_size")
        self.gc = g("ssm.group_count")
        self.rk = g("ssm.time_step_rank")
        self.iv = g("ssm.inner_size")
        self.ne = g("expert_count")
        self.nu = g("expert_used_count")
        self.ff = g("expert_feed_forward_length")
        self.sf = g("expert_shared_feed_forward_length")
        self.kd = self.gc * self.ss
        self.vd = self.iv
        self.qkv = 2 * self.kd + self.vd
        self.qd = self.heads * self.hd
        self.kvd = self.kv * self.hd

    def t(self, name):
        typ, dims, toff = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        if typ == 0:
            # Row-major flat layout with dims[0] fastest: reshape reversed
            # gives [rows..., cols] directly.
            return np.frombuffer(self.data, dtype="<f4",
                                 count=n, offset=self.base + toff).reshape(
                                     tuple(reversed(dims))).copy()
        rb = {8: 34, 12: 144, 13: 176, 14: 210}[typ]
        bq = {8: 32, 12: 256, 13: 256, 14: 256}[typ]
        assert dims[0] % bq == 0
        out = np.empty(n, dtype=np.float32)
        per = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        # row-major decode: rows of dims[0] over the outer block
        rows_outer = n // dims[0]
        pos = self.base + toff
        o = 0
        for _ in range(rows_outer):
            for b in range(dims[0] // bq):
                blob = self.data[pos:pos + rb]
                pos += rb
                out[o:o + bq] = np.asarray(
                    gguf.dequantize(np.frombuffer(blob, dtype=np.uint8),
                                    qtype(typ))).reshape(-1)
                o += bq
        return out.reshape(tuple(reversed(dims))).copy()

    def is_linear(self, l):
        return ((l + 1) % self.interval) != 0


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    path, tok, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    m = Model(path)
    h = m.t(f"token_embd.weight")[tok].astype(np.float64)
    S = {}
    C = {}
    K = {}
    n = 1
    for l in range(m.nl):
        p = f"blk.{l}."
        x = rms(h, m.t(p + "attn_norm.weight"), m.eps)
        if l == 3:
            x.astype("<f4").tofile("/tmp/normed3_qnpy.bin")
            x.astype("<f4").tofile("/tmp/normed3_np.bin")
        if m.is_linear(l):
            qkv = x @ m.t(p + "attn_qkv.weight").T
            z = x @ m.t(p + "attn_gate.weight").T
            # depthwise causal conv + silu (single token, state zeros first)
            key = (l,)
            if key not in C:
                C[key] = np.zeros((m.qkv, m.ck - 1))
            win = np.concatenate([C[key], qkv[None].T], 1)
            C[key] = win[:, 1:]
            wconv = m.t(p + "ssm_conv1d.weight")
            yc = (win * wconv).sum(1)
            yc = silu(yc)
            q, k, v = np.split(yc, [m.kd, 2 * m.kd])
            q = q.reshape(m.gc, m.ss)
            k = k.reshape(m.gc, m.ss)
            v = v.reshape(m.rk, m.iv // m.rk)
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
            q = q / np.sqrt(m.ss)
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
            qr = np.repeat(q, m.rk // m.gc, 0)
            kr = np.repeat(k, m.rk // m.gc, 0)
            b = 1.0 / (1.0 + np.exp(-(x @ m.t(p + "ssm_beta.weight").T)))
            a = x @ m.t(p + "ssm_alpha.weight").T
            dt = m.t(p + "ssm_dt.bias")
            A = m.t(p + "ssm_a")
            g = A * np.log1p(np.exp(a + dt))
            if key not in S:
                S[key] = np.zeros((m.rk, m.ss, m.iv // m.rk))
            St = S[key] * np.exp(g)[:, None, None]
            kv = (St * kr[:, :, None]).sum(1)
            St = St + kr[:, :, None] * (v - kv)[:, None, :] * b[:, None, None]
            S[key] = St
            o = (St * qr[:, :, None]).sum(1)
            zr = z.reshape(m.rk, m.iv // m.rk)
            o = rms(o, np.tile(m.t(p + "ssm_norm.weight"), m.rk).reshape(
                m.rk, m.iv // m.rk), m.eps) * silu(zr)
            mix = o.reshape(m.vd) @ m.t(p + "ssm_out.weight").T
        else:
            qg = x @ m.t(p + "attn_q.weight").T
            q, gate = qg[:m.qd], qg[m.qd:]
            q = q.reshape(m.heads, m.hd)
            q = rms(q, m.t(p + "attn_q_norm.weight"), m.eps)
            fr = 1.0 / (m.theta ** (2.0 * np.arange(m.rot // 2) / m.rot))
            ang = 0.0 * fr
            c, s = np.cos(ang), np.sin(ang)
            a, bq = q[..., :m.rot // 2], q[..., m.rot // 2:m.rot]
            q[..., :m.rot // 2] = a * c - bq * s
            q[..., m.rot // 2:m.rot] = a * s + bq * c
            k = (x @ m.t(p + "attn_k.weight").T).reshape(m.kv, m.hd)
            k = rms(k, m.t(p + "attn_k_norm.weight"), m.eps)
            ak, bqk = k[..., :m.rot // 2], k[..., m.rot // 2:m.rot]
            k[..., :m.rot // 2] = ak * c - bqk * s
            k[..., m.rot // 2:m.rot] = ak * s + bqk * c
            if l == 3:
                qg.astype("<f4").tofile("/tmp/qg3_np.bin")
            v = (x @ m.t(p + "attn_v.weight").T).reshape(m.kv, m.hd)
            kk = np.repeat(k, m.heads // m.kv, 0)
            vv = np.repeat(v, m.heads // m.kv, 0)
            # single token, single cached key: attention weight is 1.0 for
            # every head (softmax over positions would be degenerate here).
            o = vv
            o = o * (1.0 / (1.0 + np.exp(-gate.reshape(m.heads, m.hd))))
            if l == 3:
                v.astype("<f4").tofile("/tmp/vv3_np.bin")
                gate.astype("<f4").tofile("/tmp/gate3_np.bin")
                o.astype("<f4").tofile("/tmp/gated3_np.bin")
            mix = o.reshape(m.qd) @ m.t(p + "attn_output.weight").T
            if l == 3:
                mix.astype("<f4").tofile("/tmp/mixer3_np.bin")
        h = h + mix
        x = rms(h, m.t(p + "post_attention_norm.weight"), m.eps)
        rl = x @ m.t(p + "ffn_gate_inp.weight").T
        rl = rl - rl.max()
        pr = np.exp(rl)
        pr = pr / pr.sum()
        idx = np.argsort(-pr, kind="stable")[:m.nu]
        tw = pr[idx]
        tw = tw / tw.sum()
        moe = np.zeros_like(x)
        GE = m.t(p + "ffn_gate_exps.weight")
        UE = m.t(p + "ffn_up_exps.weight")
        DE = m.t(p + "ffn_down_exps.weight")
        for j in range(m.nu):
            e = idx[j]
            act = silu(x @ GE[e].T) * (x @ UE[e].T)
            moe = moe + tw[j] * (act @ DE[e].T)
        gs = 1.0 / (1.0 + np.exp(-(x * m.t(p + "ffn_gate_inp_shexp.weight")).sum()))
        sh = silu(x @ m.t(p + "ffn_gate_shexp.weight").T) * (
            x @ m.t(p + "ffn_up_shexp.weight").T)
        moe = moe + gs * (sh @ m.t(p + "ffn_down_shexp.weight").T)
        h = h + moe
        h.astype("<f4").tofile(os.path.join(outdir, f"layer{l:02d}.bin"))
    print("done", m.nl)


if __name__ == "__main__":
    main()
