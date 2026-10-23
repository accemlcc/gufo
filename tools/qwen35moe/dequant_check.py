#!/usr/bin/env python3
"""Cross-checks tools/qwen35moe/dequant_dump.cpp rows against the canonical
gguf package decoder. Usage:
  dequant_dump <model.gguf> > cpp.txt
  dequant_check.py <model.gguf> cpp.txt
Exits nonzero on any max_abs above 2e-3 (Q8_0/Q4_K/Q5_K/Q6_K decode noise
floor between two correct decoders is ~1e-6; larger means a stride, offset
or decoder bug on one side).
"""
import struct
import sys

import gguf
import numpy as np


def read_gguf(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"GGUF"
    off = 8
    n_tensors = struct.unpack("<Q", data[off:off + 8])[0]
    off += 8
    n_kv = struct.unpack("<Q", data[off:off + 8])[0]
    off += 8

    def rs():
        nonlocal off
        (ln,) = struct.unpack("<Q", data[off:off + 8])
        off += 8
        s = data[off:off + ln].decode()
        off += ln
        return s

    for _ in range(n_kv):
        rs()
        kind = struct.unpack("<I", data[off:off + 4])[0]
        off += 4
        if kind == 8:
            rs()
        elif kind == 9:
            it = struct.unpack("<I", data[off:off + 4])[0]
            off += 4
            (c,) = struct.unpack("<Q", data[off:off + 8])
            off += 8
            if it == 8:
                for _ in range(c):
                    rs()
            else:
                off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[it] * c
        else:
            off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}[kind]
    tensors = {}
    for _ in range(n_tensors):
        name = rs()
        (rank,) = struct.unpack("<I", data[off:off + 4])
        off += 4
        dims = struct.unpack(f"<{rank}Q", data[off:off + 8 * rank])
        off += 8 * rank
        (typ,) = struct.unpack("<I", data[off:off + 4])
        off += 4
        (toff,) = struct.unpack("<Q", data[off:off + 8])
        off += 8
        tensors[name] = (typ, dims, toff)
    while off % 32:
        off += 1
    base = off
    return data, tensors, base


QTYPES = {8: gguf.GGMLQuantizationType.Q8_0,
          12: gguf.GGMLQuantizationType.Q4_K,
          13: gguf.GGMLQuantizationType.Q5_K,
          14: gguf.GGMLQuantizationType.Q6_K}


def main():
    path, cpp_path = sys.argv[1], sys.argv[2]
    data, tensors, base = read_gguf(path)
    blocks = {}
    for line in open(cpp_path):
        line = line.strip()
        if not line:
            continue
        if line[0].isalpha():
            parts = line.split()
            assert len(parts) == 8, line[:80]
            name = parts[0]
            row = int(parts[1].split("=")[1])
            ex = int(parts[2].split("=")[1])
            cols = int(parts[3].split("=")[1])
            blocks[f"{name}@{ex}"] = (name, row, ex, cols, [])
        else:
            name = list(blocks.keys())[-1]
            blocks[name][4].extend(float(x) for x in line.split())
    fails = 0
    for key, (name, row, ex, cols, cpp_vals) in blocks.items():
        typ, dims, toff = tensors[name]
        assert typ in QTYPES, f"{name} has unexpected type {typ}"
        if typ == 8:
            rb, bq = 34, 32
        elif typ == 12:
            rb, bq = 144, 256
        elif typ == 13:
            rb, bq = 176, 256
        else:
            rb, bq = 210, 256
        assert cols % bq == 0
        if len(dims) == 3:
            expert_bytes = (dims[0] * dims[1] // bq) * rb
            assert ex < dims[2]
        else:
            expert_bytes, ex = 0, 0
        start = base + toff + ex * expert_bytes + row * (cols // bq) * rb
        vals = []
        for b in range(cols // bq):
            blob = data[start + b * rb:start + (b + 1) * rb]
            arr = np.frombuffer(blob, dtype=np.uint8)
            vals.extend(gguf.dequantize(arr, QTYPES[typ]).tolist())
        cpp = np.array(cpp_vals, dtype=np.float64)
        ref = np.array(vals, dtype=np.float64)
        assert cpp.shape == ref.shape, (name, cpp.shape, ref.shape)
        diff = np.abs(cpp - ref)
        ok = diff.max() < 2e-3
        fails += not ok
        print(f"{name} row={row} ex={ex}: max_abs={diff.max():.2e} "
              f"{'OK' if ok else 'FAIL'}")
    print("FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
