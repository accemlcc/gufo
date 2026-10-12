#!/usr/bin/env python3
"""Print the GGUF metadata of an artifact, filtered to the keys that decide the
graph. Reading the KV section needs only the first few megabytes, so this works
on a 47 GB file without loading it.

Usage: gguf_meta.py <model.gguf> [substring ...]
"""

from __future__ import annotations

import struct
import sys

SCALARS = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("?", 1),   # bool
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}


def main() -> int:
    path = sys.argv[1]
    needles = [arg.lower() for arg in sys.argv[2:]]
    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            print("not a GGUF file", file=sys.stderr)
            return 1
        version = struct.unpack("<I", handle.read(4))[0]
        n_tensors = struct.unpack("<Q", handle.read(8))[0]
        n_kv = struct.unpack("<Q", handle.read(8))[0]
        print(f"version {version}, tensors {n_tensors}, kv {n_kv}")

        def read_string() -> str:
            length = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(length).decode("utf-8", "replace")

        for _ in range(n_kv):
            key = read_string()
            kind = struct.unpack("<I", handle.read(4))[0]
            if kind == 8:  # string
                value: object = read_string()
            elif kind == 9:  # array
                item = struct.unpack("<I", handle.read(4))[0]
                count = struct.unpack("<Q", handle.read(8))[0]
                if item == 8:
                    value = [read_string() for _ in range(count)]
                    if len(value) > 12 and len(set(value)) <= 6:
                        # Collapse the per-layer window pattern into a summary.
                        from collections import Counter

                        value = f"{dict(Counter(value))} (n={count})"
                else:
                    fmt, size = SCALARS[item]
                    raw = handle.read(size * count)
                    value = list(struct.unpack(f"<{count}{fmt}", raw))[:12]
                    if count > 12:
                        value = f"{value} ... (n={count})"
            else:
                fmt, size = SCALARS[kind]
                value = struct.unpack(f"<{fmt}", handle.read(size))[0]
            if not needles or any(needle in key.lower() for needle in needles):
                print(f"{key} = {value}")

        if "--tensors" in sys.argv:
            # Tensor info records follow the KV section: name, n_dims, dims,
            # type, offset. Data starts at an aligned offset after all of them,
            # so this stays a header read.
            types = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
                     8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
                     13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 20: "IQ4_NL", 21: "IQ3_S",
                     23: "IQ4_XS", 30: "BF16"}
            print("--- tensors")
            counts: dict[str, int] = {}
            shown = 0
            for _ in range(n_tensors):
                name = read_string()
                rank = struct.unpack("<I", handle.read(4))[0]
                dims = [struct.unpack("<Q", handle.read(8))[0] for _ in range(rank)]
                kind = struct.unpack("<I", handle.read(4))[0]
                struct.unpack("<Q", handle.read(8))[0]
                label = types.get(kind, f"?{kind}")
                counts[label] = counts.get(label, 0) + 1
                if shown < 24 and ("blk.0." in name or "output" in name or "token_embd" in name):
                    print(f"  {name} {label} {dims}")
                    shown += 1
            print("  counts by type:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
