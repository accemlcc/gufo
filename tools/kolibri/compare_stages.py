#!/usr/bin/env python3
"""Per-operator parity between the oracle and the reference llama.cpp build.

Both dumps are produced by the two logits tools with tracing on:

  GUFO_KOLIBRI_STAGES=<dir>  reference_logits ...   -> <dir>/stages.json
  GUFO_LLAMA_STAGES=<file>   llama_logits    ...   -> stages.json

They share tensor names, so a divergence can be attributed to one operator
instead of to one layer. Names are printed with the layer suffix; the report
gives, per stage, the largest absolute difference over all elements and that
difference relative to the magnitude of the reference value at the same index.

Absolute differences alone are misleading here: the residual stream is
multiplied by roughly 100 per block (the logit scale measured on the real
artifact is 1e2 per layer), so a stage late in the network always shows a large
absolute gap. What identifies a bug is the relative gap at the earliest layer.

Exit status is 0 unless a stage differs by more than --fail-relative.

Usage: compare_stages.py <oracle-stages.json> <llama-stages.json>
                         [--fail-relative 1e-3] [--only NAME] [--all-layers]
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def layer_of(name: str) -> int:
    match = re.search(r"-(\d+)$", name)
    return int(match.group(1)) if match else -1


def base_of(name: str) -> str:
    return re.sub(r"-\d+$", "", name)


def compare(gold: list[float], got: list[float]) -> tuple[float, float, int]:
    """(largest absolute difference, that difference over the stage magnitude,
    index of the largest difference).

    The magnitude is the largest absolute value in the stage, not the value at
    the differing index: the residual stream has heavy tails, so an element near
    zero makes a per-element ratio meaningless (a 1e-4 relative slip on a unit
    stream reads as 1e3 next to a zero).
    """
    scale = 0.0
    max_diff = 0.0
    at = 0
    for i, (g, s) in enumerate(zip(gold, got)):
        ag = abs(g)
        if ag > scale:
            scale = ag
        diff = abs(g - s)
        if diff > max_diff:
            max_diff = diff
            at = i
    rel = max_diff / scale if scale > 0 else (float("inf") if max_diff > 0 else 0.0)
    return max_diff, rel, at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("got")
    parser.add_argument("--fail-relative", type=float, default=1e-3)
    parser.add_argument("--only", default=None, help="stage base name filter")
    parser.add_argument(
        "--profile",
        default=None,
        help="print one stage group layer by layer instead of the summary",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="print every stage instead of the worst per operator group",
    )
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as handle:
        gold = json.load(handle)["stages"]
    with open(args.got, encoding="utf-8") as handle:
        got = json.load(handle)["stages"]

    missing = sorted(set(gold) - set(got))
    extra = sorted(set(got) - set(gold))
    shared = sorted(
        (name for name in set(gold) & set(got)
              if args.only is None or args.only in base_of(name)),
        key=lambda name: (base_of(name), layer_of(name)),
    )
    if not shared:
        print("no stage in common - check the traced names", file=sys.stderr)
        return 2

    rows = []
    for name in shared:
        if len(gold[name]) != len(got[name]):
            print(
                f"SHAPE {name:26s} oracle {len(gold[name])} vs llama {len(got[name])}"
            )
            rows.append((name, float("inf"), float("inf")))
            continue
        diff, rel, _ = compare(gold[name], got[name])
        rows.append((name, diff, rel))

    if args.profile:
        print(f"{'layer':>6} {'max|diff|':>12} {'relative':>10}")
        for name, diff, rel in rows:
            if base_of(name) != args.profile:
                continue
            print(f"{layer_of(name):6d} {diff:12.4e} {rel:10.3e}")
        return 0

    if not args.all_layers:
        worst: dict[str, tuple[str, float, float]] = {}
        for name, diff, rel in rows:
            group = base_of(name)
            if group not in worst or rel > worst[group][2]:
                worst[group] = (name, diff, rel)
        shown = sorted(worst.values(), key=lambda item: -item[2])
    else:
        shown = [(name, diff, rel) for name, diff, rel in rows]

    print(f"{'stage':30s} {'max|diff|':>12s} {'relative':>10s}")
    for name, diff, rel in shown:
        print(f"{name:30s} {diff:12.4e} {rel:10.3e}")
    if missing:
        print(f"not captured by llama.cpp: {', '.join(missing)}")
    if extra:
        print(f"only in llama.cpp dump: {', '.join(extra)}")

    bad = [item for item in shown if item[2] > args.fail_relative]
    if bad:
        first = min(bad, key=lambda item: layer_of(item[0]))
        print(
            f"\n{len(bad)} stage(s) above {args.fail_relative:g}; "
            f"earliest: {first[0]} (relative {first[2]:.3e})"
        )
        return 1
    print(f"\nall {len(shown)} traced stage(s) within {args.fail_relative:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
