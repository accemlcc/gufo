#!/usr/bin/env python3
"""Compares two Kolibri 1 logits dumps produced by different engines.

Both inputs come from tools/kolibri/{reference_logits,llama_logits}: a
`prompts` list, each with the `tokens` that produced it and the full-vocabulary
`logits` of the last position. Prompts are matched by their token list, so a
dump whose prompts differ fails loudly instead of comparing rows side by side.

The verdict is reported in the units that matter for an operator port:

  max |diff|      the largest absolute error - catches a scale or norm mistake
  top-1 match     whether both engines put the same token first
  KL(ref||test)    the distribution shift a sampler would actually feel
  tail rank       how far the reference's top token moved in the other ranking

Two engines on the same weights cannot agree bit for bit: the reference sums in
double and llama.cpp in float32 with vectorized reductions, and the MoE mixes
products in a different order. On F32 weights the difference stays far below the
sampling noise, so the default threshold is generous for float32 rounding and
tight enough to catch a wrong window, a missing norm or a mis-selected expert -
each of those changes the distribution, not the last bits of it.

Usage: compare_logits.py REFERENCE.json TEST.json [--max-abs 1e-3] [--kl 1e-4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exp = [math.exp(v - peak) for v in logits]
    total = sum(exp)
    return [v / total for v in exp]


def kl(reference: list[float], test: list[float]) -> float:
    p = softmax(reference)
    q = softmax(test)
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0.0)


def rank_of(logits: list[float], token: int) -> int:
    value = logits[token]
    return sum(1 for other in logits if other > value)


def compare(ref: dict, test: dict, max_abs: float, max_kl: float) -> list[str]:
    problems: list[str] = []
    if ref["vocab_size"] != test["vocab_size"]:
        problems.append(
            f"vocab mismatch: {ref['vocab_size']} vs {test['vocab_size']}"
        )
        return problems

    by_tokens = {}
    by_rows = {}
    for prompt in test["prompts"]:
        key = tuple(prompt["tokens"])
        if key in by_tokens:
            problems.append(f"duplicate prompt in test: {key}")
        by_tokens[key] = prompt["logits"]
        by_rows[key] = prompt

    if len(by_tokens) != len(test["prompts"]):
        return problems

    for prompt in ref["prompts"]:
        key = tuple(prompt["tokens"])
        if "rows" in prompt and "rows" in by_rows[key]:
            for position, (left_row, right_row) in enumerate(
                    zip(prompt["rows"], by_rows[key]["rows"])):
                difference = max(abs(a - b) for a, b in zip(left_row, right_row))
                print(f"prompt {list(key)} position {position}: "
                      f"max|diff|={difference:.3e}")
                if difference > max_abs:
                    problems.append(
                        f"prompt {list(key)} position {position}: "
                        f"max|diff| {difference:.3e} > {max_abs:.1e}"
                    )

        if key not in by_tokens:
            problems.append(f"prompt {list(key)} missing from the test dump")
            continue
        left, right = prompt["logits"], by_tokens[key]
        if len(left) != len(right):
            problems.append(f"prompt {list(key)}: row length differs")
            continue
        for field in ("ref_rows", "test_rows"):
            pass
        worst = max(range(len(left)), key=lambda i: abs(left[i] - right[i]))
        difference = abs(left[worst] - right[worst])
        top_ref = max(range(len(left)), key=lambda i: left[i])
        top_test = max(range(len(right)), key=lambda i: right[i])
        shift = kl(left, right)
        moved = rank_of(right, top_ref)
        print(
            f"prompt {list(key)}: max|diff|={difference:.3e} "
            f"top1={'ok' if top_ref == top_test else f'MISMATCH {top_ref} vs {top_test}'} "
            f"KL={shift:.3e} rank_of_ref_top={moved}"
        )
        if difference > max_abs:
            problems.append(
                f"prompt {list(key)}: max|diff| {difference:.3e} > {max_abs:.1e} "
                f"(token {worst}: {left[worst]:.6f} vs {right[worst]:.6f})"
            )
        if shift > max_kl:
            problems.append(
                f"prompt {list(key)}: KL {shift:.3e} > {max_kl:.1e}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("test")
    parser.add_argument("--max-abs", type=float, default=1e-3)
    parser.add_argument("--kl", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.reference) as handle:
        ref = json.load(handle)
    with open(args.test) as handle:
        test = json.load(handle)

    problems = compare(ref, test, args.max_abs, args.kl)
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(
        f"agree within max|diff|={args.max_abs:.1e} and KL={args.kl:.1e} "
        f"over {len(ref['prompts'])} prompt(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
