#!/usr/bin/env python3
"""Compares the qwen35moe oracle frontiers against the llama-server oracle.

Inputs:
  frontiers.json  from the /completion prefix sweep (token ids, per-frontier
                  top-10 logprobs)
  driver.out      stdout of tools/qwen35moe/parity.cpp over the same ids
                  (per-frontier top-5 id:logit)

Reports per-frontier argmax agreement, the rank of the reference argmax in
the oracle list, and the top-1 logprob gap. Exits nonzero on any argmax
mismatch.
"""
import json
import math
import sys


def main():
    with open(sys.argv[1]) as f:
        expected = json.load(f)
    lines = [ln.split() for ln in open(sys.argv[2]).read().splitlines()
             if ln.strip()]
    assert len(lines) == len(expected["frontiers"]), "frontier count"
    fails = 0
    for line, frontier in zip(lines, expected["frontiers"]):
        assert int(line[0]) == frontier["prefix_len"]
        got = [(int(p.split(":")[0]), float(p.split(":")[1]))
               for p in line[1:]]
        oracle = {tid: lp for tid, lp in frontier["top"]}
        oracle_argmax = frontier["top"][0][0]
        got_argmax = got[0][0]
        rank = next((i for i, (tid, _) in enumerate(got)
                     if tid == oracle_argmax), None)
        # Logprobs vs logits differ by the log-partition function; the gap
        # of the top pair cancels it.
        top_gap = (got[0][1] - got[1][1]) - (
            oracle[frontier["top"][0][0]] - oracle[frontier["top"][1][0]])
        ok = got_argmax == oracle_argmax
        fails += not ok
        print(f"k={line[0]:>3} {'OK ' if ok else 'FAIL'} "
              f"oracle={oracle_argmax} got={got_argmax} "
              f"oracle_rank_in_got={rank} top_gap={top_gap:+.3f}")
    print("FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
