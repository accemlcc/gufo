#!/usr/bin/env python3
"""Teacher-forced agreement: device top-k vs llama-server logprobs.

Reads the device dump (pos id:logit ...) and queries /completion once per
prefix with the exact token-id prompt, comparing the argmax and the top-pair
gap (logit gaps equal logprob gaps). Usage:
compare_teacherforced.py <dev_topk.txt> <ids.txt> <url>
"""

import json
import sys
import urllib.request


def complete(url, ids, n_probs):
    body = json.dumps(
        {
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "repeat_penalty": 1.0,
            "n_probs": n_probs,
            "cache_prompt": True,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    dev_path, ids_path, url = sys.argv[1:4]
    with open(ids_path) as f:
        ids = [int(x) for x in f.read().split()]
    dev_rows = []
    with open(dev_path) as f:
        for line in f:
            parts = line.split()
            dev_rows.append(
                [(int(p.split(":")[0]), float(p.split(":")[1])) for p in parts[1:]]
            )
    n = min(len(ids), len(dev_rows))
    agree = 0
    flips = []
    for k in range(n):
        res = complete(url, ids[: k + 1], 8)
        entry = res["completion_probabilities"][0]
        serv_top = [(t["id"], t["logprob"]) for t in entry["top_logprobs"]]
        dev_top = dev_rows[k][0][0]
        serv_ids = [t[0] for t in serv_top]
        ok = serv_top[0][0] == dev_top
        agree += ok
        mark = "OK " if ok else "FLIP"
        gap_dev = dev_rows[k][0][1] - dev_rows[k][1][1]
        gap_srv = serv_top[0][1] - serv_top[1][1]
        if not ok:
            flips.append(k + 1)
        print(
            f"pos {k + 1:3d} {mark} dev={dev_top} srv={serv_top[0][0]} "
            f"gap_dev={gap_dev:.3f} gap_srv={gap_srv:.3f}"
        )
    print(f"agreement {agree}/{n}, flips at: {flips if flips else 'none'}")


if __name__ == "__main__":
    main()
