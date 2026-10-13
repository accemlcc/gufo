#!/usr/bin/env python3
"""One cold prompt through a served Kolibri, reporting the server's own rate.

The companion to `throughput`, which measures a forward inside the engine: only
the served path says what a prompt costs with the chat template, the scheduler
and the continuation cache in the way, and it is the number the server itself
prints (`timings.prompt_per_second`) rather than one derived from a wall clock.

Every request carries a fresh marker in front of the filler. Prefix reuse here
is byte-exact, so a prompt that repeats an earlier one measures a cache hit and
not a prefill - `--repeats 2` shows that difference on purpose: the second run
of the same bytes comes back `cached=<prompt_tokens>` in hundredths of a second.

  KOLIBRI_URL     base URL (default http://127.0.0.1:8102)
  KOLIBRI_MODEL   served model name (default "Kolibri 1", as /v1/models reports)
  --sentences     prompt length; one sentence is about thirteen tokens, so 1,000
                  of them ask for roughly 13,000
  --repeats       requests to send, each with its own marker

Serve the model first, e.g. on the box that has the artifact:

  gufo serve llm --host 127.0.0.1 --port 8102 --model <Kolibri-1-Q4_K_M.gguf> \
      --context 16384 --sessions 1 --think off --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

BASE = os.environ.get("KOLIBRI_URL", "http://127.0.0.1:8102").rstrip("/")
MODEL = os.environ.get("KOLIBRI_MODEL", "Kolibri 1")
UNIT = "The ledger line eleven records a shipment of copper wire from Leipzig. "

parser = argparse.ArgumentParser()
parser.add_argument("--sentences", type=int, default=800)
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()

filler = UNIT * args.sentences
for run in range(args.repeats):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": f"[probe {os.urandom(8).hex()}]\n{filler}"}
            ],
            # One token: the generation leg is a rounding error on the prompt,
            # and keeping it tiny is what makes two harnesses comparable.
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=3600) as reply:
        answer = json.load(reply)
    usage = answer.get("usage", {})
    timings = answer.get("timings", {})
    print(
        f"run {run}: prompt={usage.get('prompt_tokens')} "
        f"cached={usage.get('cached_tokens')} "
        f"pp={timings.get('prompt_per_second')} "
        f"tg={timings.get('generation_per_second')} "
        f"wall={time.time() - start:.2f}s"
    )
