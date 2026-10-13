#!/usr/bin/env python3
"""Where a continuation's cached prefix actually stops, on a served Kolibri-1.

`serve_check.py prefix` answers the easy question: does a repeated conversation
reuse at all. This probe answers the one that decides what a deep conversation
pays per turn — *how far* into the previous request reuse reaches, and which
checkpoint in the stack supplied it. Measured here (2026-10-13, 128-line block):

  block   The family's first request has no trailing user turn, so the template
          declares its whole prompt stable and the retained snapshot is keyed to a
          token stream ending in the opened assistant tag. Cold: nothing to reuse.

  turn0   The first real turn shares only the system message with that key, and the
          assistant tag differs, so it lands on a multiple of the capture interval
          (`kCacheIntervalTokens` = 2048) instead of the boundary it could have
          used. Measured 6144 of 6320.

  continue  `ChatTemplate::Render` reports `stable_prefix_offset` as the end of the
            turn before a trailing text-only user turn
            (`src/models/kolibri1/chat_template.hpp`) and `TextModelRunner`
            truncates the retained snapshot there
            (`src/cli/serve/text_model_runner.cpp`). Turn 0 left exactly that
            checkpoint, and the continuation takes it: measured 6297 of 6339, 42
            tokens re-prefilled, which is the frame around the turn. The cost per
            turn is therefore the previous turn plus the frame, not the whole
            conversation — at cache-depth's depth, 20606 - 598 = 20008.

  repeat  Byte-identical request: the whole prompt (6339 of 6339).

The thinking family sends the same shape with `reasoning_effort: low` and replays
turn 0's answer twice — once as answer only, once with `preserve_thinking: true` —
so the price of dropping an earlier think block is measured, not assumed.

Each family numbers its filler from its own offset, so one family can never reuse
the other's prefix. Exit code is non-zero only where an invariant breaks: a cold
leg that reused, a continuation that did not go deeper than turn 0, a repeat that
did not take the whole prompt, or a `preserve` that reused less than `drop`.

Usage:
  prefix_probe.py [off|thinking|all] [--lines N] [--max-tokens N]

Environment:
  KOLIBRI_URL    base URL (default http://127.0.0.1:8102)
  KOLIBRI_MODEL  served model name (default Kolibri-1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import serve_check

CACHE_INTERVAL = 2048
QUESTION = "Welche Stadt wurde genannt? Antworte mit genau einem Wort."
FOLLOWUP = "Und jetzt? Antworte mit genau einem Wort."


def ask(messages, max_tokens, reasoning_effort=None, preserve_thinking=None):
    """`serve_check.chat` plus the two fields the thinking family needs."""
    request = {"model": serve_check.MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.0}
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        request["preserve_thinking"] = preserve_thinking
    req = urllib.request.Request(
        f"{serve_check.BASE}/v1/chat/completions",
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}")
    return payload, time.monotonic() - started


class Legs:
    """Prompt/reuse rows, the naming of each boundary, and the invariant failures."""

    def __init__(self):
        self.rows = []
        self.failures = []

    def record(self, label, payload, elapsed):
        prompt = serve_check.prompt_tokens(payload)
        reused = serve_check.cached(payload)
        previous = self.rows[-1] if self.rows else None
        text, channel = serve_check.answer(payload)
        line = (f"{label:<20} prompt {prompt:>6}  cached {reused:>6}  "
                f"{elapsed:7.2f}s")
        if previous:
            # `lag` is how far the reused prefix stays under the previous request's
            # prompt; the boundary line below reports the tail of this request's own
            # prompt. They differ by the turn that was added.
            line += (f"  turn +{prompt - previous['prompt']:>4}"
                     f"  lag {previous['prompt'] - reused:>4}")
        print(line, flush=True)
        print(f"{'':<20}answer[{channel}] {text.strip()[:66]!r}", flush=True)
        self.rows.append({"label": label, "prompt": prompt, "cached": reused})
        return prompt, reused

    def boundary(self, label, prompt, reused):
        """Name the checkpoint the reused prefix came from.

        Naming is not the verdict — `verify` owns that — but a leg whose boundary
        cannot be named is where a reader has to look.
        """
        previous = self.rows[-2] if len(self.rows) > 1 else None
        tail = prompt - reused
        if len(self.rows) == 1:
            reason = "kalt: Konversation neu"
        elif reused == prompt:
            reason = "full: identischer Request, exakter Token-Treffer"
        elif reused and reused % CACHE_INTERVAL == 0:
            reason = (f"raster: nur ein {CACHE_INTERVAL}-Checkpoint, die stabil "
                      f"erklaerte Grenze des Vorgaenders traegt eine andere Rahmung")
        elif previous and reused > previous["cached"]:
            reason = ("stabilpraefix: der Checkpoint des vorherigen Turns, "
                      f"{tail} Tokens neu gefillt")
        else:
            reason = f"keine bekannte Grenze (tail {tail})"
        print(f"{'':<20}boundary {reason}", flush=True)
        return reason

    def verify(self, prefix):
        """What a boundary regression would break."""
        rows = {row["label"]: row for row in self.rows}
        block, turn0 = rows[f"{prefix}_block"], rows[f"{prefix}_turn0"]
        if block["cached"]:
            self.failures.append(f"{block['label']}: cold leg reused {block['cached']}")
        repeat = rows.get(f"{prefix}_repeat")
        if repeat and repeat["cached"] != repeat["prompt"]:
            self.failures.append(
                f"{repeat['label']}: repeat reused {repeat['cached']} of "
                f"{repeat['prompt']}")
        # Every leg after turn 0 has turn 0's stable-prefix checkpoint available, so
        # a leg that does not go deeper than turn 0 lost it.
        for row in self.rows[2:]:
            if row["cached"] <= turn0["cached"]:
                self.failures.append(
                    f"{row['label']}: reused {row['cached']} not deeper than turn0 "
                    f"{turn0['cached']}")
        drop, keep = rows.get(f"{prefix}_drop"), rows.get(f"{prefix}_preserve")
        if drop and keep:
            print(f"{'':<20}preserve_minus_drop {keep['cached'] - drop['cached']} "
                  "(der mitgenommene Think-Anteil; der Drop kostet nichts zustzlich)",
                  flush=True)
            if keep["cached"] < drop["cached"]:
                self.failures.append(f"{keep['label']} reused less than drop")


def family(name, lines, offset, effort, max_tokens, replays) -> int:
    """One conversation shape: cold block, first turn, then each replay variant.

    `replays` maps turn 0's reply message to the assistant history a continuation
    sends, so both families share the measurement code and differ only in what
    travels back.
    """
    print(f"{name} ({lines} Zeilen Systemblock, numbering {offset}, effort "
          f"{effort or 'default'})", flush=True)
    legs = Legs()
    # A nonce at the head of the block: every leg of this family carries it, so the
    # family still shares one prefix, but a second run against a warm server cannot
    # mistake yesterday's checkpoint for its own cold start.
    block = (f"Ablage {uuid.uuid4().hex[:8]}\n"
             + serve_check.build_long_prompt(lines, offset=offset, needles=False))
    block_payload, seconds = ask([{"role": "system", "content": block}], max_tokens)
    block_prompt, block_reused = legs.record(f"{name}_block", block_payload, seconds)
    legs.boundary(f"{name}_block", block_prompt, block_reused)

    turn0 = [{"role": "system", "content": block},
             {"role": "user", "content": QUESTION}]
    payload, seconds = ask(turn0, max_tokens, effort)
    prompt0, reused0 = legs.record(f"{name}_turn0", payload, seconds)
    legs.boundary(f"{name}_turn0", prompt0, reused0)

    message = payload["choices"][0]["message"]
    for label, history, preserve in replays(message):
        messages = turn0 + history + [{"role": "user", "content": FOLLOWUP}]
        variant, seconds = ask(messages, max_tokens, effort, preserve)
        prompt, reused = legs.record(f"{name}_{label}", variant, seconds)
        legs.boundary(f"{name}_{label}", prompt, reused)
    legs.verify(name)
    print(f"{name.upper()} " + ("FEHLER: " + "; ".join(legs.failures)
                                if legs.failures else "OK"), flush=True)
    return 1 if legs.failures else 0


def off_replays(message):
    """Content-only replay, sent twice: the repeat must take the whole prompt."""
    history = [{"role": "assistant", "content": message.get("content") or ""}]
    return [("continue", history, None), ("repeat", history, None)]


def thinking_replays(message):
    """Same turn replayed without and with its think block; only that field moves."""
    thought = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    if not thought:
        raise SystemExit("keine reasoning_content geliefert; die thinking-Familie "
                         "kann den Drop nicht gegen das Mitnehmen rechnen")
    return [("drop", [{"role": "assistant", "content": content}], False),
            ("preserve", [{"role": "assistant", "content": content,
                           "reasoning_content": thought}], True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("off", "thinking", "all"), default="all",
                    nargs="?")
    ap.add_argument("--lines", type=int, default=128,
                    help="system-block lines; 128 shows the boundary cheaply, "
                         "2048 is the depth of the cache-depth fixture")
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    status = 0
    if args.mode in ("off", "all"):
        status |= family("off", args.lines, 0, "none", args.max_tokens, off_replays)
    if args.mode in ("thinking", "all"):
        status |= family("thinking", args.lines, 500_000, "low", args.max_tokens,
                         thinking_replays)
    return status


if __name__ == "__main__":
    sys.exit(main())
