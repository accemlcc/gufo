#!/usr/bin/env python3
"""Long-context, multi-session and prefix-reuse checks against a served Kolibri-1.

The engine proves pp/tg offline; what only the served path can show is

  recall   a prompt deeper than the sliding rings still answers from the middle
           of the context (needles at 10/50/90% depth),
  sessions interleaving two conversations never lets one read the other's state,
  prefix   a repeated conversation reuses the cached prefix, proven by the
           server's own `usage.cached_tokens` and by wall-clock, with a
           different-prefix control so a warm cache cannot masquerade as reuse.

All three run over HTTP. The server is neither modified nor instrumented.

Usage:
  serve_check.py recall   [--tokens N]
  serve_check.py sessions
  serve_check.py prefix   [--tokens N]
  serve_check.py all

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

BASE = os.environ.get("KOLIBRI_URL", "http://127.0.0.1:8102").rstrip("/")
MODEL = os.environ.get("KOLIBRI_MODEL", "Kolibri-1")

# Facts are distinctive triples, so a plausible-but-wrong answer cannot collide.
NEEDLES = [
    ("Merkur", "4417", "Frauke Ilgen"),
    ("Venus", "8802", "Bertram Seitz"),
    ("Neptun", "2319", "Ida Reinhold"),
]


def chat(messages, max_tokens=64, temperature=0.0, timeout=7200,
         reasoning_effort=None):
    request = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # The server names the reason (context, size, schema); guessing the
        # filler size from a bare status code wastes the whole run.
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code}: {detail[:400]}") from None
    return payload, time.monotonic() - started


def content(payload):
    return payload["choices"][0]["message"].get("content") or ""


def answer(payload):
    """The reply plus the channel it arrived in.

    A reasoning model can spend the whole budget thinking, which leaves
    `content` empty. Scoring only the content channel would report that as a
    miss; scoring only "some text" would hide it. So the channel is returned and
    printed.
    """
    msg = payload["choices"][0]["message"]
    text = msg.get("content") or ""
    if text.strip():
        return text, "content"
    return msg.get("reasoning_content") or "", "reasoning"


def cached(payload):
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens", usage.get("cached_tokens", 0)))


def prompt_tokens(payload):
    return int((payload.get("usage") or {}).get("prompt_tokens", 0))


def filler(line_no: int) -> str:
    return (
        f"Zeile {line_no:05d}: Aktenzeichen {line_no % 9973:04d}-"
        f"{line_no % 37:02d} wurde am {line_no % 28 + 1:02d}. Tag gesichert "
        f"und von der Pruefstelle {line_no % 61:02d} abgenommen. "
        f"Die Ablage trug das Kennwort {line_no % 991:03d}.\n"
    )


def build_long_prompt(lines: int, offset: int = 0) -> str:
    """Filler with one needle per entry at 10/50/90% depth.

    `offset` shifts the numbering, which gives a same-sized context whose prefix
    differs from the first token — the control the reuse probe needs.
    """
    marks = {int(lines * f): i for i, f in enumerate((0.10, 0.50, 0.90))}
    parts = []
    for i in range(1, lines + 1):
        hit = marks.get(i)
        if hit is not None:
            name, code, person = NEEDLES[hit]
            parts.append(
                f"Wichtiger Vermerk {hit + 1}: Der Projektcode fuer {name} "
                f"lautet {code}. Die zustaendige Person ist {person}.\n"
            )
        parts.append(filler(i + offset))
    return "".join(parts)


def prompt_with_needles(target_tokens: int, offset: int = 0):
    """Size the filler from one small calibration, never from a guess.

    The rings are a token distance, so the probe has to report how deep it
    actually went; and a 100k-token calibration run would cost minutes per try.
    """
    probe = build_long_prompt(200)
    payload, _ = chat([{"role": "system", "content": probe}], max_tokens=1)
    got = prompt_tokens(payload)
    if not got:
        raise SystemExit("server meldet keine prompt_tokens")
    per_line = got / 200.0
    lines = max(64, int(target_tokens / per_line))
    print(f"kalibrierung: {got} tokens / 200 zeilen = {per_line:.1f}; "
          f"gewaehlt {lines} zeilen")
    return build_long_prompt(lines, offset), lines


def score_needles(text: str) -> list:
    failed = []
    for name, code, person in NEEDLES:
        ok_code, ok_person = code in text, person.split()[0] in text
        print(f"  {name}: code={'ja' if ok_code else 'NEIN'} "
              f"person={'ja' if ok_person else 'NEIN'}")
        if not (ok_code and ok_person):
            failed.append(name)
    return failed


def cmd_recall(target_tokens: int) -> int:
    context, _ = prompt_with_needles(target_tokens)
    question = (
        "Beantworte ausschliesslich aus dem Kontext. Nenne fuer jedes der drei "
        "Projekte den Code und die zustaendige Person, je eine Zeile im Format "
        "NAME | CODE | PERSON.\nProjekte: Merkur, Venus, Neptun.\n"
    )
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]
    payload, elapsed = chat(messages, max_tokens=512)
    text, where = answer(payload)
    depth = prompt_tokens(payload)
    print(f"tiefe: {depth} prompt_tokens; {where} nach {elapsed:.1f}s: "
          f"{text.strip()[:400]}")
    failed = score_needles(text)
    if failed and where == "reasoning":
        # The thinking channel answered; ask once more with reasoning off so the
        # content channel is tested too, without hiding the miss above.
        print("zweiter Versuch ohne Reasoning")
        payload, elapsed = chat(messages, max_tokens=256, reasoning_effort="low")
        text2 = content(payload)
        print(f"content nach {elapsed:.1f}s: {text2.strip()[:400]}")
        if score_needles(text2):
            print("HINWEIS: mit Reasoning fehlten " + ",".join(failed))
    # The middle needle only exercises the rings if the prompt outgrows them.
    if depth < 4096:
        print(f"WARNUNG: Kontext zu flach ({depth} tokens) fuer einen Ringtest")
        failed.append("depth")
    print("RECALL " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def cmd_sessions() -> int:
    """Two conversations, strictly interleaved, each stateless over HTTP.

    A session-state bug shows up as an answer quoting the other conversation's
    code or person, or as an answer that only makes sense for the other prompt.
    """
    a_name, a_code, a_person = NEEDLES[0]
    b_name, b_code, b_person = NEEDLES[1]
    seed = ("Merke dir dauerhaft: {name} hat Code {code} und verantwortlich "
            "ist {person}. Antworte knapp.")
    ask = ("Welchen Code und welche Person hast du dir fuer {name} gemerkt? "
           "Nur diese eine Zeile im Format NAME | CODE | PERSON.")

    hist_a = [{"role": "user",
               "content": seed.format(name=a_name, code=a_code,
                                      person=a_person)}]
    hist_b = [{"role": "user",
               "content": seed.format(name=b_name, code=b_code,
                                      person=b_person)}]
    for label, hist, name in (("A1", hist_a, a_name), ("B1", hist_b, b_name)):
        payload, elapsed = chat(hist, max_tokens=256)
        text, where = answer(payload)
        print(f"{label} ({name}) {elapsed:.2f}s [{where}] "
              f"cached={cached(payload)}: {text.strip()[:120]}", flush=True)
        hist.append(payload["choices"][0]["message"])

    probes = (
        ("A2", hist_a, a_name, a_code, b_code),
        ("B2", hist_b, b_name, b_code, a_code),
        ("A3", hist_a, a_name, a_code, b_code),
    )
    failed = []
    for label, hist, name, want, forbidden in probes:
        q = {"role": "user", "content": ask.format(name=name)}
        payload, elapsed = chat(hist + [q], max_tokens=256)
        msg = payload["choices"][0]["message"]
        text, where = answer(payload)
        print(f"{label} ({name}) {elapsed:.2f}s [{where}]: "
              f"{text.strip()[:160]}", flush=True)
        hist.append(q)
        hist.append(msg)
        if want not in text:
            print(f"  FEHLER: {want} fehlt")
            failed.append(label)
        if forbidden in text:
            print(f"  FEHLER: Zustand der anderen Session ({forbidden})")
            failed.append(label)
    print("SESSIONS " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def cmd_prefix(target_tokens: int) -> int:
    """One unseen conversation cold, the same conversation warm, then a control
    of the same size that was never sent.

    The cold leg has to be a prompt this server has not seen: reusing the
    filler the recall probe already paid for measures a warm prefix and calls it
    cold. The control answers the only alternative that matters — a cache
    matching on length would take the second conversation from the first.
    """
    summarize = "Fasse in einem Satz zusammen, was diese Ablage enthaelt."
    context, _ = prompt_with_needles(target_tokens, offset=7_000_000)
    first = [{"role": "system", "content": context},
             {"role": "user", "content": summarize}]
    cold, t_cold = chat(first, max_tokens=32)
    depth = prompt_tokens(cold)
    warm = first + [cold["choices"][0]["message"],
                    {"role": "user",
                     "content": "Und welches Kennwort taucht am haeufigsten auf?"}]
    warm_p, t_warm = chat(warm, max_tokens=32)
    other_ctx, _ = prompt_with_needles(target_tokens, offset=10_000_000)
    other_p, t_other = chat([{"role": "system", "content": other_ctx},
                             {"role": "user", "content": summarize}],
                            max_tokens=32)

    print(f"tiefe  {depth} prompt_tokens")
    print(f"kalt   {t_cold:6.2f}s cached={cached(cold)}")
    print(f"warm   {t_warm:6.2f}s cached={cached(warm_p)}")
    print(f"fremd  {t_other:6.2f}s cached={cached(other_p)}")
    failed = []
    if cached(cold) > 0.05 * depth:
        print(f"FEHLER: nie gesehener Prafix traf den Cache "
              f"({cached(cold)} von {depth})")
        failed.append("cold-hit")
    if cached(warm_p) < 0.9 * depth:
        print(f"FEHLER: warme Wiederverwendung zu klein "
              f"({cached(warm_p)} von {depth})")
        failed.append("warm-miss")
    if t_warm >= t_cold:
        print("FEHLER: warm war nicht schneller als kalt")
        failed.append("warm-time")
    if cached(other_p) > 0.05 * depth:
        print("FEHLER: fremder Prafix traf den Cache")
        failed.append("cold-cache")
    print("PREFIX " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def cmd_coldpair() -> int:
    """Two prompts of the same length that the server has never seen.

    A nonce puts the divergence in the first sentence, so any `cached_tokens`
    beyond the shared template is prefix matching by length rather than by
    bytes. Repeating both then proves reuse still works when it should.
    """
    nonce = uuid.uuid4().hex[:8]
    ask = "Antworte mit genau einer Zeile: NAME | CODE | PERSON."
    a = (f"Merk dir: Projekt ALPHA-{nonce} hat Code 4417 und zustaendig ist "
         f"Ilgen. {ask}")
    b = (f"Merk dir: Projekt BETA-{nonce} hat Code 8802 und zustaendig ist "
         f"Seitz. {ask}")
    pa, _ = chat([{"role": "user", "content": a}], max_tokens=64)
    pb, _ = chat([{"role": "user", "content": b}], max_tokens=64)
    a_len, b_len = prompt_tokens(pa), prompt_tokens(pb)
    text_a, where_a = answer(pa)
    text_b, where_b = answer(pb)
    print(f"A kalt: {a_len} tokens, cached={cached(pa)}, {where_a}: "
          f"{text_a.strip()[:120]}")
    print(f"B kalt: {b_len} tokens, cached={cached(pb)}, {where_b}: "
          f"{text_b.strip()[:120]}")

    pa2, _ = chat([{"role": "user", "content": a}], max_tokens=64)
    pb2, _ = chat([{"role": "user", "content": b}], max_tokens=64)
    text_a2, _ = answer(pa2)
    text_b2, _ = answer(pb2)
    print(f"A warm: cached={cached(pa2)} von {a_len}: {text_a2.strip()[:120]}")
    print(f"B warm: cached={cached(pb2)} von {b_len}: {text_b2.strip()[:120]}")

    failed = []
    if a_len != b_len:
        print(f"HINWEIS: ungleiche Laengen ({a_len} vs {b_len}) — "
              f"Laengengleichheit ist hier nur ein Hinweis, kein Beweis")
    if cached(pa) > 4 or cached(pb) > 4:
        print("FEHLER: nie gesehener Prompt traf den Cache — Prafix nach Laenge")
        failed.append("cold-hit")
    for label, text, want, other in (("A", text_a2, "4417", "8802"),
                                     ("B", text_b2, "8802", "4417")):
        if want not in text:
            print(f"FEHLER: {label} nennt den eigenen Code nicht")
            failed.append(f"{label}-answer")
        if other in text:
            print(f"FEHLER: {label} antwortet aus der anderen Konversation")
            failed.append(f"{label}-leak")
    if cached(pa2) < a_len - 4 or cached(pb2) < b_len - 4:
        print("FEHLER: echte Wiederholung wurde nicht wiederverwendet")
        failed.append("warm-miss")
    print("COLDPAIR " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    choices=("recall", "sessions", "prefix", "coldpair", "all"))
    ap.add_argument("--tokens", type=int, default=110_000,
                    help="approximate prompt size for recall/prefix probes")
    args = ap.parse_args()
    rc = 0
    if args.mode in ("recall", "all"):
        rc |= cmd_recall(args.tokens)
    if args.mode in ("sessions", "all"):
        rc |= cmd_sessions()
    if args.mode in ("prefix", "all"):
        rc |= cmd_prefix(args.tokens)
    if args.mode in ("coldpair", "all"):
        rc |= cmd_coldpair()
    return rc


if __name__ == "__main__":
    sys.exit(main())
