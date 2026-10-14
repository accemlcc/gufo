#!/usr/bin/env python3
"""Long-context, multi-session and prefix-reuse checks against a served Kolibri-1.

The engine proves pp/tg offline; what only the served path can show is

  recall   a prompt deeper than the sliding rings still answers from the middle
           of the context (needles at 10/50/90% depth),
  sessions interleaving two conversations never lets one read the other's state,
  prefix   a repeated conversation reuses the cached prefix, proven by the
           server's own `usage.cached_tokens` and by wall-clock, with a
           different-prefix control so a warm cache cannot masquerade as reuse,
  sweep    served pp and tg against context depth, every leg cold, so the
           falling decode rate with depth is a curve and not two legs.

All four run over HTTP. The server is neither modified nor instrumented.

Usage:
  serve_check.py recall   [--tokens N]
  serve_check.py sessions
  serve_check.py prefix   [--tokens N]
  serve_check.py sweep    [--depths 1024,4096,...] [--gen N]
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


# The server's own accounting, for the run where a client-side number and the
# answer disagree: only the log says which tier served a request and how many
# tokens it really prefilled. Pass --log with the path the server writes to.
LOG_PATH = ""


def log_mark():
    try:
        with open(LOG_PATH, "r", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


KEEP = ("request", "duration_ms", "prompt_tokens", "prefill_tokens",
        "cached_tokens", "common_prefix_tokens", "cache", "cache_miss_reason",
        "cache_restore_ms", "prefill_tps", "decode_tps")


def log_since(mark):
    """The server's completed-request lines after `mark`, as field dicts."""
    if not LOG_PATH:
        return []
    entries = []
    try:
        with open(LOG_PATH, "r", errors="replace") as handle:
            for _ in range(mark):
                next(handle, None)
            for line in handle:
                if "event=completed" not in line:
                    continue
                entries.append(dict(
                    pair.split("=", 1)
                    for pair in line.split()
                    if "=" in pair
                ))
    except OSError:
        return []
    return entries


def show_log(entries):
    for entry in entries:
        print("      log " + " ".join(
            f"{key}={entry[key]}" for key in KEEP if key in entry))


def check_accounting(entries, label):
    """Assert `prefill_tokens + cached_tokens == prompt_tokens`.

    The client only sees `usage.cached_tokens`, and on its own that number can
    say "everything was reused" while the request visibly prefilled for three
    minutes. Which of the two it was decides whether a probe found a cache bug
    or an accounting bug, so the invariant is checked where both numbers are
    written down together.
    """
    bad = []
    for entry in entries:
        try:
            prompt = int(entry["prompt_tokens"])
            prefill = int(entry["prefill_tokens"])
            hit = int(entry["cached_tokens"])
        except (KeyError, ValueError):
            continue
        if prompt and prefill + hit != prompt:
            bad.append(f"{entry.get('request', '?')}: prefill={prefill}"
                       f" cached={hit} prompt={prompt}")
    if bad:
        print(f"  FEHLER: {label} — Abrechnung passt nicht: " + "; ".join(bad))
    return bad

def log_int(entry, key):
    try:
        return int(entry[key])
    except (KeyError, TypeError, ValueError):
        return None


def check_reuse(label, client_cached, server_prefill, depth, want):
    """Judge reuse by the work that was done, not by one reported field.

    `want="cold"`: the prompt was never seen, so full prefill is the correct
    outcome. `want="warm"`: the opposite. With `--log` the server's own
    `prefill_tokens` decides. Without it the client field is all there is and a
    hit stays an error — that is what caught the 40 K control in the first place.
    When the two disagree, the run prints an accounting complaint and keeps its
    verdict with the work, because a request that took three minutes did not
    reuse anything no matter what the usage object claims.
    """
    if server_prefill is None:
        hit = client_cached > 0.05 * depth
        if want == "cold" and hit:
            print(f"  FEHLER: {label}: nie gesehen, usage.cached="
                  f"{client_cached} von {depth}")
            return [f"{label}-hit"]
        if want == "warm" and not hit:
            print(f"  FEHLER: {label}: keine Wiederverwendung "
                  f"(usage.cached={client_cached} von {depth})")
            return [f"{label}-miss"]
        return []
    if want == "cold":
        if server_prefill < 0.5 * depth:
            print(f"  FEHLER: {label}: fremder Praefix wurde wirklich "
                  f"uebersprungen (prefill={server_prefill} von {depth})")
            return [f"{label}-hit"]
        if client_cached > 0.05 * depth:
            print(f"  WARN: {label}: usage.cached={client_cached} meldt "
                  f"Treffer, der Server lief {server_prefill} Tokens — "
                  "usage-Feld beschreibt etwas anderes als die Arbeit")
        return []
    if server_prefill > 0.1 * depth:
        print(f"  FEHLER: {label}: kein uebersprungener Praefix "
              f"(prefill={server_prefill} von {depth})")
        return [f"{label}-miss"]
    return []

def filler(line_no: int) -> str:
    return (
        f"Zeile {line_no:05d}: Aktenzeichen {line_no % 9973:04d}-"
        f"{line_no % 37:02d} wurde am {line_no % 28 + 1:02d}. Tag gesichert "
        f"und von der Pruefstelle {line_no % 61:02d} abgenommen. "
        f"Die Ablage trug das Kennwort {line_no % 991:03d}.\n"
    )


def build_long_prompt(lines: int, offset: int = 0, needles: bool = True) -> str:
    """Filler with one needle per entry at 10/50/90% depth.

    `offset` shifts the numbering, which gives a same-sized context whose prefix
    differs from the first token — the control the reuse probe needs.

    `needles=False` yields pure filler: the same lines, numbered from the same
    start, without the three planted facts. That is what the sizing calibration
    wants, because a needle inside it ends the shared prefix at 10% of its own
    length and leaves the graded prompt with nothing long enough to reuse.
    """
    marks = ({int(lines * f): i for i, f in enumerate((0.10, 0.50, 0.90))}
             if needles else {})
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
    # Needle-free, so the calibration's 200 lines are a real prefix of the graded
    # prompt: a deep leg then either reuses them or shows why it could not.
    probe = build_long_prompt(200, needles=False)
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
    mark = log_mark()
    cold, t_cold = chat(first, max_tokens=32)
    cold_log = log_since(mark)
    depth = prompt_tokens(cold)
    warm = first + [cold["choices"][0]["message"],
                    {"role": "user",
                     "content": "Und welches Kennwort taucht am haeufigsten auf?"}]
    mark = log_mark()
    warm_p, t_warm = chat(warm, max_tokens=32)
    warm_log = log_since(mark)
    other_ctx, _ = prompt_with_needles(target_tokens, offset=10_000_000)
    mark = log_mark()
    other_p, t_other = chat([{"role": "system", "content": other_ctx},
                             {"role": "user", "content": summarize}],
                            max_tokens=32)
    other_log = log_since(mark)

    def server_prefill(entries):
        """Tokens the server says it actually ran, or None without --log."""
        return log_int(entries[-1], "prefill_tokens") if entries else None

    print(f"tiefe  {depth} prompt_tokens")
    for label, elapsed, payload, entries in (
            ("kalt", t_cold, cold, cold_log),
            ("warm", t_warm, warm_p, warm_log),
            ("fremd", t_other, other_p, other_log)):
        ran = server_prefill(entries)
        print(f"{label}  {elapsed:6.2f}s usage.cached={cached(payload)}"
              + (f" server.prefill={ran}" if ran is not None else ""))
        show_log(entries)

    failed = []
    # Reuse is only real if the work was skipped. `usage.cached_tokens` alone
    # cannot tell a taken cache from a field that counts the retained prefix, so
    # the verdict needs the cost: a warm leg finishes in a heartbeat, a cold one
    # pays per token. Where the log is available it decides, together with the
    # prefill + cached == prompt invariant.
    failed += check_accounting(cold_log + warm_log + other_log, "prefix")
    failed += check_reuse("kalt", cached(cold),
                          server_prefill(cold_log), depth, want="cold")
    failed += check_reuse("warm", cached(warm_p),
                          server_prefill(warm_log), depth, want="warm")
    failed += check_reuse("fremd", cached(other_p),
                          server_prefill(other_log), depth, want="cold")
    if t_warm >= t_cold:
        print("FEHLER: warm war nicht schneller als kalt")
        failed.append("warm-time")
    print("PREFIX " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def cmd_coldpair() -> int:
    """Two prompts of the same length that the server has never seen.

    A nonce puts the divergence in the first sentence, so the only tokens the
    two can legitimately share are the ones the chat template writes around the
    user turn — measured at 42 for this model (the system line plus the opened
    assistant tag), and the runner keeps a checkpoint of that frame from the
    first request on. Repeating both then proves reuse still works when it
    should.

    Reuse beyond the frame is what would mean matching by length. Two tests,
    by decreasing sharpness: with `--log`, every leg must report
    `cached_tokens == common_prefix_tokens`, so the server itself states that
    what it skipped really is the common prefix; without it, a cold leg may keep
    the frame but not the prompt, and the answers carry the proof — a prompt
    that had taken the other's cache would name the other's code.
    """
    nonce = uuid.uuid4().hex[:8]
    ask = "Antworte mit genau einer Zeile: NAME | CODE | PERSON."
    a = (f"Merk dir: Projekt ALPHA-{nonce} hat Code 4417 und zustaendig ist "
         f"Ilgen. {ask}")
    b = (f"Merk dir: Projekt BETA-{nonce} hat Code 8802 und zustaendig ist "
         f"Seitz. {ask}")
    legs = []
    for label, text in (("A kalt", a), ("B kalt", b),
                        ("A warm", a), ("B warm", b)):
        mark = log_mark()
        payload, _elapsed = chat([{"role": "user", "content": text}],
                                 max_tokens=64)
        entries = log_since(mark)
        body, where = answer(payload)
        legs.append((label, payload, body, entries))
        print(f"{label}: {prompt_tokens(payload)} tokens, "
              f"cached={cached(payload)} [{where}]: {body.strip()[:120]}")
        show_log(entries)

    a_len = prompt_tokens(legs[0][1])
    b_len = prompt_tokens(legs[1][1])
    failed = []
    if a_len != b_len:
        print(f"HINWEIS: ungleiche Laengen ({a_len} vs {b_len}) — "
              f"Laengengleichheit ist hier nur ein Hinweis, kein Beweis")

    # What the two cold prompts may share is the frame, and the frame is not
    # half a prompt; what settles the question is the content, since a prompt
    # that had taken the other's cache names the other's code. With --log the
    # server's own split is checked too: `prefill + cached` is the prompt on
    # every leg, hit or miss, which is what a field that counts something else
    # cannot satisfy.
    failed += check_accounting(
        [entry for *_rest, entries in legs for entry in entries], "coldpair")
    for label, payload, *_rest in legs[:2]:
        depth = prompt_tokens(payload)
        if cached(payload) > 0.5 * depth:
            print(f"  FEHLER: {label} uebernahm die Haelfte des fremden Prompts "
                  f"({cached(payload)} von {depth}) — Prafix nach Laenge")
            failed.append("cold-hit")

    for (label, _payload, body, _log), want, other in (
            (legs[2], "4417", "8802"), (legs[3], "8802", "4417")):
        if want not in body:
            print(f"  FEHLER: {label} nennt den eigenen Code nicht")
            failed.append(f"{label}-answer")
        if other in body:
            print(f"  FEHLER: {label} antwortet aus der anderen Konversation")
            failed.append(f"{label}-leak")

    for (label, payload, _body, _log), own in ((legs[2], a_len),
                                               (legs[3], b_len)):
        if cached(payload) < own - 4:
            print(f"  FEHLER: {label} wiederholte ohne Wiederverwendung "
                  f"({cached(payload)} von {own})")
            failed.append("warm-miss")
    print("COLDPAIR " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def per_line_tokens() -> float:
    """Tokens per filler line, from one short leg instead of a guess.

    A sweep has to report how deep it actually went, so the filler is sized from
    the server's own count. Two hundred needle-free lines cost about a second and
    the marker in front of each leg keeps that calibration from being a cached
    prefix of the legs behind it.
    """
    payload, _ = chat([{"role": "user",
                        "content": build_long_prompt(200, needles=False)}],
                      max_tokens=1)
    got = prompt_tokens(payload)
    if not got:
        raise SystemExit("server meldet keine prompt_tokens")
    return got / 200.0


def cmd_sweep(depths: list, gen: int) -> int:
    """Served pp and tg against context depth, one cold leg per depth.

    Both rates come from the server (`timings`), over the tokens it really
    processed: pp says what a prompt of that depth costs, tg says what a
    generated token costs once the ten patternless layers hold `depth`
    positions. The weights do not move between legs - only the attention does -
    so the fall in tg across the run is the depth itself.

    Each prompt carries a fresh marker in front: prefix reuse here is byte-exact
    from the first token, so a repeated prefix would report a cache hit as a
    prefill. `cached` is printed because that is the one way to see a leg was
    not cold. A small constant in it is expected and means nothing: the chat
    template's opening tokens are the same bytes in every request, so they are
    always reused. What would mean something is a hit that grows with the leg -
    filler that was already prefilled - so the check is on growth, not on the
    count.

    `timings` names the decode leg `predicted_*` (`generation_metrics.hpp`);
    reading a `generation_*` key here would silently report zero decode.

    """
    per_line = per_line_tokens()
    print(f"kalibrierung: {per_line:.2f} tokens/zeile, {gen} generationstokens")
    print(f"{'prompt':>9} {'pp':>8} {'tg':>8} {'cached':>7} {'wall':>9}")
    failed = []
    for depth in depths:
        lines = max(64, int(depth / per_line))
        prompt = (f"[sweep {os.urandom(8).hex()}]\n"
                  + build_long_prompt(lines, needles=False))
        payload, wall = chat([{"role": "user", "content": prompt}],
                             max_tokens=gen)
        timings = payload.get("timings") or {}
        got, hit = prompt_tokens(payload), cached(payload)
        usage = payload.get("usage") or {}
        pp = timings.get("prompt_per_second") or 0.0
        tg = (timings.get("predicted_per_second")
              or usage.get("completion_tokens_per_second") or 0.0)
        print(f"{got:>9} {pp:>8.1f} {tg:>8.1f} {hit:>7} {wall:>8.1f}s")
        if hit > max(64, got // 50):
            print(f"  FEHLER: leg {got} prefillte nicht kalt "
                  f"({hit} cached von {got})")
            failed.append(f"{got}-warm")
    print("SWEEP " + ("FEHLER: " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


def main() -> int:
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    choices=("recall", "sessions", "prefix", "coldpair",
                             "sweep", "all"))
    ap.add_argument("--tokens", type=int, default=110_000,
                    help="approximate prompt size for recall/prefix probes")
    ap.add_argument("--log", default=os.environ.get("KOLIBRI_LOG", ""),
                    help="server log file; enables prefill/cached accounting")
    ap.add_argument("--depths", default="1024,4096,16384,32768,65536,98304",
                    help="prompt depths for sweep, comma separated")
    ap.add_argument("--gen", type=int, default=64,
                    help="generated tokens per sweep leg (64 matches the "
                         "depth legs already in BENCHMARKS.md)")
    args = ap.parse_args()
    LOG_PATH = args.log
    rc = 0
    if args.mode in ("recall", "all"):
        rc |= cmd_recall(args.tokens)
    if args.mode in ("sessions", "all"):
        rc |= cmd_sessions()
    if args.mode in ("prefix", "all"):
        rc |= cmd_prefix(args.tokens)
    if args.mode in ("coldpair", "all"):
        rc |= cmd_coldpair()
    if args.mode == "sweep":
        # Deliberately not part of `all`: the sweep holds the server's single
        # session slot for tens of minutes, the checks above run in a minute.
        rc |= cmd_sweep([int(n) for n in args.depths.split(",")], args.gen)
    return rc


if __name__ == "__main__":
    sys.exit(main())
