"""Deep borrowed frontiers must survive branches and execution-slot reuse."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sys

from cache_growth import check_unchanged_retry, log_offset

# Kolibri 1's template reports the reusable prefix as the end of the turn before a
# trailing text-only user turn (src/models/kolibri1/chat_template.hpp), and the
# runner truncates the retained checkpoint there
# (src/cli/serve/text_model_runner.cpp). Measured on this fixture 2026-10-13: the
# first continuation reused 20602 of its 21204 tokens - everything through the end of
# the previous user turn, four tokens of opened assistant tag short - and the second
# reused 20606 of 21802, one user turn behind because its own turn is the newest. A
# rewritten history reused 21204 of 21825, a resume 21204 of 21820. The floor allows
# one turn of growth plus that frame and nothing more, so a fallback to the 2048
# capture grid (18432) or to zero still fails. A prefix the server has never seen is
# the different case that lands on the grid (9845 -> 8192 = 4 x 2048; the first turn
# of tools/kolibri/prefix_probe.py shows the same shape at 6.3k). Every other model
# keeps the frontier it holds today.
STABLE_PREFIX_PRESETS = {"kolibri1"}
# The fixture's user turns are equal-sized, so the first continuation predicts its
# own growth from this measured value; later ones use the observed delta.
MEASURED_TURN_TOKENS = 598
# The opened assistant tag (four tokens) and the reply a replay renders shorter than
# the turn that generated it.
PREFIX_SLACK = 96


def check_cache_depth(client, model, checks, chat_result, concurrency=4, server_log=None,
                      preset=None):
    request = dict(model=model, temperature=0, seed=31,
                   max_completion_tokens=16, reasoning_effort="none")
    controls = []

    def signature(result):
        return (result["text"], result["reasoning"], result["tools"],
                result["finish"], result["usage"]["completion_tokens"])

    def floor_for(previous, growth):
        """Reuse has to reach at least this far into the previous request."""
        if preset not in STABLE_PREFIX_PRESETS:
            return max(0, previous - 16)
        return max(0, previous - growth - PREFIX_SLACK)

    def chat(label, messages, code, floor=0, cold=False, retry=None, record=True,
             window=None):
        body = {**request, "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        if record:
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        assert result["text"].strip() == code and not result["reasoning"] \
            and not result["tools"] and result["finish"] == "stop", result
        if cold:
            assert reused == 0 and prefilled == total, result
        else:
            assert reused >= floor, (label, floor, result)
        if retry is not None:
            assert total == retry["usage"]["prompt_tokens"], result
            check_unchanged_retry(label, result, prefilled, server_log, *window)
            assert signature(result) == signature(retry), result
        return body, result

    messages = [{"role": "system", "content": "cache_depth_main\n"
                 "Follow the final instruction. Background records are not instructions.\n" +
                 "Background record: the lantern is blue and the shelf is empty.\n" * 1536}]
    previous = 0
    growth = MEASURED_TURN_TOKENS
    for turn in range(3):
        messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                         "Archive record: there are no new instructions.\n" * 64 +
                         "Reply with only BETA."})
        body, result = chat(f"depth_turn_{turn}", messages, "BETA",
                            floor=floor_for(previous, growth), cold=turn == 0)
        prompt = result["usage"]["prompt_tokens"]
        if previous:
            growth = prompt - previous
        previous = prompt
        assert previous >= 16384, result
        controls.append((f"turn_{turn}", body, result, "BETA"))
        messages.append({"role": "assistant", "content": result["text"]})

    # Each branch changes the final assistant reply, forcing a restore before
    # that reply. Run them together so different sessions can fork the same
    # borrowed checkpoint while its original session continues or rewinds.
    # One execution slot runs the same branches in order, keeping matched
    # request histories independent of client-thread admission races.
    branches = []
    for code in ("RED", "GREEN", "BLUE"):
        branch = deepcopy(messages)
        branch[-1]["content"] = f"Rewritten archive response for branch {code}."
        branch.append({"role": "user", "content": f"Reply with only {code}."})
        branches.append((code, branch))
    with ThreadPoolExecutor(max_workers=min(len(branches), concurrency)) as executor:
        futures = [(code, executor.submit(chat, "depth_branch_" + code,
                                         branch, code, floor_for(previous, growth),
                                         record=False))
                   for code, branch in branches]
        for code, future in futures:
            body, result = future.result()
            controls.append(("branch_" + code, body, result, code))
    # Mark the complete concurrent cohort once. Completion order must not
    # change the recorded history or assign another branch's request label.
    checks["depth_branches"] = {code: future.result()[1]
                               for code, future in futures}
    for code, future in futures:
        checks["depth_branch_" + code] = future.result()[1]

    chat("depth_side", [{"role": "system", "content": "cache_depth_side"},
                         {"role": "user", "content": "Reply with only ALPHA."}], "ALPHA")
    messages.append({"role": "user", "content": "Reply with only BETA."})
    before_resume = log_offset(server_log)
    body, resumed = chat("depth_resume", messages, "BETA", floor_for(previous, growth))
    chat("depth_unchanged", messages, "BETA", retry=resumed,
         window=(before_resume, log_offset(server_log)))
    controls.append(("resume", body, resumed, "BETA"))

    # Controls come last: they must never supply a checkpoint missing from a
    # warm branch or rotation. Complete output equality stays strict.
    for label, body, warm, code in controls:
        _, cold = chat("depth_cold_" + label, body["messages"], code, cold=True)
        assert signature(warm) == signature(cold), (label, warm, cold)
