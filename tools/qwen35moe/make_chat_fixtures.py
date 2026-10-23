#!/usr/bin/env python3
"""Render the released Ornith chat templates with full Jinja2 ground truth.

Writes tests/models/qwen35moe/fixtures/chat/<variant>_<case>.txt, the bytes
the C++ renderer must reproduce exactly. Variants are the two qualified
template revisions (35b, 122b); error cases live in the C++ test, not here,
because Jinja raises through a different path than the compiled renderer.
"""

from pathlib import Path

from jinja2 import Environment


def bomb(message):
    raise RuntimeError(message)


ENV = Environment()
ENV.globals["raise_exception"] = bomb

T35 = Path("/tmp/opencode/tmpl_35b.jinja").read_text()
T122 = Path("/tmp/opencode/tmpl_122b.jinja").read_text()

CASES = {
    "user_only": {
        "messages": [{"role": "user", "content": "Hello world"}],
        "kwargs": {"add_generation_prompt": True},
    },
    "system_user": {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    "multiturn_thought": {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hey there!",
                "reasoning_content": "greet back",
            },
            {"role": "user", "content": "Bye"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    "think_extract": {
        "messages": [
            {"role": "user", "content": "Think this over"},
            {
                "role": "assistant",
                "content": "<think>\npondering\n</think>\n\nDone.",
            },
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    "no_thinking": {
        "messages": [{"role": "user", "content": "Short answer"}],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": False},
    },
    "no_gen_prompt": {
        "messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Q?"},
        ],
        "kwargs": {"add_generation_prompt": False},
    },
    "tool_roundtrip": {
        "messages": [
            {"role": "user", "content": "How are you?"},
            {
                "role": "assistant",
                "content": "Let me check.",
            },
            {"role": "tool", "content": "fine"},
            {"role": "tool", "content": "indeed"},
            {"role": "user", "content": "Thanks"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    "double_system": {
        "messages": [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Go"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    "developer": {
        "messages": [
            {"role": "developer", "content": "Dev note."},
            {"role": "user", "content": "Go"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    outdir = root / "tests" / "models" / "qwen35moe" / "fixtures" / "chat"
    outdir.mkdir(parents=True, exist_ok=True)
    for variant, source in (("35b", T35), ("122b", T122)):
        template = ENV.from_string(source)
        for case, spec in CASES.items():
            try:
                rendered = template.render(
                    messages=spec["messages"], **spec["kwargs"]
                )
            except RuntimeError as exc:
                print(f"skip {variant}_{case}: template raised {exc}")
                continue
            (outdir / f"{variant}_{case}.txt").write_text(rendered)
            print(f"wrote {variant}_{case}.txt ({len(rendered)} bytes)")


if __name__ == "__main__":
    main()
