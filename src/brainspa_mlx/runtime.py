"""Load a model with its loop count and generate with thinking on."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOOPS = 2
ALLOWED_LOOPS = (2, 4)
TEMPERATURE = 1.0
TOP_P = 0.95


def model_config_override(config: dict, loops: int | None) -> dict | None:
    """The config change that sets Dory's loop count. Other models have no loops and get no change."""
    if config.get("model_type") != "dory":
        if loops not in (None, DEFAULT_LOOPS):
            raise ValueError(f"--loops applies to Dory only, not to model_type {config.get('model_type')!r}")
        return None
    loops = DEFAULT_LOOPS if loops is None else loops
    if loops not in ALLOWED_LOOPS:
        raise ValueError(f"Dory runs with 2 or 4 loops, not {loops}")
    return {"n_recurrent_loops": loops}


def resolve(model: str) -> Path:
    """A local folder, or a Hugging Face repo id downloaded to the Hugging Face cache."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def read_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


def load(model: str, loops: int | None = None, fast: bool = False):
    """Return (model, tokenizer, model_dir). ``fast`` turns on the Dory Metal kernels and two-stage head."""
    from mlx_lm import load as mlx_load

    model_dir = resolve(model)
    config = read_config(model_dir)
    override = model_config_override(config, loops)
    if fast:
        enable_fast_path(config)
    loaded, tokenizer = mlx_load(str(model_dir), model_config=override)
    if fast and config.get("model_type") == "dory":
        from . import fasthead

        fasthead.install_head(loaded, "filter")
    return loaded, tokenizer, model_dir


def enable_fast_path(config: dict) -> None:
    if config.get("model_type") != "dory":
        raise ValueError("--fast applies to Dory only")
    from . import fastops

    fastops.install()


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer). The chat template opens ``<think>`` in the prompt, so the output starts inside it."""
    head, sep, tail = text.partition("</think>")
    if not sep:
        return text.removeprefix("<think>").strip(), ""
    return head.removeprefix("<think>").strip(), tail.strip()


@dataclass
class Result:
    reasoning: str
    answer: str
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    seconds: float
    peak_memory_gb: float


def generate(model, tokenizer, prompt: str, max_tokens: int = 32768, on_text=None) -> Result:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
    text, last, start = "", None, time.perf_counter()
    for last in stream_generate(model, tokenizer, rendered, max_tokens=max_tokens, sampler=sampler):
        text += last.text
        if on_text:
            on_text(last.text)
    reasoning, answer = split_thinking(text)
    return Result(reasoning, answer, last.prompt_tokens, last.generation_tokens, last.generation_tps,
                  time.perf_counter() - start, last.peak_memory)
