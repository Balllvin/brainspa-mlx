"""brainspa-mlx: run our MLX models on an Apple Silicon Mac.

    brainspa-mlx generate --model REPO_OR_DIR [--loops 2|4] [--fast] "prompt"
    brainspa-mlx serve    --model REPO_OR_DIR [--loops 2|4] [--fast] [--port 8080] [mlx_lm.server options]
    brainspa-mlx memory   --model REPO_OR_DIR [--loops 2|4] --budget-gb 44.9
"""

from __future__ import annotations

import argparse
import sys

from . import memory, runtime


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Hugging Face repo id or local folder")
    p.add_argument("--loops", type=int, choices=runtime.ALLOWED_LOOPS, default=None,
                   help="Dory loop count: 2 (default) or 4 for hard tasks")
    p.add_argument("--fast", action="store_true", help="Dory only: Metal kernels for the norm steps and a two-stage head")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="brainspa-mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="answer one prompt, then print tokens per second")
    _common(g)
    g.add_argument("--max-tokens", type=int, default=32768)
    g.add_argument("--quiet", action="store_true", help="print only the final summary, not the streamed text")
    g.add_argument("prompt")
    s = sub.add_parser("serve", help="OpenAI-compatible server (mlx_lm.server)")
    _common(s)
    m = sub.add_parser("memory", help="KV cache per token and the longest context that fits a memory budget")
    m.add_argument("--model", required=True)
    m.add_argument("--loops", type=int, choices=runtime.ALLOWED_LOOPS, default=None)
    m.add_argument("--budget-gb", type=float, required=True, help="memory left for the KV cache, in GiB")
    return ap


def cmd_generate(a) -> int:
    model, tokenizer, _ = runtime.load(a.model, a.loops, a.fast)
    stream = None if a.quiet else (lambda t: print(t, end="", flush=True))
    r = runtime.generate(model, tokenizer, a.prompt, a.max_tokens, on_text=stream)
    print()
    print(f"\n--- reasoning: {len(r.reasoning)} chars | answer: {r.answer[:200]!r}")
    print(f"--- {r.generated_tokens} tokens, {r.tokens_per_second:.1f} tokens/s, prompt {r.prompt_tokens} tokens, "
          f"peak memory {r.peak_memory_gb:.2f} GB")
    return 0 if r.answer else 1


def server_argv(rest: list[str]) -> list[str]:
    """mlx_lm.server options with the model card's sampling unless the caller set them."""
    argv = list(rest)
    if "--temp" not in argv:
        argv += ["--temp", str(runtime.TEMPERATURE)]
    if "--top-p" not in argv:
        argv += ["--top-p", str(runtime.TOP_P)]
    return argv


def cmd_serve(a, rest: list[str]) -> int:
    import mlx_lm.server as server

    model_dir = runtime.resolve(a.model)
    config = runtime.read_config(model_dir)
    override = runtime.model_config_override(config, a.loops)
    if a.fast:
        runtime.enable_fast_path(config)
    stock_load = server.load

    def load_with_loops(*args, **kwargs):
        if override:
            kwargs["model_config"] = {**(kwargs.get("model_config") or {}), **override}
        loaded = stock_load(*args, **kwargs)
        if a.fast:
            from . import fasthead

            fasthead.install_head(loaded[0], "filter")
        return loaded

    server.load = load_with_loops
    sys.argv = ["mlx_lm.server", "--model", str(model_dir), *server_argv(rest)]
    server.main()
    return 0


def cmd_memory(a) -> int:
    model_dir = runtime.resolve(a.model)
    config = runtime.read_config(model_dir)
    shape = memory.kv_shape(config, a.loops)
    print(f"full-attention caches {shape.full_caches}, sliding-window caches {shape.sliding_caches} "
          f"(window {shape.window})")
    print(f"KV per token: {shape.bytes_per_token / 1024:.0f} KiB; sliding windows per sequence: "
          f"{shape.sliding_bytes / memory.GIB:.3f} GiB")
    for batch, ctx in memory.fits_table(config, a.budget_gb, a.loops).items():
        print(f"batch {batch}: longest context {ctx:,} tokens")
    return 0


def main(argv: list[str] | None = None) -> int:
    args, rest = parser().parse_known_args(argv)
    if args.cmd == "serve":
        return cmd_serve(args, rest)
    if rest:
        parser().error(f"unrecognized arguments: {' '.join(rest)}")
    return cmd_generate(args) if args.cmd == "generate" else cmd_memory(args)


if __name__ == "__main__":
    sys.exit(main())
