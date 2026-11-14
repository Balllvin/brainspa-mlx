"""Max new tokens per request = the context length minus that request's prompt (no fixed cap).

mlx_lm.server gives every request one default `max_tokens`. `install` wraps the two places the server starts a
generation, where the full prompt length is known, and lowers each request's limit to the room left in the context.
A client's own `max_tokens`, when smaller, still wins.
"""

from __future__ import annotations


def fit(requested: int, prompt_tokens: int, context: int) -> int:
    """The request's limit, lowered to the tokens left in the context (at least 1)."""
    return max(1, min(requested, context - prompt_tokens))


def _cached_tokens(cache) -> int:
    return int(getattr(cache[0], "offset", 0) or 0) if cache else 0


def install(server, context: int) -> None:
    """Patch an imported `mlx_lm.server` module in place."""
    stock_stream_generate = server.stream_generate

    def stream_generate(*args, **kwargs):
        prompt = kwargs.get("prompt", args[3] if len(args) > 3 else [])
        n = _cached_tokens(kwargs.get("prompt_cache")) + len(prompt)
        kwargs["max_tokens"] = fit(kwargs.get("max_tokens", context), n, context)
        return stock_stream_generate(*args, **kwargs)

    class BudgetedBatchGenerator(server.BatchGenerator):
        def insert_segments(self, segments, max_tokens=None, caches=None, all_tokens=None, *rest, **kwargs):
            if max_tokens is not None:
                done = all_tokens or [[] for _ in segments]
                max_tokens = [fit(m, len(t or []) + sum(len(s) for s in seg), context)
                              for m, seg, t in zip(max_tokens, segments, done)]
            return super().insert_segments(segments, max_tokens, caches, all_tokens, *rest, **kwargs)

    server.stream_generate = stream_generate
    server.BatchGenerator = BudgetedBatchGenerator
