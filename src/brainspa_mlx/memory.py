"""KV cache size from a model's config.json, in bf16 (2 bytes per value).

Dory is a looped model. Its attention layers sit in three blocks: input, recurrent and output. With
``recurrent_kv_cache_mode: per_loop`` each loop of the recurrent block keeps its own cache, so the recurrent block's
caches count once per loop. Sliding-window layers keep at most ``sliding_window`` tokens per sequence; full-attention
layers grow with the context.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 2**30
KV_BYTES = 2  # bf16


@dataclass(frozen=True)
class KvShape:
    full_caches: int
    sliding_caches: int
    bytes_per_layer_token: int
    window: int

    @property
    def bytes_per_token(self) -> int:
        """Bytes each new token adds once the sliding windows are full."""
        return self.full_caches * self.bytes_per_layer_token

    @property
    def sliding_bytes(self) -> int:
        """Bytes one sequence holds in its sliding-window caches once the context passes the window."""
        return self.sliding_caches * self.window * self.bytes_per_layer_token

    def sequence_bytes(self, context: int) -> int:
        return self.bytes_per_token * context + self.sliding_caches * self.bytes_per_layer_token * min(context, self.window)

    def max_context(self, budget_bytes: float, batch: int, limit: int) -> int:
        """Longest context that fits ``batch`` sequences of equal length in ``budget_bytes``, capped at ``limit``."""
        per_sequence = budget_bytes / batch - self.sliding_bytes
        if per_sequence <= 0:
            return 0
        return min(limit, int(per_sequence // self.bytes_per_token))


def attention_layer_types(config: dict) -> list[str]:
    types = config.get("layer_types")
    if types:
        return [t for t in types if t in ("full_attention", "sliding_attention")]
    return ["full_attention"] * config["num_hidden_layers"]


def kv_shape(config: dict, loops: int | None = None) -> KvShape:
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    per_layer = 2 * config["num_key_value_heads"] * head_dim * KV_BYTES
    types = attention_layer_types(config)
    full = sliding = 0
    if config.get("model_type") == "dory":
        loops = loops or config["n_recurrent_loops"]
        per_loop = config.get("recurrent_kv_cache_mode", "per_loop") == "per_loop"
        # n_*_layers count mixer and MLP layers; each attention block is one of each.
        lo = config["n_input_layers"] // 2
        hi = lo + config["n_recurrent_layers"] // 2
        for i, kind in enumerate(types):
            copies = (loops if per_loop else 1) if lo <= i < hi else 1
            if kind == "full_attention":
                full += copies
            else:
                sliding += copies
    else:
        full = types.count("full_attention")
        sliding = types.count("sliding_attention")
    return KvShape(full, sliding, per_layer, config.get("sliding_window") or 0)


def fits_table(config: dict, budget_gib: float, loops: int | None = None, batches=(1, 4, 8)) -> dict[int, int]:
    shape = kv_shape(config, loops)
    limit = config.get("max_position_embeddings", 262144)
    return {b: shape.max_context(budget_gib * GIB, b, limit) for b in batches}
