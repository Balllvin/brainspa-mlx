import json
from pathlib import Path

import pytest

from brainspa_mlx import memory

DORY = json.loads((Path(__file__).parent / "dory_config.json").read_text())
GIB = memory.GIB


def test_dory_layer_split():
    types = memory.attention_layer_types(DORY)
    assert len(types) == 36
    assert types.count("full_attention") == 9
    assert types.count("sliding_attention") == 27


@pytest.mark.parametrize("loops, full, sliding, kib", [(2, 14, 42, 56), (4, 24, 72, 96)])
def test_dory_caches_per_loop(loops, full, sliding, kib):
    shape = memory.kv_shape(DORY, loops)
    assert (shape.full_caches, shape.sliding_caches) == (full, sliding)
    assert shape.bytes_per_layer_token == 4096  # K and V, 4 heads of 256, bf16
    assert shape.bytes_per_token == kib * 1024


def test_default_loops_come_from_config():
    assert memory.kv_shape(DORY) == memory.kv_shape(DORY, 2)


def test_sliding_window_caps_short_and_long_context():
    shape = memory.kv_shape(DORY, 2)
    assert shape.sequence_bytes(512) == 512 * 56 * 4096
    assert shape.sequence_bytes(262144) == 262144 * 57344 + 42 * 1024 * 4096
    assert round(shape.sequence_bytes(262144) / GIB, 2) == 14.16


def test_measured_mlx_cache_full_attention_part():
    """A 4-loop MLX run held 3,673,915,392 bytes at 32,768 + 64 tokens. MLX grows a cache in steps of 256 tokens; the
    full-attention part matches exactly and the rest is the sliding windows (427,524,096 bytes at 8K, 32K and 64K)."""
    shape = memory.kv_shape(DORY, 4)
    allocated = -(-(32768 + 64) // 256) * 256
    assert 3_673_915_392 - shape.bytes_per_token * allocated == 427_524_096


def test_max_context_on_64gb_mac_budget():
    budget = (51.84 - 2.98 - 4) * GIB
    two = memory.kv_shape(DORY, 2)
    assert two.max_context(budget, 1, 262144) == 262144
    assert 204_800 <= two.max_context(budget, 4, 262144) < 262144
    assert 98_304 <= two.max_context(budget, 8, 262144) < 131072
    assert 8 * two.sequence_bytes(262144) > budget  # batch 8 at 256K does not fit


def test_no_room_gives_zero():
    assert memory.kv_shape(DORY, 4).max_context(0.1 * GIB, 8, 262144) == 0


def test_plain_transformer_counts_every_layer():
    cfg = {"model_type": "llama", "num_hidden_layers": 4, "num_key_value_heads": 2, "num_attention_heads": 8,
           "hidden_size": 512}
    shape = memory.kv_shape(cfg)
    assert (shape.full_caches, shape.sliding_caches) == (4, 0)
    assert shape.bytes_per_token == 4 * 2 * 2 * 64 * 2
