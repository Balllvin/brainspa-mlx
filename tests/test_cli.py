import json
import os
import sys
import types
from pathlib import Path

import pytest

from brainspa_mlx import cli, runtime

DORY = json.loads((Path(__file__).parent / "dory_config.json").read_text())


def test_loops_override_for_dory():
    assert runtime.model_config_override(DORY, None) == {"n_recurrent_loops": 2}
    assert runtime.model_config_override(DORY, 4) == {"n_recurrent_loops": 4}
    with pytest.raises(ValueError):
        runtime.model_config_override(DORY, 1)


def test_no_loops_for_other_models():
    assert runtime.model_config_override({"model_type": "lfm3"}, None) is None
    with pytest.raises(ValueError):
        runtime.model_config_override({"model_type": "lfm3"}, 4)


@pytest.mark.parametrize("text, reasoning, answer", [
    ("Some steps.\n</think>\n391", "Some steps.", "391"),
    ("<think>a</think>b", "a", "b"),
    ("still thinking", "still thinking", ""),
])
def test_split_thinking(text, reasoning, answer):
    assert runtime.split_thinking(text) == (reasoning, answer)


def test_server_argv_adds_card_sampling():
    assert cli.server_argv(["--port", "9000"]) == ["--port", "9000", "--temp", "1.0", "--top-p", "0.95"]
    assert cli.server_argv(["--temp", "0.6", "--top-p", "0.9"]) == ["--temp", "0.6", "--top-p", "0.9"]


def test_serve_injects_loops_into_server_load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(DORY))
    calls = {}
    fake = types.ModuleType("mlx_lm.server")
    fake.load = lambda *a, **k: calls.setdefault("load", k) or ("model", "tok")
    fake.main = lambda: calls.setdefault("main", fake.load(str(tmp_path), adapter_path=None) and sys.argv[:])
    monkeypatch.setitem(sys.modules, "mlx_lm.server", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    sys.modules["mlx_lm"].server = fake
    assert cli.main(["serve", "--model", str(tmp_path), "--loops", "4", "--port", "9001"]) == 0
    assert calls["load"]["model_config"] == {"n_recurrent_loops": 4}
    assert calls["main"][:3] == ["mlx_lm.server", "--model", str(tmp_path)]
    assert "9001" in calls["main"]


def test_memory_command(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps(DORY))
    assert cli.main(["memory", "--model", str(tmp_path), "--budget-gb", "44.86"]) == 0
    out = capsys.readouterr().out
    assert "KV per token: 56 KiB" in out
    assert "batch 1: longest context 262,144 tokens" in out


def test_unknown_option_is_an_error():
    with pytest.raises(SystemExit):
        cli.main(["memory", "--model", "x", "--budget-gb", "1", "--bogus"])


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get("BRAINSPA_MLX_MODEL"), reason="set BRAINSPA_MLX_MODEL to run a real model")
def test_real_model_answers():
    model, tokenizer, _ = runtime.load(os.environ["BRAINSPA_MLX_MODEL"], 2)
    r = runtime.generate(model, tokenizer, "What is 17 times 23? Answer with the number.", max_tokens=4096)
    assert "391" in r.answer
    assert r.reasoning
    assert r.tokens_per_second > 0
