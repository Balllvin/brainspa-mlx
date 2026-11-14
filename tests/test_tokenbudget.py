import types

from brainspa_mlx import tokenbudget


def test_fit_is_context_minus_prompt_unless_the_client_asks_for_less():
    assert tokenbudget.fit(262144, 1000, 262144) == 261144
    assert tokenbudget.fit(500, 1000, 262144) == 500
    assert tokenbudget.fit(262144, 262144, 262144) == 1


def _fake_server():
    calls = {}

    class BatchGenerator:
        def insert_segments(self, segments, max_tokens=None, caches=None, all_tokens=None, samplers=None):
            calls["batch"] = max_tokens

    def stream_generate(model=None, stream=None, tokenizer=None, prompt=None, max_tokens=0, prompt_cache=None):
        calls["single"] = max_tokens

    return types.SimpleNamespace(BatchGenerator=BatchGenerator, stream_generate=stream_generate), calls


def test_batch_path_counts_cached_and_new_prompt_tokens():
    server, calls = _fake_server()
    tokenbudget.install(server, 10_000)
    server.BatchGenerator().insert_segments(segments=[[[1] * 300, [2] * 200]], max_tokens=[10_000],
                                            caches=[None], all_tokens=[[0] * 500], samplers=[None])
    assert calls["batch"] == [9_000]
    server.BatchGenerator().insert_segments(segments=[[[1] * 10]], max_tokens=[64], caches=[None], all_tokens=[[]])
    assert calls["batch"] == [64]


def test_single_path_counts_the_prompt_cache():
    server, calls = _fake_server()
    tokenbudget.install(server, 10_000)
    cache = [types.SimpleNamespace(offset=4_000)]
    server.stream_generate(prompt=[1] * 1_000, max_tokens=10_000, prompt_cache=cache)
    assert calls["single"] == 5_000
    server.stream_generate(prompt=[1] * 1_000, max_tokens=300, prompt_cache=None)
    assert calls["single"] == 300
