# brainspa-mlx

Run our [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) 1.0 4B models on an [Apple Silicon](https://support.apple.com/en-us/116943) Mac with [Apple](https://www.apple.com)'s [MLX](https://github.com/ml-explore/mlx). This repo holds
only the inference code: load a model, answer a prompt, and serve it over an OpenAI-compatible API.

## Which file do I use?

| Your hardware | Build | Repo |
|---|---|---|
| [Apple Silicon](https://support.apple.com/en-us/116943) Mac ([M1](https://support.apple.com/en-us/116943) or later) | [MLX](https://github.com/ml-explore/mlx): the same model for [Apple Silicon](https://support.apple.com/en-us/116943) Macs, run with [Apple](https://www.apple.com)'s [MLX](https://github.com/ml-explore/mlx). | this code plus an `-MLX` repo below |
| [NVIDIA](https://www.nvidia.com) GPU ([CUDA](https://developer.nvidia.com/cuda-toolkit)) | [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): 4-bit weights for [NVIDIA](https://www.nvidia.com) GPUs ([CUDA](https://developer.nvidia.com/cuda-toolkit)), for example [vLLM](https://github.com/vllm-project/vllm) on [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) ([RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)). | an [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) repo below, served with [vLLM](https://github.com/vllm-project/vllm) |

## Models

| Model | [MLX](https://github.com/ml-explore/mlx) (Mac) | [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) ([NVIDIA](https://www.nvidia.com)) |
|---|---|---|
| [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) 1.0 4B abliterated | [balllvin/Dory1.0-4B-abliterated-NVFP4-MLX](https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4-MLX) | [balllvin/Dory1.0-4B-abliterated-NVFP4](https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4) |
| [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) 1.0 4B abliterated + [Arete](https://huggingface.co/datasets/NousResearch/arete-data-pairs) | [balllvin/Dory1.0-4B-abliterated-Arete-NVFP4-MLX](https://huggingface.co/balllvin/Dory1.0-4B-abliterated-Arete-NVFP4-MLX) | [balllvin/Dory1.0-4B-abliterated-Arete-NVFP4](https://huggingface.co/balllvin/Dory1.0-4B-abliterated-Arete-NVFP4) |

Both are built from [NVIDIA](https://www.nvidia.com)'s
[Nemotron-Labs Dory 1.0 4B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4). Each model card
has the results, the memory tables and how the model was made.

## Install

You need an [Apple Silicon](https://support.apple.com/en-us/116943) Mac, Python 3.12 to 3.14 and [uv](https://docs.astral.sh/uv/) (or [pip](https://pip.pypa.io)).

```bash
git clone https://github.com/Balllvin/brainspa-mlx
cd brainspa-mlx
uv sync
```

`uv sync` installs the exact versions in `uv.lock`: [MLX](https://github.com/ml-explore/mlx) 0.32.4 and [mlx-lm](https://github.com/ml-explore/mlx-lm) from
the [pmichalas fork](https://github.com/pmichalas/mlx-lm) at commit `208065f3`, which adds the [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) model code. With [pip](https://pip.pypa.io),
use the same pins:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install --no-deps .
```

### Accept the terms and log in

Before the first download, open the model page on [Hugging Face](https://huggingface.co), log in, and accept the
[terms of use](TERMS.md) (access is granted at once). Then log in on this Mac with the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), or set `HF_TOKEN` to a read token:

```bash
uv run hf auth login
```


## Answer one prompt

```bash
uv run brainspa-mlx generate --model balllvin/Dory1.0-4B-abliterated-NVFP4-MLX "What is 17 times 23?"
```

The first run downloads the model (3.0 GB) from [Hugging Face](https://huggingface.co). The model thinks first, so the
output shows `Reasoning:`, then `Answer:`, then a line with prefill and decode tokens per second and peak memory.
`--context-file FILE` puts a long text in front of the question, for a realistic prefill test.

[Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) is a looped model: a block of its layers runs several times per token. Use 2 loops (the default) for most work
and `--loops 4` for hard tasks. Thinking is always on, and sampling follows [NVIDIA](https://www.nvidia.com)'s card (temperature 1.0, top-p 0.95).

## Serve it to other apps

```bash
uv run brainspa-mlx serve --model balllvin/Dory1.0-4B-abliterated-NVFP4-MLX --port 8080
```

This starts [mlx-lm's server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) with the loop count set.
Any OpenAI-compatible client can use `http://127.0.0.1:8080/v1`. The reply carries the reasoning in `reasoning` and the
answer in `content`. Reasoning length varies with the question, from a few hundred tokens to several thousand, so set
`max_tokens` high; a low limit cuts the reply inside the reasoning and returns an empty answer:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":4000}'
```

The server holds one model. Clients can leave out the `model` field; apps that need a model name (for example
[Opencode](https://opencode.ai)) use `default_model`. Each reply may use all the context its prompt leaves (the
model's 262,144 tokens minus the prompt, or `--context-length`); a client's own `max_tokens` can lower that. In an
app, raise its output limit for this model to the context length too, or the app cuts long reasoning short (for
[Opencode](https://opencode.ai), `limit.output` on the model in `opencode.json`). The server keeps at most 4 GB of prompt cache
(`--prompt-cache-bytes`). Other `mlx_lm.server` options pass through,
for example `--host 0.0.0.0`.

`--fast` ([Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) only) turns on our [Metal](https://developer.apple.com/metal/) kernels for [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)'s normalisation steps and a two-stage output layer (a 4-bit copy of the
output layer picks the top 1,024 tokens and the original bf16 weights score them). Against the plain path it agreed on
99.4% of next tokens over 2,046 steps (mean KL 0.0006) and ran 15% faster at one request (54.7 against 47.7 tokens per
second on an [M5 Pro](https://www.apple.com/macbook-pro/)); at 8 and 12 requests the two paths ran within 2% of each other.

## How much context fits?

```bash
uv run brainspa-mlx memory --model balllvin/Dory1.0-4B-abliterated-NVFP4-MLX --loops 2 --budget-gb 44.9
```

This prints the KV cache per token and the longest context that fits 1, 4 and 8 parallel requests in the memory you
give it. The model cards carry the full tables. On a 64 GB Mac at 2 loops, one request fits the full 256K context, 4
requests fit 202K each and 8 requests fit 99K each.

## Set it up with a coding agent

Paste this into a coding agent such as [Claude Code](https://claude.com/claude-code) or
[OpenCode](https://opencode.ai), with your app in the brackets.

```text
Download Dory 1.0 4B abliterated and set it up for [your app, for example OpenCode]. Do every step yourself, except the account steps, which are mine.
1. Check whether the Hugging Face CLI (hf) is installed. If it isn't, tell me and show me how to install it. Don't install it yourself.
2. Check this computer. Apple Silicon Mac: Mac build. NVIDIA Blackwell GPU (RTX 50xx, RTX PRO 6000, B200): NVIDIA build. Anything else: tell me it won't run here and suggest workarounds.
3. Walk me through the account steps: make a Hugging Face account if I don't have one, accept the terms on the model page, and make a token. Then run hf auth login for me, or tell me how.
4. Download only the build for this computer and check every file. Mac: https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4-MLX NVIDIA: https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4
5. Get the code. Mac: clone https://github.com/Balllvin/brainspa-mlx. NVIDIA: install vLLM as the model card says.
6. Pick the context length from the memory table on the model card.
7. Ask it one question, check the answer makes sense, and tell me the tokens per second.
8. Start the OpenAI compatible server and connect it to my app, with the app's output limit raised to the context length. No app: show me one curl call.
9. Tell me which build you chose, the server address if there is one, and one command to run it again.
```

## Tests

```bash
uv run pytest
BRAINSPA_MLX_MODEL=balllvin/Dory1.0-4B-abliterated-NVFP4-MLX uv run pytest -m model   # loads the real model
```

## Credits and licence

- [NVIDIA](https://www.nvidia.com), for [Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4).
- [Apple](https://www.apple.com)'s [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) ([MIT](https://opensource.org/license/mit)), and the
  [pmichalas mlx-lm fork](https://github.com/pmichalas/mlx-lm) with the [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) model code.

Using the models means agreeing to the [terms of use](TERMS.md): you alone are responsible and liable for your use of
them and their outputs. The code here is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`LICENSE`). The model weights carry their own licence, the
[OpenMDW 1.1](https://openmdw.ai/license/1-1/) for the [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) builds; see each model card. This is an independent
community project, unaffiliated with [NVIDIA](https://www.nvidia.com) or [Apple](https://www.apple.com).
