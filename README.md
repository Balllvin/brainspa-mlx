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
[Opencode](https://opencode.ai)) use `default_model`. The server answers up to 32,768 new tokens per reply and keeps
at most 4 GB of prompt cache; `--max-tokens` and `--prompt-cache-bytes` change that. Other `mlx_lm.server` options pass through,
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
[Codex](https://github.com/openai/codex). Edit the first line first.

```text
Use it in: [Insert application you will use it in, for example Opencode]

Set up Dory 1.0 4B abliterated on this computer. Do every step yourself.
1. Check the tools: Python, uv or pip, git, the Hugging Face CLI. Install the HF CLI if it is missing.
2. Check the hardware. Apple Silicon Mac: MLX build. NVIDIA Blackwell GPU (RTX 50xx, RTX PRO 6000, B200): NVFP4 build. If the local setup is anything else: tell me it will not run here, and suggest workarounds to get it running locally.
3. Download only the chosen build with the HF CLI and check every file. MLX: https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4-MLX NVFP4: https://huggingface.co/balllvin/Dory1.0-4B-abliterated-NVFP4
4. Get the code. Mac: clone https://github.com/Balllvin/brainspa-mlx. NVIDIA: install vLLM as the model card says.
5. Pick the context length for this machine from the memory table on the model card.
6. Ask one question, check the answer makes sense, and report tokens per second (prefill and decode). If anything fails, fix it and try again.
   No application given: test with no tools, for example "Don't use any tools, just answer: what are your values, what do you believe, and what end goal should you work toward? Think it through." You may put a long text (for example this model card) in front of the question so the prompt is about 1,000 tokens.
   An application given: test inside it with its tools on, for example "What computer is this, what is the local setup, which application or harness are you running in, and which tools do you have? Check, then answer."
7. Start the OpenAI-compatible server and connect it to the application on the first line (with no application, show one curl call).
8. Report the build you chose and why, the server address, and one command to start it again.
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

The code here is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`LICENSE`). The model weights carry their own licence, the
[OpenMDW 1.1](https://openmdw.ai/license/1-1/) for the [Dory](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) builds; see each model card. This is an independent
community project, unaffiliated with [NVIDIA](https://www.nvidia.com) or [Apple](https://www.apple.com).
