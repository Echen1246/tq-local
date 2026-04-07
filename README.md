<img width="246" height="50" alt="Screenshot 2026-04-06 at 8 53 35 PM" src="https://github.com/user-attachments/assets/78510409-d458-40a9-829e-6f59d3c7b2b6" />

KV cache compression for HuggingFace Transformers. Reduces VRAM usage
by ~74% for long-context inference using DeepMind's TurboQuant algorithm
(ICLR 2026), with a fused Triton attention kernel.

> **Research details:** Algorithm math, Q_prod vs Q_mse, kernel design,
> benchmarks, and tradeoffs → [research/README.md](research/README.md)
>
> **Model support:** Tested models, known issues, Triton status →
> [models.md](models.md)

---

## Install

```bash
pip install git+https://github.com/Echen1246/local-turboquant.git
```

Or for development:

```bash
git clone https://github.com/Echen1246/local-turboquant.git
cd local-turboquant
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (for real inference)
- Triton (auto-installed, enables fused kernel)

**Cloud GPUs (Vast.ai, Lambda, Modal):** see [remote_setup.md](remote_setup.md).

---

## Quick start

### Attach to any Transformers model

```python
import turboquant
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

# One line — all subsequent generate() calls use compressed KV cache
turboquant.activate(model, tokenizer, bits=4)

# Use model.generate() exactly as before
inputs = tokenizer("What is KV cache compression?", return_tensors="pt").to("cuda")
output = model.generate(inputs.input_ids, max_new_tokens=1024)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Check compression stats
print(turboquant.last_telemetry(model))
# → {'payload_savings_percent': 73.8, ...}

# Detach when done
turboquant.deactivate(model)
```

That's it. Every `model.generate()` call automatically compresses the KV
cache. No changes to your prompting code, tokenizer setup, or generation
config.

### Session API (more control)

```python
from turboquant import TurboQuantSession

session = TurboQuantSession.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    bits=4,
    device_map="auto",
)
print(session.generate("Explain quantum computing.", max_new_tokens=1024))
print(session.last_telemetry())
```

---

## CLI

### Welcome / help

```bash
turboquant
```

### Run a prompt

```bash
# Defaults: Llama 3.1 8B, 4-bit Q_prod, telemetry shown
turboquant run --prompt "Explain KV cache compression."

# Custom model, 3-bit
turboquant run --model Qwen/Qwen2.5-7B-Instruct --bits 3 --prompt "Hello"

# Q_mse instead of Q_prod
turboquant run --no-qjl --prompt "Hello"

# JSON output (for scripting)
turboquant run --prompt "Hello" --json
```

### Interactive session

```bash
turboquant attach

# Or with a different model / generation cap
turboquant attach --model Qwen/Qwen2.5-7B-Instruct --bits 3 --max-new-tokens 2048
```

Default `--max-new-tokens` matches the session API (see `turboquant.constants.DEFAULT_MAX_NEW_TOKENS`). Inside the REPL, type `/tokens N` or `/help`.

Loads the model once, then gives you an interactive prompt:

```
  ✓ TurboQuant 4-bit Q_prod active
  Type a prompt and press Enter. Ctrl+C or 'exit' to quit.

> What is attention?
[response]
[TurboQuant] 74% KV saved | 3871 MB freed | 16.3 tok/s

> /stats       # show last telemetry
> /tokens 512  # change max generation length
> /help        # list commands
```

### System info

```bash
turboquant setup
```

Detects GPU, VRAM, CUDA version, Triton availability, and recommends
models + bit widths for your hardware.

---

## Defaults

| Setting | Default | Override |
|---------|---------|----------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` | `--model <id>` |
| Bits | 4 | `--bits 3` |
| Algorithm | Q_prod (3-bit MSE keys + 1-bit QJL) | `--no-qjl` for Q_mse |
| Decode quant | On (re-quantize generated tokens) | `--no-quantize-decode` |
| Max tokens | 256 | `--max-new-tokens N` |

### Bit width guide

| Bits | Quality | KV savings | Notes |
|------|---------|------------|-------|
| 4 | Near-lossless (cosine sim ~0.995) | ~74% | Recommended default |
| 3 | Very good (cosine sim ~0.983) | ~80% | VRAM-constrained setups |
| 2 | Experimental | ~87% | Visibly lossy |

---

## HuggingFace token

Gated models (Llama, Gemma) require accepting the license on HuggingFace
and setting a token:

```bash
export HF_TOKEN="hf_your_token_here"
```

Or pass directly: `turboquant run --token hf_your_token_here --prompt "Hi"`

---

## Tested models

| Model | Status | Notes |
|-------|--------|-------|
| Llama 3.1 8B Instruct | Fully working | 74% savings, all layers quantize cleanly |
| Qwen 2.5 7B Instruct | Works with norm guard | 3/28 layers kept dense, 71.5% savings |

See [models.md](models.md) for details, known issues, and work in progress.

---

## Troubleshooting

**Gated repo error** — Accept the model license on HuggingFace and set `HF_TOKEN`.

**CUDA OOM** — Try `--bits 3`, reduce `--max-new-tokens`, or use a smaller model.

**Slow generation** — Expected tradeoff: TurboQuant trades decode latency
for VRAM. At long contexts (32K+), the fused Triton kernel runs at ~2x
baseline. The latency gap narrows as context grows because the VRAM
savings increasingly matter.

**No module 'turboquant'** — Activate your venv and run `pip install -e .`

---

## Paper

- TurboQuant: https://openreview.net/forum?id=tO3ASKZlok
- QJL: https://arxiv.org/abs/2406.03482
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
