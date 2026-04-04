# tq-local

This repo contains two layers:

- [src/turboquant](/Users/eddie/Documents/turboquant/src/turboquant): the reusable TurboQuant-style runtime and math package
- [research](/Users/eddie/Documents/turboquant/research): the proof-of-concept research harness, benchmark workflow, and published results

Current status:

- `TurboQuant_mse` is implemented
- a packed `Q_mse` cache runtime is implemented
- the proof of concept was validated on `Qwen/QwQ-32B` with Modal and NIAH

If you want the research implementation and reproduced results, start in [research/README.md](/Users/eddie/Documents/turboquant/research/README.md).

If you want the installable library, start in [src/turboquant](/Users/eddie/Documents/turboquant/src/turboquant).

## Install from GitHub

```bash
pip install git+https://github.com/Echen1246/tq-local.git
```

or

```bash
uv pip install git+https://github.com/Echen1246/tq-local.git
```

For local development from this repo:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Library quickstart

```python
from turboquant import TurboQuantSession

session = TurboQuantSession.from_pretrained(
    "Qwen/QwQ-32B",
    variant="qmse_packed",
    bits=3,
    device_map="auto",
    dtype="auto",
)

text = session.generate(
    messages=[{"role": "user", "content": "Explain KV cache compression in one paragraph."}],
    max_new_tokens=256,
)

print(text)
print(session.last_metrics())
print(session.last_telemetry())
print(session.compatibility_report())
```

## CLI quickstart

Check whether a model looks compatible:

```bash
turboquant inspect --model Qwen/QwQ-32B
```

Run a prompt with packed `3-bit` TurboQuant and print telemetry:

```bash
turboquant run \
  --model Qwen/QwQ-32B \
  --variant qmse_packed \
  --bits 3 \
  --prompt "Explain KV cache compression in one paragraph." \
  --show-telemetry
```

Run baseline generation for comparison:

```bash
turboquant run \
  --model Qwen/QwQ-32B \
  --variant baseline \
  --prompt "Explain KV cache compression in one paragraph." \
  --show-telemetry
```

Use a prompt file and JSON output:

```bash
turboquant run \
  --model /path/to/local/model \
  --prompt-file ./prompt.txt \
  --variant qmse_packed \
  --bits 3 \
  --json
```

Useful knobs:

- `--variant baseline|qmse|qmse_packed`
- `--bits 2|3|4`
- `--max-new-tokens 256`
- `--attn-implementation sdpa`
- `--device-map auto`
- `--dtype auto`

## Current library direction

The library is being shaped for decoder-only Hugging Face `transformers` models with standard KV cache behavior.

The long-term plan is:

- keep the math and packed-cache runtime in the installable package
- keep Modal, benchmarks, and reproducibility flows under `research/`
- broaden compatibility from Qwen-first validation to a wider `transformers` model family

Current v1 target:

- decoder-only causal LLMs
- standard Hugging Face cache/update behavior
- full-attention models first

Out of scope for v1:

- encoder-decoder models
- model-specific cache implementations
- sliding/chunked attention families we have not explicitly validated
- Ollama / vLLM / llama.cpp backends
