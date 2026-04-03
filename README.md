# turboquant

TurboQuant replication workspace for `Qwen/QwQ-32B`, with Modal as the execution environment and official upstream benchmarks as the evaluation surface.

## Principles

- Use the real Hugging Face weights from `Qwen/QwQ-32B`.
- Use the real public benchmark repos and datasets first.
- Keep `TurboQuant_mse` and `TurboQuant_prod` separate in the architecture.
- Do not silently turn `Q_prod` into a merged KV vector.
- Establish a clean baseline before adding any quantization logic.

## What is in this repo

- [`RESEARCH.md`](/Users/eddie/Documents/turboquant/RESEARCH.md): paper notes, benchmark scope, and implementation caveats.
- `src/turboquant/sources.py`: official source registry for the model, paper, and benchmarks.
- `src/turboquant/modal_app.py`: Modal baseline app for `Qwen/QwQ-32B` on `H200`, `cpu=4`, `memory=16384`.
- `src/turboquant/quantization/protocol.py`: quantization interfaces with a hard guardrail against unsafe `Q_prod` reconstruction.
- `src/turboquant/runtime/kv_capture.py`: prompt-level KV extraction and per-layer norm summaries.
- `src/turboquant/config.py`: pinned model revision and shared runtime constants.

## First milestones

1. Verify baseline `Qwen/QwQ-32B` generation on Modal with the official weights.
2. Run official `NIAH` and `LongBench` infrastructure without shortcuts.
3. Add offline KV extraction and distortion analysis.
4. Add `TurboQuant_mse` reference math.
5. Add a custom attention path for `TurboQuant_prod`.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[benchmarks,modal,dev]"
```

## Quick checks

Show the pinned upstream sources:

```bash
turboquant sources
```

Run the Modal baseline app:

```bash
modal run src/turboquant/modal_app.py --prompt "Explain KV cache compression in one paragraph."
```

Warm the remote Modal weight cache without generating:

```bash
modal run src/turboquant/modal_app.py --prefetch-only
```

## Modal setup

1. Install Modal locally:

```bash
uv tool install modal
```

2. Authenticate your local CLI with your Modal account:

```bash
modal setup
```

If that does not work, Modal's docs say `python -m modal setup` is the fallback.

3. Verify the CLI can see your account:

```bash
modal token info
```

4. Optionally set a Hugging Face token locally for more reliable model downloads:

```bash
export HF_TOKEN=...
```

The current Modal app automatically forwards a local `HF_TOKEN` into the remote container if it is set.

5. Run the baseline remote generation job:

```bash
modal run src/turboquant/modal_app.py --prompt "Explain KV cache compression in one paragraph."
```

6. If you want to warm the cache first, do this once:

```bash
modal run src/turboquant/modal_app.py --prefetch-only
```

7. If you want to pin a specific Hugging Face revision:

```bash
modal run src/turboquant/modal_app.py --prompt "Say hello." --revision 976055f8c83f394f35dbd3ab09a285a984907bd0
```

The model weights are cached in a named Modal Volume so later runs do not need to re-download everything.

## Tested baseline dependency set

The current baseline is pinned to public stable releases checked on 2026-03-30:

- `modal==1.3.5`
- `torch==2.10.0`
- `transformers==5.3.0`
- `accelerate==1.13.0`
- `huggingface_hub[hf_transfer]==1.6.0`
- `safetensors==0.7.0`
- `numpy==2.4.3`
- `scipy==1.17.1`

The baseline runtime intentionally uses `attn_implementation="sdpa"` instead of `flash_attention_2`.
That avoids forcing an extra CUDA extension before we have a reason to install and validate it.

## Paper-faithful workflow

1. Prefetch and pin the exact `QwQ-32B` snapshot revision.
2. Run a baseline generation and write metadata to the artifacts volume.
3. Capture prompt KV tensors from the real model and store them as safetensors.
4. Compare TurboQuant math against those real tensors before any custom kernel work.
5. Only after the data pipeline is stable, wire in official `NIAH` and `LongBench-E` runs.

Run a pinned baseline generation and write artifacts:

```bash
modal run src/turboquant/modal_app.py \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name baseline-smoke \
  --prompt "Summarize how KV-cache compression differs from weight quantization."
```

Capture real prompt KV tensors:

```bash
modal run src/turboquant/modal_app.py \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --capture-kv \
  --run-name kv-smoke \
  --prompt "Summarize how KV-cache compression differs from weight quantization."
```

Analyze the captured KV artifact with the paper-faithful `TurboQuant_mse` reference path:

```bash
modal run src/turboquant/modal_app.py \
  --analyze-turboquant-mse kv-smoke \
  --bits 3 \
  --target both
```

Run a conservative bit sweep on the same captured artifact:

```bash
modal run src/turboquant/modal_app.py \
  --analyze-turboquant-mse kv-smoke \
  --bits-list 2,3,4 \
  --target both
```

Capture a small built-in prompt suite for repeated offline analysis:

```bash
modal run src/turboquant/modal_app.py \
  --capture-suite science_smoke \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0
```

Every capture and analysis run also appends a compact permanent record to:

`/vol/artifacts/logs/experiment_log.jsonl`

Important caution:
the current `inner_product_mse` field in the TurboQuant MSE analysis is explicitly a
`random_unit_query_proxy`, not yet a metric based on actual model query vectors.

Print the paper-faithful benchmark manifest:

```bash
turboquant benchmarks
```

Run a single local `NIAH` case with deterministic scoring:

```bash
modal run src/turboquant/modal_app.py \
  --niah-context-length 8000 \
  --niah-depth-percent 50 \
  --variant baseline \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-smoke
```

Run a small baseline `NIAH` grid within the current model limit:

```bash
modal run src/turboquant/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant baseline \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-baseline
```

Run the first live `Q_mse` `NIAH` grid:

```bash
modal run src/turboquant/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant qmse \
  --qmse-bits 3 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-qmse-b3
```

Run the packed-cache `Q_mse` `NIAH` grid that should impact runtime cache memory:

```bash
modal run src/turboquant/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant qmse_packed \
  --qmse-bits 3 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-qmse-packed-b3
```

Compare the quantized `NIAH` grid against the baseline:

```bash
modal run src/turboquant/modal_app.py \
  --compare-niah-baseline niah-baseline \
  --compare-niah-candidate niah-qmse-packed-b3
```

Important caution:
the upstream `Needle In A Haystack` repo is API-oriented. Our implementation mirrors the
official protocol locally for QwQ by varying context length, insertion depth, and deterministic
retrieval scoring inside our own runtime, rather than calling the provider-specific upstream package.

For NIAH, all variants use the same manual greedy decoding path after prefill:
- `baseline`: dense KV cache
- `qmse`: reconstructed dense `TurboQuant_mse` cache
- `qmse_packed`: packed `TurboQuant_mse` cache that dequantizes one layer at a time during attention

## Notes on authenticity

This project is intentionally wired around official public sources:

- model weights: [`Qwen/QwQ-32B`](https://huggingface.co/Qwen/QwQ-32B)
- TurboQuant paper: [OpenReview](https://openreview.net/forum?id=tO3ASKZlok)
- Needle In A Haystack: [official repo](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- LongBench: [official repo](https://github.com/THUDM/LongBench)
- RULER: [official repo](https://github.com/NVIDIA/RULER)
- LVEval: [official repo](https://github.com/infinigence/LVEval)
- ZeroSCROLLS: [official repo](https://github.com/tau-nlp/zero_scrolls)
