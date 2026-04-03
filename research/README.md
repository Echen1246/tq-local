# turboquant-modal research

Unofficial TurboQuant-style KV-cache compression for `Qwen/QwQ-32B` on Modal.

This research harness implements and evaluates the `TurboQuant_mse` path from the TurboQuant paper: normalize each KV vector, rotate it with a fixed orthogonal transform, quantize rotated coordinates with a Lloyd-Max scalar codebook, then inverse-rotate and rescale. It also includes a packed-cache runtime variant that stores quantized KV state in compressed form and dequantizes one layer at a time during attention.

We intentionally do **not** claim the full paper is reproduced. The current implementation covers the `Q_mse` path, not the full `Q_prod` residual-QJL path or a fused custom kernel.

## What is implemented

- Real `Qwen/QwQ-32B` loading on Modal with pinned Hugging Face revision.
- Reference `TurboQuant_mse` math:
  - exact coordinate density for a random point on the unit sphere
  - numerical Lloyd-Max scalar codebook construction
  - fixed seeded random orthogonal rotation
  - vector normalization and norm rescaling
- Offline KV capture and distortion analysis on real QwQ KV tensors.
- Three NIAH runtime variants:
  - `baseline`: dense KV cache
  - `qmse`: dense reconstructed `TurboQuant_mse` cache
  - `qmse_packed`: packed `TurboQuant_mse` cache that dequantizes one layer at a time during attention
- Runtime memory accounting for:
  - dense KV cache bytes
  - packed KV payload bytes
  - post-cache-setup GPU memory
  - decode-time peak GPU memory

## Algorithm

For one KV vector `x ∈ R^d`:

1. Compute the norm `||x||`.
2. Normalize to `u = x / ||x||`.
3. Apply a fixed random orthogonal rotation `R` to get `z = u R^T`.
4. Quantize each coordinate of `z` independently with a Lloyd-Max scalar quantizer learned for the coordinate distribution of a random unit vector in dimension `d`.
5. Inverse-rotate the quantized vector.
6. Multiply by the stored norm.

That is the core `Q_mse` path in [turboquant_mse.py](/Users/eddie/Documents/turboquant/src/turboquant/quantization/turboquant_mse.py).

The packed runtime stores:

- per-coordinate quantizer indices, bit-packed
- one stored norm per KV vector

That runtime path is implemented in [packed_qmse_cache.py](/Users/eddie/Documents/turboquant/src/turboquant/runtime/packed_qmse_cache.py).

## Why `Q_mse` first

The paper also describes `Q_prod`, which adds a residual 1-bit QJL sketch to improve inner-product estimation. We do **not** use that path yet, because:

- `Q_prod` is not a generic reconstructed vector format
- merging the residual correction back into a normal KV vector is known to be unsafe
- the correct `Q_prod` path wants a custom attention-score implementation

So this repo focuses on the safer, drop-in `Q_mse` path first.

## Measured results

Current tested setup:

- model: `Qwen/QwQ-32B`
- revision: `976055f8c83f394f35dbd3ab09a285a984907bd0`
- runtime: Modal `H200`, `cpu=4`, `memory=16384`
- attention backend: `sdpa`
- benchmark: local NIAH protocol aligned to the official benchmark structure

### Validation matrix

- Offline vector fidelity:
  - captured real KV tensors from `QwQ-32B`
  - ran `2/3/4`-bit `TurboQuant_mse` sweeps
  - checked cosine, MSE, and actual-query causal logit error
- Live benchmark parity:
  - `baseline` vs `qmse`
  - `baseline` vs `qmse_packed`
  - NIAH grid at `4000, 8000, 16000, 32000` tokens and `10%, 50%, 90%` depths
- Runtime memory accounting:
  - dense KV payload bytes
  - packed KV payload bytes
  - post-cache-setup GPU memory
  - decode-time peak GPU memory

### Offline KV analysis

On captured QwQ KV tensors, the `TurboQuant_mse` bit sweep behaved sensibly:

- `2-bit`: visibly lossy
- `3-bit`: promising
- `4-bit`: near-transparent

Across multiple captured prompts, `3-bit` reconstruction stayed around `~0.983` mean cosine for both keys and values, with key-side actual-query causal logit error still in a plausible range.

### Live NIAH results

For NIAH at context lengths `4000, 8000, 16000, 32000` and insertion depths `10%, 50%, 90%`:

- `baseline`: `100%` exact match
- `qmse`: `100%` exact match
- `qmse_packed`: `100%` exact match

### Runtime memory results

On the packed runtime path (`qmse_packed`, `3-bit`), the measured cache payload shrank from roughly:

- dense KV cache: `~3.98 GB`
- packed KV payload: `~778 MB`

That is about an `80.4%` reduction in KV payload size.

Preliminary runtime profiling also showed materially lower global GPU memory:

- post-cache-setup allocated memory dropped by about `7-8 GB`
- decode-time peak allocated memory dropped by about `~6.9 GB`

The payload-byte result is the strongest claim here. The global GPU-memory deltas are real profiling outputs from the packed runtime, but they should still be treated as runtime measurements rather than a final kernel-optimized number.

## What this research does **not** prove yet

- It does not implement the full `Q_prod` path.
- It does not include a fused Triton/CUDA attention kernel.
- It does not yet claim broad benchmark parity beyond NIAH.
- It does not yet claim ultra-long-context results beyond the tested runtime/model setup.

This is best viewed today as a serious research implementation and packed-cache prototype, not a finished production library.

## Research structure

- [research/modal_app.py](/Users/eddie/Documents/turboquant/research/modal_app.py): research/benchmark entrypoint for Modal
- [src/turboquant/quantization/turboquant_mse.py](/Users/eddie/Documents/turboquant/src/turboquant/quantization/turboquant_mse.py): reference `Q_mse` math
- [src/turboquant/runtime/packed_qmse_cache.py](/Users/eddie/Documents/turboquant/src/turboquant/runtime/packed_qmse_cache.py): packed cache format and layer-local dequantization
- [src/turboquant/runtime/memory_accounting.py](/Users/eddie/Documents/turboquant/src/turboquant/runtime/memory_accounting.py): cache byte accounting and GPU memory sampling
- [src/turboquant/benchmarks/niah.py](/Users/eddie/Documents/turboquant/src/turboquant/benchmarks/niah.py): local NIAH protocol
- [RESEARCH.md](/Users/eddie/Documents/turboquant/RESEARCH.md): longer-form notes and caveats

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[benchmarks,modal,dev]"
```

Authenticate Modal:

```bash
modal setup
modal token info
```

Optionally export a Hugging Face token locally:

```bash
export HF_TOKEN=...
```

## Reproduce

Warm the model snapshot cache:

```bash
modal run research/modal_app.py --prefetch-only
```

Run the baseline NIAH grid:

```bash
modal run research/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant baseline \
  --max-new-tokens 256 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-baseline
```

Run the packed `3-bit` NIAH grid:

```bash
modal run research/modal_app.py \
  --niah-grid \
  --context-lengths 4000,8000,16000,32000 \
  --depth-percents 10,50,90 \
  --variant qmse_packed \
  --qmse-bits 3 \
  --max-new-tokens 256 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --run-name niah-qmse-packed-b3
```

Compare them:

```bash
modal run research/modal_app.py \
  --compare-niah-baseline niah-baseline \
  --compare-niah-candidate niah-qmse-packed-b3
```

If you want the offline KV analysis path as well:

```bash
modal run research/modal_app.py \
  --capture-kv \
  --run-name kv-smoke \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --prompt "Summarize how KV-cache compression differs from weight quantization."
```

```bash
modal run research/modal_app.py \
  --analyze-turboquant-mse kv-smoke \
  --bits-list 2,3,4 \
  --target both
```
