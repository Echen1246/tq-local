# TurboQuant research notes

## Scope

This note captures the public-paper understanding we should treat as ground truth before implementing anything for `Qwen/QwQ-32B`.

Date checked: 2026-03-30

## Bottom line

TurboQuant is worth replicating, but the most important implementation detail is easy to get wrong:

- `TurboQuant_mse` is the safe drop-in reconstruction path when we need an actual reconstructed KV vector.
- `TurboQuant_prod` is an inner-product estimator first, not a high-fidelity vector reconstruction method.
- If we materialize the residual-QJL correction as a single merged vector and then use that merged vector as a normal KV cache entry inside ordinary attention, we are relying on guarantees the theorem does not provide.

This is the likely root of the "garbage outputs after merging the QJL correction back into the cache" failure mode reported in custom implementations.

## Public-paper math

### 1. `TurboQuant_mse`

The public paper defines the MSE-optimized method as:

1. Normalize `x` to unit norm and store the original `||x||_2` separately.
2. Apply a random orthogonal rotation `Pi` to get `y = Pi x`.
3. Quantize each coordinate `y_j` independently with an optimal scalar Lloyd-Max quantizer for the rotated-coordinate distribution.
4. Dequantize by looking up centroids and rotating back with `Pi^T`.
5. Rescale by the stored original norm.

Why this works:

- After random rotation, each coordinate has a known Beta-distributed marginal on the sphere.
- In high dimension, the coordinates become close to independent.
- That lets them replace hard vector quantization with per-coordinate scalar quantization while staying near the information-theoretic distortion rate.

What the theorem guarantees:

- Near-optimal MSE rate.
- Small residual `r = x - x_hat_mse`.

What this means in practice:

- `TurboQuant_mse` is the right tool if we care about vector reconstruction quality, cosine similarity, or plugging reconstructed tensors into an existing attention stack without changing the math of attention itself.

### 2. `TurboQuant_prod`

The inner-product method is defined as:

1. Run `TurboQuant_mse` at bit-width `b - 1`.
2. Compute the residual `r = x - x_hat_mse`.
3. Apply QJL to the residual: `qjl = sign(S r)`, where `S` is a random Gaussian/JL matrix.
4. Store `(idx_mse, qjl, ||r||_2)`.

The paper's dequantization formula is conceptually:

`x_hat_prod = x_hat_mse + sqrt(pi / 2) / d * ||r||_2 * S^T qjl`

What the theorem actually guarantees:

- For any query vector `y`, `E[<y, x_hat_prod>] = <y, x>`.
- The inner-product variance is bounded.

What the theorem does NOT guarantee:

- Low MSE of `x_hat_prod` as a vector.
- High cosine similarity between `x` and `x_hat_prod`.
- Stability after softmax.
- Good behavior when the same one-shot noisy reconstruction is reused as a plain vector in nonlinear downstream operations.

This distinction matters.

## Why `Q_prod` is dangerous as a naive drop-in KV cache

The "known issue" description is directionally correct and matches the public math.

### Core reason

`TurboQuant_prod` is optimized so that the linear statistic `<y, x_hat_prod>` is unbiased in expectation over quantization randomness.

Attention is not just a linear statistic:

1. key logits are noisy estimates
2. logits go through softmax, which is nonlinear
3. value vectors are then mixed using those noisy softmax weights

Unbiased logits do not imply unbiased softmax outputs, and they definitely do not imply a good reconstructed key/value vector.

### What goes wrong if we merge the QJL residual into a stored vector

If we explicitly materialize

`x_hat_prod = x_hat_mse + qjl_correction`

and then store that as if it were a normal reconstructed KV tensor:

- the QJL term behaves like dense correction noise in the vector space
- this can badly hurt cosine similarity / reconstruction quality
- the noisy key vector perturbs many logits at once
- softmax can amplify those perturbations
- the same one-shot noisy realization is reused for all future queries

So even though `<q, x_hat_prod>` is unbiased in expectation for a fixed query `q`, the resulting attention distribution and generated text can still degrade sharply.

### Safer interpretation

Use `TurboQuant_prod` only when the kernel directly consumes the two-part representation:

- MSE part: standard LUT-style contribution from `Pi q`
- residual-QJL part: direct query-dependent correction using `S q`

This is also how the paper's implementation appendix describes efficient use for search and KV-cache kernels: compute the two contributions directly rather than treating the whole thing as a generic "better reconstructed vector."

## Practical rule for our project

### Use `TurboQuant_mse` when

- we want a drop-in cache format first
- we are validating quantization quality offline
- we are testing cosine similarity, MSE, or perplexity sensitivity
- we are integrating with an existing attention kernel that expects reconstructed vectors

### Use `TurboQuant_prod` when

- we are building a custom attention kernel
- the kernel can separately handle the MSE part and the QJL residual part
- we are explicitly targeting attention-logit estimation rather than generic vector reconstruction

## Important public-source discrepancy

The public ICLR 2026 paper and the March 24, 2026 Google Research blog do not describe exactly the same public benchmark scope.

### Public paper explicitly shows

- distortion experiments on DBpedia / OpenAI embeddings
- Needle-In-A-Haystack
- LongBench-E / LongBench-V1 tables
- nearest-neighbor search experiments

Models shown in the paper's KV-cache sections:

- `Llama-3.1-8B-Instruct`
- `Ministral-7B-Instruct`

### Google Research blog additionally claims

- LongBench
- Needle In A Haystack
- ZeroSCROLLS
- RULER
- L-Eval

and says those were run on open models including Gemma and Mistral.

At the time of checking, those extra benchmark details do not appear in the public paper PDF.

Interpretation:

- The paper is the safer source for what we can reproduce exactly from public material.
- The blog may reflect additional internal or unpublished experiments.
- If we want to validate the public claims first, we should start with the paper-level benchmark set.

## Benchmarks we should use

### Priority 1: exact public-paper replication

1. Needle In A Haystack
2. LongBench / LongBench-E

Why:

- these are explicitly described in the public paper
- they are the clearest starting point for reproducing the public claims
- they cover both retrieval and end-to-end generation

### Priority 2: blog-level follow-up

3. RULER
4. L-Eval
5. ZeroSCROLLS

Why:

- these appear in the Google blog
- they are strong long-context industry benchmarks
- they help test whether the stronger marketing claims transfer beyond the paper's published tables

## Official benchmark tools / datasets

### Needle In A Haystack

Official repo:

- `https://github.com/gkamradt/LLMTest_NeedleInAHaystack`

Notes:

- original public NIAH framework
- measures retrieval across context length and depth percent
- the TurboQuant paper says it follows Fu et al. (2024) and tests 4k to 104k tokens on `Llama-3.1-8B-Instruct`

### LongBench

Official repo:

- `https://github.com/THUDM/LongBench`

Dataset:

- `https://huggingface.co/datasets/THUDM/LongBench`

Notes:

- the paper says it uses the more uniformly distributed `LongBench-E` subset for fairness across context lengths
- the public table reports task groups like SingleQA, MultiQA, Summarization, Few-shot, Synthetic, and Code

### RULER

Official repo:

- `https://github.com/NVIDIA/RULER`

Notes:

- not shown in the public TurboQuant paper PDF, but cited in the Google blog
- useful to stress-test "real context size" beyond simple NIAH retrieval

### L-Eval

Official repo:

- `https://github.com/infinigence/LVEval`

Notes:

- also blog-level, not visible in the public paper PDF
- useful for broader long-context evaluation if we decide to chase the blog claims

### ZeroSCROLLS

Official repo:

- `https://github.com/tau-nlp/zero_scrolls`

Notes:

- also blog-level, not visible in the public paper PDF
- useful for long-input summarization / QA style robustness checks

## Implications for `Qwen/QwQ-32B`

Public `QwQ-32B` details that appear stable across sources:

- 64 layers
- 40 query heads
- 8 KV heads
- hidden size 5120
- 128-dim per KV head

Important source inconsistency as of 2026-03-30:

- the current Hugging Face model card claims "Full 131,072 tokens" and says to enable YaRN above 8,192 tokens
- the current `config.json` on `main` shows `max_position_embeddings: 40960` and `sliding_window: 32768`
- an earlier `QwQ-32B` config commit showed `max_position_embeddings: 131072`
- a March 6, 2025 Hugging Face discussion argued the YaRN warning was stale and should be removed

So we should not treat the long-context configuration as fully settled from metadata alone.

This means:

- TurboQuant-style KV compression should transfer architecturally
- the KV-cache memory savings are potentially very large
- but context extension and KV compression are separate issues
- TurboQuant can reduce KV memory pressure; it does not itself solve RoPE extrapolation or YaRN configuration mistakes

For `QwQ-32B`, we should treat long-context setup as:

1. pin an exact model revision and record its hash
2. decide whether to trust native long context or explicitly apply YaRN
3. verify the runtime's effective long-context behavior with a small controlled test
4. then run KV-cache compression experiments

If we skip this, we could easily blame TurboQuant for failures that actually come from a mismatched QwQ context setup.

## Approximate KV-cache scale for `QwQ-32B`

Using the stable architecture numbers above:

- 64 layers
- 8 KV heads
- 128 dims per KV head
- K and V both stored
- bf16/fp16 uses 2 bytes per scalar

Per token KV memory is approximately:

`64 * 8 * 128 * 2 * 2 = 262,144 bytes`

That is about `256 KiB / token`.

Rough total KV size:

- 32k tokens -> about 8 GiB
- 64k tokens -> about 16 GiB
- 128k tokens -> about 32 GiB

So the memory pressure TurboQuant is trying to solve is very real for `QwQ-32B`.

## Recommended implementation order

1. Build `TurboQuant_mse` first as an offline Python reference.
2. Validate distortion, cosine, and attention-logit error on saved QwQ KV tensors.
3. Add a drop-in KV cache experiment using only `Q_mse`.
4. Only after that, implement `Q_prod` for keys inside a custom attention-score kernel.
5. Keep values on `Q_mse` initially unless we have evidence that `Q_prod` helps there without destabilizing outputs.

This order is much less risky than trying to land full `Q_prod` end-to-end in the first pass.

## Why this should be a normal Python project, not a notebook

The work naturally splits into:

- math/reference implementation
- packing/layout logic
- kernel path
- benchmark runners
- result collection

That is exactly the shape where notebooks become hard to debug and hard to trust.

Notebooks are still fine for:

- checking Lloyd-Max codebooks
- plotting distortion curves
- one-off KV tensor analysis

But the main project should be a normal Python package with reproducible scripts.

## Sources

- TurboQuant paper: `https://openreview.net/forum?id=tO3ASKZlok`
- Google Research blog, March 24, 2026: `https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/`
- QJL paper: `https://arxiv.org/abs/2406.03482`
- QJL official repo: `https://github.com/amirzandieh/QJL`
- PolarQuant paper: `https://arxiv.org/abs/2502.02617`
- QwQ-32B model card: `https://huggingface.co/Qwen/QwQ-32B`
- QwQ-32B current config: `https://huggingface.co/Qwen/QwQ-32B/blob/main/config.json`
- QwQ-32B older 131k config commit: `https://huggingface.co/Qwen/QwQ-32B/blob/b6306e1ddff8bc0d2cd2b50011968d6257039e13/config.json`
- QwQ-32B discussion about stale long-context note: `https://huggingface.co/Qwen/QwQ-32B/discussions/28`
- QwQ-32B commit removing long-context warning text: `https://huggingface.co/Qwen/QwQ-32B/commit/5479a1e0c9d8da0a47b8169fc37ae55a62e46227`
- LongBench official repo: `https://github.com/THUDM/LongBench`
- LongBench dataset: `https://huggingface.co/datasets/THUDM/LongBench`
- Needle In A Haystack official repo: `https://github.com/gkamradt/LLMTest_NeedleInAHaystack`
- RULER official repo: `https://github.com/NVIDIA/RULER`
- LV-Eval official repo: `https://github.com/infinigence/LVEval`
- ZeroSCROLLS official repo: `https://github.com/tau-nlp/zero_scrolls`
