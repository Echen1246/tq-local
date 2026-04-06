# Model Compatibility Log

## Tested Models

### Llama 3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`)

**Status: Fully working**

- Key norms are moderate and uniform across all 32 layers (max ~20, no outliers)
- Norm guard keeps 0/32 layers dense — all layers quantize cleanly
- 3-bit Q_mse: key cosine sim 0.983, value cosine sim 0.983 — coherent output
- 4-bit Q_mse: key cosine sim 0.995, value cosine sim 0.995 — near-lossless
- 3-bit Q_prod (QJL keys + quantize decode): key cosine sim 0.920 — works, slightly lower quality
- Memory savings: 80% KV payload reduction at 3-bit

**Long-context results (3-bit Q_mse, chunked attention):**

| Prompt tokens | Dense KV | Packed KV | Peak VRAM overhead | Savings |
|:---:|:---:|:---:|:---:|:---:|
| 5K | 477 MB | 93 MB | 161 MB (vs 499 MB baseline) | 67.6% |
| 36K | 4.77 GB | 932 MB | 999 MB (vs 4.86 GB baseline) | 79.5% |
| 73K | 9.54 GB | 1.86 GB | 1.94 GB (vs 9.71 GB baseline) | 80.1% |

At 73K tokens, TurboQuant saves 7.8 GB of peak VRAM — the difference between
fitting on a 24 GB consumer GPU vs needing 32 GB+.

### Qwen 2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct`)

**Status: Works with norm guard (3 layers kept dense)**

- Qwen 2.5 has pathologically high key norms in specific layers:
  - Layer 0: key_mean_norm = 273.7
  - Layer 1: key_mean_norm = 66.3
  - Layer 27: key_mean_norm = 239.5
  - Normal layers (e.g., Layer 10): key_mean_norm = 16.8
- These extreme norms cause catastrophic Q_prod failure without norm guard
  because QJL logit variance scales with norm² — at norm 239, logit std ~4.5
  swamps typical attention logits of O(10)
- Norm guard automatically detects these layers and keeps them dense (3/28)
- With norm guard: 3-bit Q_prod produces coherent output, 71.5% savings
- Q_mse (without QJL) also works well with norm guard
- Note: This method was explicitly NOT used in the paper but is a temporary stop-gap until I can find more root cause as to why Qwen models sputter with TQ, while it works for Llama. 

**Why the paper didn't hit this:** the paper tested on Llama-3.1-8B and
Ministral-7B, which don't have Qwen's "massive activation" pathology.

## Architecture Independence

TurboQuant operates on the KV cache after projection and RoPE — it is
architecture-agnostic in principle. The norm guard handles model-specific
activation pathologies automatically. No per-model code paths are needed;
the same configuration works across model families.

## Known Issues

- **Per-channel outlier splitting makes Q_prod worse**: splitting 128 dims into
  32 outlier + 96 normal concentrates energy in fewer QJL dimensions, increasing
  variance. Recommend `num_outlier_channels=0` for Q_prod.
- **Decode speed tradeoff**: tile-parallel fused Triton kernel is ~2.3x slower
  than fused CUDA SDPA at 128 generated tokens (8K context). Steady-state
  per-token overhead is ~1.85x (37ms vs 20ms); the headline 2.3x includes
  ~2s of one-time JIT compilation (cached by Triton for subsequent runs).
- **Long context untested beyond 80K tokens** — will use NIAH / LongBench to
  verify output integrity at 100K+ with 99.5% cosine sim at 4-bit.

**Triton kernel benchmark (4-bit Q_mse, Llama 3.1-8B, 8K prompt, B200):**

| Tokens generated | Baseline | TurboQuant | Slowdown | VRAM savings |
|:---:|:---:|:---:|:---:|:---:|
| 64 | 1.75–1.89s | 4.34–4.67s | 2.5x | 73.6% |
| 128 | 2.74–3.04s | 6.69–6.80s | 2.3x | 72.8% |

Output text is identical between baseline and TurboQuant in all tests.

## TODO

### Triton kernel (shipped — optimization saturated)

- [x] **Tile-parallel fused Triton attention kernel** — reads packed K+V
  indices directly, computes online softmax without materializing keys or
  values. Grid = (n_tiles, Q_heads). Per-position cost O(D), rotation deferred
  to single matmul after kernel. **Shipped.**

**Alternatives tested and rejected (all worse than baseline TILE_N=64):**

| Variant | Result | Why it lost |
|:---|:---|:---|
| TILE_N=128 | 7.64s @ 128 tok | Register pressure from [128,128] tiles |
| TILE_N=32 | worse occupancy | Too many programs, per-program setup overhead |
| num_warps=8 | 7.06s @ 128 tok | Warp scheduling overhead outweighs benefits |
| GQA-fused kernel | 5.16s @ 64 tok | Low occupancy (1144 programs vs 4576); K+V kept in registers across group loop causes spilling |
| @triton.autotune | 12.05s @ 64 tok | Explores 12 configs on cold start — devastating for fresh containers |
| Triton reduction kernel | 4.67s @ 64 tok | Its own JIT cost offsets the savings from fewer PyTorch ops |

**Remaining overhead breakdown (per-token, per-layer):**
- Tile attention kernel: ~0.3ms (bit-unpack ALU + codebook gather)
- Pre/post rotation matmuls: ~0.1ms each (q@R^T, out@R)
- Dense buffer merge: ~0.1ms
- Python wrapper overhead: ~0.2ms
- Total: ~0.8ms × 32 layers = ~26ms per token (vs ~14ms baseline)

The irreducible gap vs cuDNN SDPA is the bit-unpacking ALU and codebook
gather — work that a dense attention kernel simply doesn't need to do.
Further gains likely require a custom CUDA kernel (not Triton).

### Model testing

- [ ] **DeepSeek-R1** — MLA (Multi-head Latent Attention) uses a compressed
  latent KV representation. Need to verify TurboQuant interacts correctly with
  MLA's already-compressed KV structure. DeepSeek may have its own norm
  pathologies similar to Qwen.
- [ ] **DeepSeek-V3** — same MLA architecture, different scale
- [ ] **Gemma 2 / Gemma 3** — Google's model family, mentioned in the TurboQuant
  blog post but not in the paper. Likely well-behaved norms.
- [ ] **Ministral-7B** — paper's other validated model, should work cleanly
- [ ] **Llama 3.3-70B** — test at scale where KV cache dominates VRAM
- [ ] **Phi-4** — Microsoft's small model, different architecture patterns

### Benchmarks

- [ ] LongBench / LongBench-E at scale for quality validation
- [ ] NIAH grid at 100K+ tokens (paper tests up to 104K)
- [ ] RULER (long-context stress test beyond simple retrieval)

### Features

- [ ] Adaptive per-layer bit allocation (more bits for high-norm layers instead
  of binary dense/quantized)
- [ ] Streaming quantization (quantize during prefill rather than after)
