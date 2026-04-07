# Remote GPU Setup

TurboQuant requires a CUDA GPU. If you don't have one locally, rent one
from a cloud provider and install TurboQuant over SSH or a web terminal.

---

## Quick setup (any provider)

```bash
# Install TurboQuant
pip install git+https://github.com/Echen1246/local-turboquant.git

# Set HF token for gated models (Llama, Gemma, etc.)
export HF_TOKEN="hf_your_token_here"

# Verify GPU detected
turboquant setup

# Run
turboquant attach
```

If the instance has limited disk, point the model cache to a large volume:

```bash
export HF_HOME=/workspace/hf-cache   # or wherever you have space
mkdir -p $HF_HOME
```

---

## Vast.ai

1. Rent a GPU instance at [vast.ai](https://vast.ai) — pick a template
   with PyTorch pre-installed (e.g. "PyTorch 2.x")
2. From the instance page, click **Jupyter Terminal → Launch Application**
3. Run in the terminal:

```bash
pip install git+https://github.com/Echen1246/local-turboquant.git
export HF_TOKEN="hf_your_token_here"
turboquant setup
turboquant attach
```

### Dependency conflicts

Some Vast.ai images ship older `torch`/`torchvision` builds that conflict.
If you see errors like `torchvision::nms does not exist`, reinstall the
full stack together:

```bash
pip install -U --no-cache-dir \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "transformers==5.3.0" \
  "accelerate==1.13.0"
pip install git+https://github.com/Echen1246/local-turboquant.git
```

### Disk space

Llama 3.1 8B needs ~16 GB of disk for model weights. If downloads fail
with `No space left on device`, set `HF_HOME` to a volume with 50+ GB:

```bash
export HF_HOME=/workspace/hf-cache
mkdir -p $HF_HOME
```

---

## RunPod

1. Create a pod at [runpod.io](https://runpod.io) with a PyTorch template
2. Open the web terminal or SSH in
3. Same install:

```bash
pip install git+https://github.com/Echen1246/local-turboquant.git
export HF_TOKEN="hf_your_token_here"
turboquant setup
turboquant attach
```

---

## Lambda Labs

1. Launch an instance at [lambdalabs.com](https://lambdalabs.com)
2. SSH in with the key they provide
3. Install:

```bash
pip install git+https://github.com/Echen1246/local-turboquant.git
export HF_TOKEN="hf_your_token_here"
turboquant setup
turboquant attach
```

If Lambda's pre-installed torch is too old, reinstall from the
[PyTorch install matrix](https://pytorch.org/get-started/locally/)
for your CUDA version.

---

## Modal (serverless)

Modal doesn't use SSH — you define a function that runs on their GPU.
The repo includes ready-made Modal scripts:

```bash
pip install modal
modal setup

# Quick test
modal run examples/modal_smoke.py \
  --model meta-llama/Llama-3.1-8B-Instruct --bits 4

# Memory benchmark
modal run examples/modal_smoke.py --memory-benchmark --prompt-tokens 32768
```

Set your HF token as a Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

---

## Google Colab

See `examples/colab_vram_test.ipynb` for a ready-made notebook. Quick
version:

```python
!pip install git+https://github.com/Echen1246/local-turboquant.git -q
import os; os.environ["HF_TOKEN"] = "hf_your_token_here"

from turboquant import TurboQuantSession
s = TurboQuantSession.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", bits=4, device_map="auto")
print(s.generate("What is KV cache compression?", max_new_tokens=128))
```

---

## Generation length

Default max tokens is 256. Override with:

- CLI: `turboquant run --max-new-tokens 1024`
- Attach: `turboquant attach --max-new-tokens 2048`
- REPL: `/tokens 2048` (changes for the current session)
- Python: `session.generate("...", max_new_tokens=1024)`

Longer generation increases decode time and KV cache size proportionally.
