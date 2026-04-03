# turboquant-modal

This repo contains two layers:

- [src/turboquant](/Users/eddie/Documents/turboquant/src/turboquant): the reusable TurboQuant-style runtime and math package
- [research](/Users/eddie/Documents/turboquant/research): the proof-of-concept research harness, benchmark workflow, and published results

Current status:

- `TurboQuant_mse` is implemented
- a packed `Q_mse` cache runtime is implemented
- the proof of concept was validated on `Qwen/QwQ-32B` with Modal and NIAH

If you want the research implementation and reproduced results, start in [research/README.md](/Users/eddie/Documents/turboquant/research/README.md).

If you want to work toward a cleaner reusable runtime, start in [src/turboquant](/Users/eddie/Documents/turboquant/src/turboquant).
