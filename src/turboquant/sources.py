from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UpstreamSource:
    name: str
    kind: str
    url: str
    notes: str


MODEL_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        name="Qwen/QwQ-32B",
        kind="model",
        url="https://huggingface.co/Qwen/QwQ-32B",
        notes="Official Hugging Face weights and metadata for the target model.",
    ),
)


PAPER_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        name="TurboQuant paper",
        kind="paper",
        url="https://openreview.net/forum?id=tO3ASKZlok",
        notes="Primary public source for the algorithm and benchmark claims.",
    ),
    UpstreamSource(
        name="QJL paper",
        kind="paper",
        url="https://arxiv.org/abs/2406.03482",
        notes="Residual inner-product estimator used inside TurboQuant_prod.",
    ),
    UpstreamSource(
        name="PolarQuant paper",
        kind="paper",
        url="https://arxiv.org/abs/2502.02617",
        notes="Nearby prior work from the same line of research.",
    ),
)


BENCHMARK_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        name="Needle In A Haystack",
        kind="benchmark",
        url="https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
        notes="Official public NIAH repo referenced by the long-context ecosystem.",
    ),
    UpstreamSource(
        name="LongBench",
        kind="benchmark",
        url="https://github.com/THUDM/LongBench",
        notes="Official public LongBench repo; LongBench-E is the paper-aligned subset.",
    ),
    UpstreamSource(
        name="LongBench dataset",
        kind="dataset",
        url="https://huggingface.co/datasets/THUDM/LongBench",
        notes="Official dataset hosting for LongBench tasks.",
    ),
    UpstreamSource(
        name="RULER",
        kind="benchmark",
        url="https://github.com/NVIDIA/RULER",
        notes="Official public repo for the RULER long-context benchmark suite.",
    ),
    UpstreamSource(
        name="LVEval",
        kind="benchmark",
        url="https://github.com/infinigence/LVEval",
        notes="Official public repo for L-Eval / LV-Eval style evaluation.",
    ),
    UpstreamSource(
        name="ZeroSCROLLS",
        kind="benchmark",
        url="https://github.com/tau-nlp/zero_scrolls",
        notes="Official public ZeroSCROLLS evaluation suite.",
    ),
)


def all_sources() -> tuple[UpstreamSource, ...]:
    return MODEL_SOURCES + PAPER_SOURCES + BENCHMARK_SOURCES


def as_serializable_sources() -> list[dict[str, str]]:
    return [asdict(source) for source in all_sources()]
