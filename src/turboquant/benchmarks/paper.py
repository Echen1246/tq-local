from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PaperBenchmarkSpec:
    name: str
    official_repo: str
    local_checkout: str
    dataset_or_suite: str
    paper_aligned: bool
    notes: str


PAPER_BENCHMARKS: tuple[PaperBenchmarkSpec, ...] = (
    PaperBenchmarkSpec(
        name="niah",
        official_repo="https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
        local_checkout="upstreams/LLMTest_NeedleInAHaystack",
        dataset_or_suite="Needle In A Haystack",
        paper_aligned=True,
        notes="Paper-faithful retrieval benchmark for long-context recall.",
    ),
    PaperBenchmarkSpec(
        name="longbench_e",
        official_repo="https://github.com/THUDM/LongBench",
        local_checkout="upstreams/LongBench",
        dataset_or_suite="THUDM/LongBench (LongBench-E subset)",
        paper_aligned=True,
        notes="Paper-faithful task benchmark for long-context generation and retrieval.",
    ),
)


def as_serializable_benchmarks() -> list[dict[str, str | bool]]:
    return [asdict(spec) for spec in PAPER_BENCHMARKS]
