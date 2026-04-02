from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialBenchmark:
    name: str
    repo_url: str
    local_checkout: Path
    paper_aligned: bool
    notes: str


OFFICIAL_BENCHMARKS: tuple[OfficialBenchmark, ...] = (
    OfficialBenchmark(
        name="niah",
        repo_url="https://github.com/gkamradt/LLMTest_NeedleInAHaystack",
        local_checkout=Path("upstreams/LLMTest_NeedleInAHaystack"),
        paper_aligned=True,
        notes="Official public NIAH repo; first-class replication target.",
    ),
    OfficialBenchmark(
        name="longbench",
        repo_url="https://github.com/THUDM/LongBench",
        local_checkout=Path("upstreams/LongBench"),
        paper_aligned=True,
        notes="Official public LongBench repo; use LongBench-E for paper-aligned comparison.",
    ),
    OfficialBenchmark(
        name="ruler",
        repo_url="https://github.com/NVIDIA/RULER",
        local_checkout=Path("upstreams/RULER"),
        paper_aligned=False,
        notes="Official public RULER repo; useful for blog-level follow-up validation.",
    ),
    OfficialBenchmark(
        name="lveval",
        repo_url="https://github.com/infinigence/LVEval",
        local_checkout=Path("upstreams/LVEval"),
        paper_aligned=False,
        notes="Official public LVEval repo; useful for broader long-context stress tests.",
    ),
    OfficialBenchmark(
        name="zero_scrolls",
        repo_url="https://github.com/tau-nlp/zero_scrolls",
        local_checkout=Path("upstreams/zero_scrolls"),
        paper_aligned=False,
        notes="Official public ZeroSCROLLS repo; cited in the Google blog.",
    ),
)
