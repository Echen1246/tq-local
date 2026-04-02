from __future__ import annotations


PROMPT_SUITES: dict[str, list[str]] = {
    "science_smoke": [
        "Summarize how KV-cache compression differs from weight quantization.",
        "Explain why longer context windows increase KV-cache memory usage.",
        "Describe the difference between reconstructing vectors and preserving attention dot products.",
    ],
}


def get_prompt_suite(name: str) -> list[str]:
    if name not in PROMPT_SUITES:
        raise KeyError(f"Unknown prompt suite: {name}")
    return PROMPT_SUITES[name]
