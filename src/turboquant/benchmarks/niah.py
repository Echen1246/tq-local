from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re


_HAYSTACK_PARAGRAPHS: tuple[str, ...] = (
    "Researchers study sequence models by comparing memory use, numerical stability, and retrieval performance across prompts of different lengths.",
    "A transformer stores keys and values for previous tokens so future tokens can attend back to the relevant parts of the prompt.",
    "Quantization can target model weights, activations, or cached attention state, and each choice changes the system bottleneck in different ways.",
    "Long-context evaluation is difficult because benchmark quality depends on prompt construction, scoring rules, and the exact runtime configuration.",
    "A reliable systems experiment should record the model revision, prompt length, numerical precision, and any custom cache transformation in use.",
    "Grouped-query attention reduces the number of key-value heads relative to query heads, which changes the shape of the cache but not its conceptual role.",
    "A retrieval benchmark is most convincing when the inserted fact is unique, the question is direct, and the grading rule is deterministic.",
    "When evaluating cache compression, it is helpful to separate vector reconstruction quality from downstream attention-quality measurements.",
)


@dataclass(frozen=True)
class NeedleSpec:
    key: str
    code: str
    sentence: str
    question: str


def niah_system_prompt() -> str:
    return (
        "You are running a retrieval benchmark. "
        "Do not explain your reasoning. "
        "Return the verification code only. "
        "If you must include words, put the digits on the final line."
    )


def make_needle_spec(seed_text: str) -> NeedleSpec:
    digest = sha256(seed_text.encode("utf-8")).hexdigest()
    code = str(int(digest[:12], 16) % 9000000 + 1000000)
    key = digest[:8]
    sentence = (
        f"The special turboquant verification code for record {key} is {code}. "
        "This is the only correct verification code for that record."
    )
    question = f"What is the special turboquant verification code for record {key}?"
    return NeedleSpec(key=key, code=code, sentence=sentence, question=question)


def repeated_haystack_text() -> str:
    return "\n\n".join(_HAYSTACK_PARAGRAPHS)


def build_niah_context(
    tokenizer,
    context_length: int,
    depth_percent: float,
    needle: NeedleSpec,
) -> tuple[str, dict[str, int | float]]:
    if not 0 <= depth_percent <= 100:
        raise ValueError("depth_percent must be in [0, 100].")

    filler = repeated_haystack_text()
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("Expected non-empty filler tokenization.")

    needle_ids = tokenizer.encode(needle.sentence, add_special_tokens=False)
    if len(needle_ids) >= context_length:
        raise ValueError("Needle sentence is longer than the requested context length.")

    target_filler_tokens = context_length - len(needle_ids)
    repeated_ids: list[int] = []
    while len(repeated_ids) < target_filler_tokens:
        repeated_ids.extend(filler_ids)
    repeated_ids = repeated_ids[:target_filler_tokens]

    insertion_index = int(round((depth_percent / 100.0) * target_filler_tokens))
    insertion_index = max(0, min(insertion_index, target_filler_tokens))

    merged_ids = (
        repeated_ids[:insertion_index] + needle_ids + repeated_ids[insertion_index:]
    )
    merged_ids = merged_ids[:context_length]
    text = tokenizer.decode(merged_ids, skip_special_tokens=True)
    return text, {
        "context_length_tokens": context_length,
        "needle_token_length": len(needle_ids),
        "needle_insertion_index": insertion_index,
        "depth_percent": depth_percent,
    }


def niah_user_prompt(context: str, needle: NeedleSpec) -> str:
    return (
        "Read the context and answer one retrieval question.\n"
        "Important rules:\n"
        "1. Do not explain.\n"
        "2. Do not summarize the context.\n"
        "3. Return the verification code digits.\n"
        "4. If you include any extra text, end with a line of digits only.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {needle.question}"
    )


def extract_numeric_candidates(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def score_niah_response(response_text: str, needle: NeedleSpec) -> dict[str, object]:
    candidates = extract_numeric_candidates(response_text)
    exact_match = needle.code in candidates or needle.code in response_text
    return {
        "expected_code": needle.code,
        "predicted_candidates": candidates[:10],
        "exact_match": bool(exact_match),
    }
