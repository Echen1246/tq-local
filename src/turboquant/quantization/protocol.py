from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurboQuantMSECache:
    """A reconstructed-vector-safe representation for KV cache entries."""

    bits: int
    rotation_id: str
    codebook_id: str


@dataclass(frozen=True)
class TurboQuantProdCache:
    """
    A two-part representation for inner-product estimation.

    This is intentionally not a generic reconstructed KV vector. The QJL
    residual term must be consumed by a custom attention-score path.
    """

    bits: int
    rotation_id: str
    codebook_id: str
    sketch_id: str


def merged_prod_reconstruction_is_unsafe() -> None:
    """
    Guardrail against the known Q_prod misuse.

    TurboQuant_prod does not come with a guarantee that a merged reconstruction
    is a high-quality vector for ordinary downstream attention. The safe path is
    to keep the MSE component and the QJL residual component separate inside a
    custom score kernel.
    """

    raise RuntimeError(
        "TurboQuant_prod must not be treated as a merged drop-in KV vector. "
        "Use TurboQuant_mse for reconstructed caches, or consume the two-part "
        "TurboQuant_prod representation directly in a custom attention kernel."
    )
