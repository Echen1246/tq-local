"""Shared defaults for generation and CLI."""

# Default cap for new tokens (CLI `run` / `attach`, session.generate, patched model.generate).
# Higher values allow longer answers; they increase decode time and peak KV size, not a breaking change.
DEFAULT_MAX_NEW_TOKENS = 1024
