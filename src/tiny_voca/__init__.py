"""tiny-voca: a small two-voice TTS trained from scratch.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config

__all__ = ["Config", "TinyVoca"]


def __getattr__(name):
    """Import the model lazily so the front end and config can be used
    (and tested) without torch installed."""
    if name == "TinyVoca":
        from .model import TinyVoca
        return TinyVoca
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
