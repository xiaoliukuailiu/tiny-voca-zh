"""Front-end selector.

The model only ever sees the symbols exported here, so choosing a language is
choosing a symbol inventory, and nothing downstream has to change.

    TINY_VOCA_LANG=en  (default) American English, eSpeak/dizione phonemes
    TINY_VOCA_LANG=zh            Mandarin, pinyin initials and toned finals

Both modules expose the same interface: ``SYMBOLS``, ``SYMBOL_TO_ID``,
``PHONE_SET``, ``PHONE_IDS``, ``VOWELS``, ``VOWEL_SET``, ``VOICELESS``,
``PUNCTUATION``, ``SP``, ``UNK``, ``Phonemized``, ``phonemize``,
``frontend_identity``, ``frontend_runtime``, ``frontend_fingerprint``,
``normalize_text`` and ``is_phone``.
"""

import os

_REQUESTED = os.environ.get("TINY_VOCA_LANG", "en").strip().lower()

if _REQUESTED in ("zh", "zh-cn", "zh_cn", "cmn", "mandarin", "chinese"):
    LANGUAGE = "zh"
    from .zh_text import *  # noqa: F401,F403
    from .zh_text import DEFAULT_SPEAKERS, NothingToPronounce, Phonemized, Syllable  # noqa: F401
else:
    LANGUAGE = "en"
    from .en_text import *  # noqa: F401,F403
    from .en_text import NothingToPronounce, Phonemized  # noqa: F401

    DEFAULT_SPEAKERS = ("F2", "M2")
