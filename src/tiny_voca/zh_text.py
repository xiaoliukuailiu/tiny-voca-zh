"""Mandarin front end: Chinese text -> initial / final tokens carrying tone.

This mirrors ``en_text`` so the rest of the system (annotation, training,
inference) does not need to know which language it is running. Select it with
``TINY_VOCA_LANG=zh``.

Design decisions that differ from English, and why:

* **Tone lives inside the token.** A final is written ``<final><tone>``, for
  example ``a1``, ``ong2``, ``i4``, ``er5`` (5 = neutral). English keeps stress
  as a parallel feature because it is prosodic; Mandarin tone is lexical, it is
  given by the text and must reach the encoder as part of the symbol. It also
  means one token = one phone = one alignment label, which is what the
  duration layer needs.
* **No word-boundary token.** Mandarin is written without spaces. Punctuation
  carries the pauses; the model sees syllable adjacency directly.
* **Initials are separate tokens.** Consonants are short but acoustically
  distinct, and splitting them off keeps the final inventory at 35 instead of
  a few hundred whole syllables.
* **Tone sandhi is applied to the text before tokenisation**, because the audio
  we generate from that text already exhibits the sandhi: a third tone followed
  by a third tone is spoken as second + third, so it must be annotated that way.

Symbol inventory (append-only; changing it invalidates checkpoints):

    specials (5) + punctuation (12) + initials (21) + finals x 5 tones (175)
"""

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata

FRONTEND_VERSION = 1
LANGUAGE = "zh"
DEFAULT_SPEAKERS = ("ZH_F1", "ZH_M1")
# Interface parity with en_text: infer.py reads these to decide which front end
# a checkpoint wants and how the user selects one.
FRONTENDS = ("pinyin",)
FRONTEND_VARIABLE = "TINY_VOCA_LANG"

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")

INITIALS = ("b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s")
FINALS = ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "er",
          "i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
          "v", "ve", "van", "vn")
TONES = (1, 2, 3, 4, 5)
FINAL_TOKENS = tuple(final + str(tone) for final in FINALS for tone in TONES)
PHONES = INITIALS + FINAL_TOKENS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES)

# Finals are voiced by definition; the model uses this set to keep vowel frames
# voiced when the pitch tracker drops creaky ones.
VOWELS = FINAL_TOKENS
VOWEL_SET = frozenset(VOWELS)
# Unaspirated stops, aspirated stops, fricatives and affricates: the QC probe
# in the annotation layer uses this to check pitch-tracker voicing flips.
VOICELESS = frozenset(("b", "p", "d", "t", "g", "k", "f", "h",
                       "j", "q", "x", "zh", "ch", "sh", "z", "c", "s"))
VOICED_INITIALS = frozenset(("m", "n", "l", "r"))

# pinyin is written without the umlaut after these initials: ju, que, xuan, yun
_UMLAUT_INITIALS = frozenset(("j", "q", "x"))
# Syllables written with y/w are phonologically zero-initial.
_Y_W = {
    "yi": ("", "i"), "ya": ("", "ia"), "ye": ("", "ie"), "yao": ("", "iao"),
    "you": ("", "iu"), "yan": ("", "ian"), "yin": ("", "in"), "yang": ("", "iang"),
    "ying": ("", "ing"), "yong": ("", "iong"), "yo": ("", "io"),
    "yu": ("", "v"), "yue": ("", "ve"), "yuan": ("", "van"), "yun": ("", "vn"),
    "wu": ("", "u"), "wa": ("", "ua"), "wo": ("", "uo"), "wai": ("", "uai"),
    "wei": ("", "ui"), "wan": ("", "uan"), "wen": ("", "un"), "wang": ("", "uang"),
    "weng": ("", "ueng"),
}
# pinyin spelling variants that map onto one inventory final
_FINAL_ALIASES = {"iou": "iu", "uei": "ui", "uen": "un", "ü": "v", "üe": "ve", "üan": "van", "ün": "vn"}

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "–": "—", "―": "—", "－": "—", "〜": "—",
    "［": "(", "｛": "(", "【": "(", "《": "(", "「": "(", "『": "(",
    "］": ")", "｝": ")", "】": ")", "》": ")", "」": ")", "』": ")",
})
# Chinese punctuation folded onto the ASCII inventory above.
_PUNCTUATION_FOLD = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "、": ",", "．": ".", "·": "—", "～": "—", "　": " ",
}
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RUN = re.compile(r"[A-Za-z]+")

_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ("", "十", "百", "千")
_BIG_UNITS = ("", "万", "亿")


class NothingToPronounce(ValueError):
    """The text has no syllable to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


@dataclass(frozen=True)
class Syllable:
    initial: str
    final: str
    tone: int
    char: str


def _pypinyin():
    try:
        import pypinyin
    except ImportError as error:
        raise RuntimeError('the Mandarin front end needs pypinyin: pip install pypinyin') from error
    return pypinyin


def frontend():
    return "pinyin"


def set_frontend(name):
    """Kept for interface parity with the English front end; zh has one path."""
    if name not in ("pinyin", "own", "espeak"):
        raise ValueError(f"unknown front end {name!r}")


def frontend_identity():
    return {"frontend_version": FRONTEND_VERSION, "language": LANGUAGE,
            "phonemizer": "pypinyin", "style": "TONE3", "symbols": list(SYMBOLS)}


def frontend_runtime():
    return {"pypinyin_version": _pypinyin().__version__}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# text normalization
# --------------------------------------------------------------------------

def _int_to_chinese(value):
    """Read an integer the way Mandarin does, e.g. 2026 -> 两千零二十六."""
    if value == 0:
        return _DIGITS[0]
    if value < 0:
        return "负" + _int_to_chinese(-value)
    groups, rest = [], value
    while rest:
        groups.append(rest % 10000)
        rest //= 10000
    parts = []
    for power in range(len(groups) - 1, -1, -1):
        group = groups[power]
        if not group:
            continue
        text = ""
        zero = False
        for index in range(3, -1, -1):
            digit = (group // 10 ** index) % 10
            if digit == 0:
                zero = bool(text)
                continue
            if zero:
                text += _DIGITS[0]
                zero = False
            text += _DIGITS[digit] + _SMALL_UNITS[index]
        parts.append(text + _BIG_UNITS[power])
    result = _DIGITS[0].join(parts) if len(parts) > 1 and groups[0] and len(str(groups[-1])) < 4 else "".join(parts)
    # 10-19 are read "十..." and never "一十..."
    if result.startswith("一十"):
        result = result[1:]
    return result


def _digit_by_digit(digits):
    return "".join(_DIGITS[int(d)] for d in digits)


def _read_decimal(number):
    whole, _, fraction = number.partition(".")
    text = _int_to_chinese(int(whole)) if whole else _DIGITS[0]
    return text + "点" + _digit_by_digit(fraction) if fraction else text


def normalize_text(text):
    """Fold full-width forms, Chinese punctuation and numbers into the
    inventory. Numbers become the characters a speaker would actually say, so
    the front end never has to read digits."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = "".join(_PUNCTUATION_FOLD.get(char, char) for char in text)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    # percentages before plain numbers
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: "百分之" + _read_decimal(m.group(1)), text)
    # years read digit by digit
    text = re.sub(r"(\d{4})\s*年", lambda m: _digit_by_digit(m.group(1)) + "年", text)
    # decimals
    text = re.sub(r"\d+\.\d+", lambda m: _read_decimal(m.group(0)), text)
    # Long digit runs (phone numbers, IDs, account numbers) are read one digit
    # at a time; reading 13812345678 as a cardinal number would be absurd.
    text = re.sub(r"\d{7,}", lambda m: _digit_by_digit(m.group(0)), text)
    # remaining integers
    text = re.sub(r"\d+", lambda m: _int_to_chinese(int(m.group(0))), text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# pinyin -> (initial, final, tone)
# --------------------------------------------------------------------------

def split_pinyin(syllable):
    """Split one TONE3 syllable such as ``zhong1`` or ``lv4``.

    Returns ``None`` when the final is outside the inventory, which the caller
    reports as an unknown symbol rather than silently mangling it.
    """
    tone = 5
    body = syllable.replace("ü", "v")
    if body and body[-1].isdigit():
        tone, body = int(body[-1]), body[:-1]
    if not 1 <= tone <= 5:
        return None
    if body in _Y_W:
        initial, final = _Y_W[body]
    else:
        initial = ""
        for candidate in ("zh", "ch", "sh"):
            if body.startswith(candidate):
                initial = candidate
                break
        if not initial and body[:1] in tuple(i for i in INITIALS if len(i) == 1):
            initial = body[0]
        final = body[len(initial):]
        if initial in _UMLAUT_INITIALS and final.startswith("u"):
            final = "v" + final[1:]
    final = _FINAL_ALIASES.get(final, final)
    if final not in FINALS:
        return None
    return initial, final, tone


def apply_sandhi(syllables):
    """Mandarin tone sandhi, applied left to right inside one phrase.

    * 不 (bu4) before a fourth tone -> second tone
    * 一 (yi1): before 4th -> 2nd; before 1st/2nd/3rd -> 4th; otherwise 1st
    * third tone + third tone -> second + third (iterative, so 3-3-3 -> 2-2-3)
    """
    result = list(syllables)
    count = len(result)
    for index, item in enumerate(result):
        if item is None:
            continue
        following = result[index + 1] if index + 1 < count else None
        if item.char == "不" and item.tone == 4 and following and following.tone == 4:
            result[index] = Syllable(item.initial, item.final, 2, item.char)
        elif item.char == "一" and item.tone == 1:
            if following is None:
                continue
            if following.tone == 4:
                result[index] = Syllable(item.initial, item.final, 2, item.char)
            elif following.tone in (1, 2, 3):
                result[index] = Syllable(item.initial, item.final, 4, item.char)
    for index in range(count - 1):
        first, second = result[index], result[index + 1]
        if first and second and first.tone == 3 and second.tone == 3:
            result[index] = Syllable(first.initial, first.final, 2, first.char)
    return result


def phrase_syllables(phrase):
    """Phonemize one run of text between punctuation marks."""
    pypinyin = _pypinyin()
    items = pypinyin.pinyin(phrase, style=pypinyin.Style.TONE3,
                            neutral_tone_with_five=True, heteronym=False)
    syllables = []
    for index, item in enumerate(items):
        value = item[0]
        char = phrase[index] if index < len(phrase) else "?"
        if not _CJK.match(char) or not value or not value[-1].isdigit():
            syllables.append(None)
            continue
        parts = split_pinyin(value)
        if parts is None:
            syllables.append(None)
            continue
        initial, final, tone = parts
        syllables.append(Syllable(initial, final, tone, char))
    return apply_sandhi(syllables)


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

def tokenize_with_owners(pieces):
    """Tokenise and say, for every token, which syllable it came from.

    ``owners[i]`` is the index of the syllable behind ``symbols[i]``, or ``-1``
    for punctuation and the boundary tokens. The aligner needs this: it learns
    where each character lies in the audio and must hand exactly those frames to
    that character's one or two tokens.
    """
    symbols, owners, unknown = [BOS], [-1], []
    syllable = -1
    for kind, value in pieces:
        if kind == "punct":
            if symbols[-1] != value:
                symbols.append(value)
                owners.append(-1)
            continue
        syllable += 1
        if value.initial:
            symbols.append(value.initial)
            owners.append(syllable)
        token = value.final + str(value.tone)
        if token in SYMBOL_TO_ID:
            symbols.append(token)
        else:
            unknown.append(token)
            symbols.append(UNK)
        owners.append(syllable)
    symbols.append(EOS)
    owners.append(-1)
    if not any(symbol in PHONE_SET or symbol == UNK for symbol in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                      tuple(0 for _ in symbols), tuple(unknown)), tuple(owners)


def tokenize(pieces):
    """Build the token sequence from ``(kind, value)`` pieces.

    ``kind`` is ``"punct"`` (a mark) or ``"syllable"`` (a ``Syllable``).
    Pure function: testable without audio and without pypinyin.
    """
    return tokenize_with_owners(pieces)[0]


def pieces(text):
    """Split normalized text into syllable and punctuation pieces."""
    output = []
    buffer = []

    def flush():
        if buffer:
            phrase = "".join(buffer)
            for syllable in phrase_syllables(phrase):
                output.append(("syllable", syllable) if syllable else ("punct", ""))
            buffer.clear()

    for char in text:
        if char in _PUNCTUATION_CHARS:
            flush()
            output.append(("punct", char))
        elif _CJK.match(char):
            buffer.append(char)
        elif char.isspace():
            flush()
        elif _LATIN_RUN.match(char):
            # Latin runs have no Mandarin reading here; they are reported, not guessed.
            flush()
            output.append(("punct", ""))
        else:
            continue
    flush()
    return [item for item in output if not (item[0] == "punct" and item[1] == "")]


def phonemize_with_owners(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    normalized = normalize_text(text)
    if not normalized:
        raise NothingToPronounce("the text has nothing to pronounce")
    return tokenize_with_owners(pieces(normalized))


def phonemize(text):
    return phonemize_with_owners(text)[0]


def syllables_of(text):
    """The ``Syllable`` objects of ``text`` in order, matching token owners."""
    return [value for kind, value in pieces(normalize_text(text)) if kind == "syllable"]


def is_phone(token_ids):
    return [int(i) in PHONE_IDS for i in token_ids]
