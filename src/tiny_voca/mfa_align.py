"""Mandarin forced alignment through the Montreal Forced Aligner.

English aligns with a multilingual CTC phoneme recogniser (``align.py``). That
recogniser cannot label Mandarin: its vocabulary carries 84 tone-marked labels
but no third tone at all and no ``-n`` / ``-ng`` finals, so most syllables have
no label. MFA ships a Mandarin acoustic model plus a pinyin dictionary whose
phone set (initial, final+tone) is almost exactly this front end's inventory.

This module does two things and nothing else:

* ``parse_intervals`` reads the phone tier of a TextGrid.
* ``durations`` turns phone intervals into per-token frame counts on the
  acoustic grid, which is what ``tiny_voca.train`` consumes.

The conversion is deliberately explicit: MFA gives seconds, the model counts
frames of ``hop`` samples, and every token must own a whole number of frames
that sums to the utterance length.
"""

from dataclasses import dataclass
import re
import subprocess
from pathlib import Path

from .text import PHONE_SET, PUNCTUATION, SP, UNK

SILENCE = frozenset(("", "sil", "sp", "spn", "silence", "<eps>", "pau"))
DEFAULT_TIER = "phones"

# MFA's Mandarin dictionary writes the umlaut as ``v`` already, but third-party
# dictionaries differ. These fold onto the inventory before lookup.
_ALIASES = {"ü": "v", "u:": "u", "io": "iou", "iou": "iu", "uei": "ui", "uen": "un"}
_INTERVAL = re.compile(
    r"xmin\s*=\s*([-\d.eE+]+)\s*\n\s*xmax\s*=\s*([-\d.eE+]+)\s*\n\s*text\s*=\s*\"([^\"]*)\""
)
_TIER = re.compile(r"item\s*\[\d+\]:\s*(.*?)(?=item\s*\[\d+\]:|\Z)", re.S)
_NAME = re.compile(r"name\s*=\s*\"([^\"]*)\"")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def parse_intervals(text, tier=DEFAULT_TIER):
    """Read the intervals of one tier from a long or short TextGrid.

    ``tier=None`` takes the first interval tier. Interval tiers declared with a
    ``class`` other than ``IntervalTier`` are skipped.
    """
    blocks = _TIER.findall(text)
    if not blocks:
        raise ValueError("no tiers found in TextGrid")
    if tier is None:
        chosen = blocks[0]
    else:
        chosen = None
        for block in blocks:
            name = _NAME.search(block)
            if name and name.group(1) == tier:
                chosen = block
                break
        if chosen is None:
            raise ValueError(f"TextGrid has no tier named {tier!r}")
    intervals = [Interval(float(a), float(b), label) for a, b, label in _INTERVAL.findall(chosen)]
    if not intervals:
        raise ValueError("tier has no intervals")
    return intervals


def canonical_label(label, phone_set=None):
    """Map one MFA phone label onto the front-end inventory.

    Returns ``None`` for silence markers and for labels the inventory does not
    know, so the caller can report them instead of guessing.
    """
    phone_set = PHONE_SET if phone_set is None else phone_set
    value = label.strip().lower().replace(" ", "").replace("ü", "v")
    if value in SILENCE:
        return None
    if value in phone_set:
        return value
    for source, target in _ALIASES.items():
        if value.startswith(source):
            value = target + value[len(source):]
            break
    if value in phone_set:
        return value
    # ``ong1`` may arrive as ``onɡ1`` (Greek gamma) from some dictionaries.
    cleaned = value.replace("ɡ", "g").replace("ʼ", "").replace("'", "")
    return cleaned if cleaned in phone_set else None


def phone_positions(symbols, phone_set=None):
    """Indices of the tokens that must be realised as audio."""
    phone_set = PHONE_SET if phone_set is None else phone_set
    return [index for index, symbol in enumerate(symbols) if symbol in phone_set or symbol == UNK]


def pause_carrier(symbols, start, end):
    """First pause-capable token strictly between two positions, if any."""
    for index in range(start + 1, end):
        if symbols[index] in PUNCTUATION or symbols[index] == SP:
            return index
    return None


def durations(intervals, symbols, frames, hop, sample_rate, min_pause_frames=7, strict=True, phone_set=None):
    """Per-token frame counts from phone intervals.

    ``frames`` is ``len(wave) // hop``, the same grid the stored mel and pitch
    use. Frames outside every phone go to the leading/trailing boundary token or
    to the punctuation that produced the pause; a gap with no carrier stays with
    the preceding phone, which is what MFA's own silence intervals would do.
    """
    phone_set = PHONE_SET if phone_set is None else phone_set
    phones = phone_positions(symbols, phone_set)
    voiced = [interval for interval in intervals if canonical_label(interval.label, phone_set) is not None]
    if strict and len(voiced) != len(phones):
        raise ValueError(f"{len(voiced)} aligned intervals but {len(phones)} phone tokens")
    count = min(len(voiced), len(phones))
    if count < 1:
        raise ValueError("nothing to align")

    owner = [-1] * frames
    previous_end = 0
    for k in range(count):
        start = int(round(voiced[k].start * sample_rate / hop))
        end = int(round(voiced[k].end * sample_rate / hop))
        start = max(0, min(start, frames - 1))
        end = max(start + 1, min(end, frames))
        # MFA never overlaps phones; clamp anyway so a bad TextGrid cannot
        # produce non-monotonic ownership.
        start = max(start, previous_end)
        if start >= frames:
            break
        end = max(end, start + 1)
        for frame in range(start, end):
            owner[frame] = phones[k]
        previous_end = end

    first = min((i for i, value in enumerate(owner) if value >= 0), default=0)
    last = max((i for i, value in enumerate(owner) if value >= 0), default=frames - 1)
    for frame in range(first):
        owner[frame] = 0
    for frame in range(last + 1, frames):
        owner[frame] = len(symbols) - 1

    frame = 0
    while frame < frames:
        if owner[frame] >= 0:
            frame += 1
            continue
        run = frame
        while run < frames and owner[run] < 0:
            run += 1
        left = owner[frame - 1]
        right = owner[run] if run < frames else len(symbols) - 1
        carrier = pause_carrier(symbols, left, right) if right > left else None
        if carrier is not None and run - frame >= min_pause_frames:
            for index in range(frame, run):
                owner[index] = carrier
        else:
            for index in range(frame, run):
                owner[index] = left
        frame = run

    counts = [0] * len(symbols)
    for value in owner:
        counts[value] += 1
    # Every phone needs at least one frame; steal from the longest neighbour.
    for index in phones[:count]:
        if counts[index] == 0:
            donor = max(phones[:count], key=lambda i: counts[i])
            if counts[donor] <= 1:
                raise ValueError("audio is too short for this phoneme sequence")
            counts[donor] -= 1
            counts[index] += 1
    if sum(counts) != frames:
        raise ValueError("durations do not cover the acoustic frames")
    return counts


def run_mfa(corpus, dictionary, acoustic, output, jobs=4, tier=DEFAULT_TIER, clean=True):
    """Align a directory of ``<id>.wav`` + ``<id>.lab`` pairs.

    ``corpus``, ``dictionary``, ``acoustic`` and ``output`` follow the ``mfa
    align`` command line. Includes ``--single_speaker`` because every clip here
    is one sentence from one synthetic voice.
    """
    command = ["mfa", "align", str(corpus), str(dictionary), str(acoustic), str(output),
               "--single_speaker", "--jobs", str(jobs),
               "--output_format", "long_textgrid", "--quiet"]
    if clean:
        command.append("--clean")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mfa align failed: {result.stderr.strip()[-2000:]}")
    return Path(output)
