"""Character-level forced alignment: the Windows-native alternative to MFA.

Montreal Forced Aligner has no Windows build, and the multilingual eSpeak
recogniser used for English has no usable Mandarin coverage (no third tone, no
``-n``/``-ng`` finals). This module aligns with a Chinese wav2vec2 character
model instead:

1. CTC-align the **characters** of the sentence to the audio. One character is
   exactly one syllable, which is the unit the front end produces.
2. Split each character's frames between its initial and its final.

Splitting is the only approximation here, and the character's span is already
known before it happens:

* zero initial -> the final owns the whole span;
* voiceless initial -> the boundary is where the pitch tracker's voicing turns
  on inside the span (the consonant is voiceless, the final is voiced);
* otherwise -> a fraction that depends on the class of the initial.

The fractions are coarse on purpose: they only place a boundary inside a
syllable, and the duration predictor learns whatever convention the labels use.
"""

from dataclasses import dataclass
import math

import numpy as np

from .text import PUNCTUATION, SP

ALIGNER_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
ALIGNER_RATE = 16000
ALIGNER_STRIDE = 320          # samples at 16 kHz: 20 ms per frame

# Voiceless initials: the vowel after them is where voicing turns on, which is
# what locates the initial/final boundary inside a syllable. Kept local rather
# than imported from the front end so this module is testable in either
# language configuration.
VOICELESS_INITIALS = frozenset(("b", "d", "g", "p", "t", "k", "z", "c", "zh", "ch",
                                "j", "q", "f", "s", "sh", "x", "h"))

# How much of a syllable's span the initial takes when the voicing flip cannot
# be found. Stops are the shortest, fricatives the longest.
STOP = frozenset(("b", "d", "g", "p", "t", "k"))
AFFRICATE = frozenset(("z", "c", "zh", "ch", "j", "q"))
FRICATIVE = frozenset(("f", "s", "sh", "x", "h"))
SONORANT = frozenset(("m", "n", "l", "r"))
INITIAL_FRACTION = {STOP: 0.22, AFFRICATE: 0.30, FRICATIVE: 0.42, SONORANT: 0.35}
MIN_VOICED_RUN = 2            # frames of voicing before a final is believed


def initial_fraction(initial):
    for group, fraction in INITIAL_FRACTION.items():
        if initial in group:
            return fraction
    return 0.30


def split_syllable(start, end, initial, voiced=None):
    """Frames for ``(initial, final)`` inside one character's span.

    ``end`` is exclusive. Returns ``(0, length)`` when there is no initial.
    """
    length = max(0, end - start)
    if length == 0:
        return 0, 0
    if not initial:
        return 0, length
    if length == 1:
        return 1, 0
    boundary = None
    if initial in VOICELESS_INITIALS and voiced is not None and len(voiced) >= end:
        run = 0
        for frame in range(start, end):
            run = run + 1 if voiced[frame] else 0
            if run >= MIN_VOICED_RUN:
                candidate = frame - MIN_VOICED_RUN + 1
                if start + 1 <= candidate <= end - 1:
                    boundary = candidate
                break
    if boundary is None:
        width = max(1, min(length - 1, int(round(length * initial_fraction(initial)))))
        boundary = start + width
    return boundary - start, end - boundary


def _pause_carrier(owners, symbols, left, right):
    """Token index of a punctuation mark sitting between two syllables."""
    left_positions = [index for index, owner in enumerate(owners) if owner == left]
    right_positions = [index for index, owner in enumerate(owners) if owner == right]
    if not left_positions or not right_positions:
        return None
    for index in range(left_positions[-1] + 1, right_positions[0]):
        if symbols[index] in PUNCTUATION or symbols[index] == SP:
            return index
    return None


def durations_from_chars(symbols, owners, spans, voiced, frames, min_pause_frames=7):
    """Per-token frame counts from per-syllable spans.

    ``spans`` holds ``(start, end)`` in acoustic frames for every syllable, in
    order. Frames outside a syllable go to the leading/trailing boundary token,
    or to the punctuation that produced the pause; a gap with no punctuation
    carrier stays with the previous token.
    """
    if len(spans) != max(owners) + 1:
        raise ValueError(f"{len(spans)} syllable spans for {max(owners) + 1} syllables")

    owner = [-1] * frames
    for syllable, (start, end) in enumerate(spans):
        positions = [index for index, value in enumerate(owners) if value == syllable]
        start = max(0, min(int(start), frames))
        end = max(start, min(int(end), frames))
        if end <= start:
            raise ValueError(f"syllable {syllable} has no frames")
        if len(positions) == 1:
            for frame in range(start, end):
                owner[frame] = positions[0]
            continue
        head, tail = split_syllable(start, end, symbols[positions[0]], voiced)
        for frame in range(start, start + head):
            owner[frame] = positions[0]
        for frame in range(start + head, end):
            owner[frame] = positions[1]

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
        left_syllable = owners[left] if left < len(owners) else -1
        right_syllable = owners[right] if right < len(owners) else -1
        carrier = None
        if left_syllable >= 0 and right_syllable >= 0:
            carrier = _pause_carrier(owners, symbols, left_syllable, right_syllable)
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
    for index, symbol in enumerate(symbols):
        # A token belongs to a syllable iff the front end gave it an owner.
        if counts[index] == 0 and owners[index] >= 0:
            donor = max(range(len(counts)), key=lambda i: counts[i])
            if counts[donor] <= 1:
                raise ValueError("audio is too short for this phoneme sequence")
            counts[donor] -= 1
            counts[index] += 1
    if sum(counts) != frames:
        raise ValueError("durations do not cover the acoustic frames")
    return counts


class CharacterAligner:
    """CTC character aligner. Loads a Chinese wav2vec2 checkpoint locally."""

    def __init__(self, device="cuda", name=ALIGNER_MODEL, local_files_only=False):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch, self.device = torch, torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(name, local_files_only=local_files_only)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, local_files_only=local_files_only).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.vocabulary = self.processor.tokenizer.get_vocab()
        self.blank = int(self.model.config.pad_token_id)
        self.name = name

    def missing(self, characters):
        return sorted({character for character in characters if character not in self.vocabulary})

    def log_probs(self, wave, sample_rate):
        from scipy.signal import resample_poly

        torch = self.torch
        factor = math.gcd(sample_rate, ALIGNER_RATE)
        audio = resample_poly(np.asarray(wave, dtype=np.float64),
                              ALIGNER_RATE // factor, sample_rate // factor)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with torch.inference_mode():
            values = torch.from_numpy(audio.astype(np.float32))[None].to(self.device, self.model.dtype)
            return self.model(values).logits[0].float().log_softmax(-1).cpu().numpy()

    def spans(self, wave, sample_rate, characters, hop, acoustic_rate):
        """``(start, end)`` in acoustic frames for every character."""
        import torch
        from torchaudio.functional import forced_align

        missing = self.missing(characters)
        if missing:
            raise ValueError(f"characters not in the aligner vocabulary: {missing[:5]}")
        labels = [self.vocabulary[character] for character in characters]
        log_probs = self.log_probs(wave, sample_rate)
        repeats = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a == b)
        if log_probs.shape[0] < len(labels) + repeats:
            raise ValueError("audio is too short for this text")

        path, _ = forced_align(torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))[None],
                               torch.tensor([labels], dtype=torch.long), blank=self.blank)
        spans, previous = [], self.blank
        for frame, label in enumerate(path[0].tolist()):
            if label != self.blank and (previous == self.blank or label != previous):
                spans.append([frame, frame])
            elif label != self.blank:
                spans[-1][1] = frame
            previous = label
        if len(spans) != len(labels):
            raise ValueError("forced alignment did not visit every character")

        # Aligner frames are 20 ms at 16 kHz; the model counts hop/sample_rate.
        seconds_per_aligner_frame = ALIGNER_STRIDE / ALIGNER_RATE
        converted = []
        for start, end in spans:
            start_frame = int(round(start * seconds_per_aligner_frame * acoustic_rate / hop))
            end_frame = int(round((end + 1) * seconds_per_aligner_frame * acoustic_rate / hop))
            converted.append((max(0, start_frame), end_frame))
        return converted
