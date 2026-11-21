"""Boundary refinement with a small frame classifier trained from scratch.

The pretrained recogniser works on 20 ms frames and its CTC emissions are
sparse, so phone boundaries from it are coarse. This module trains a tiny
convolutional phone classifier on OUR recordings (random initialisation, labels
from the CTC alignment), then moves each boundary by at most ``band`` acoustic
frames to where that classifier sees the change. It never adds or removes a
pause and never reorders tokens. Like the recogniser it is annotation-only.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .text import PHONES, PHONE_SET

SILENCE_CLASS = len(PHONES)
OTHER_CLASS = len(PHONES) + 1
CLASSES = len(PHONES) + 2
_PHONE_CLASS = {phone: index for index, phone in enumerate(PHONES)}


def symbol_class(symbol):
    if symbol in PHONE_SET:
        return _PHONE_CLASS[symbol]
    return OTHER_CLASS if symbol == "<unk>" else SILENCE_CLASS


class FrameClassifier(nn.Module):
    """Receptive field of 11 acoustic frames: it labels what a frame sounds
    like, with too little context to learn a systematic timing offset."""

    def __init__(self, n_mels, speakers, channels=256):
        super().__init__()
        self.speaker = nn.Embedding(speakers, channels)
        self.input = nn.Conv1d(n_mels, channels, 5, padding=2)
        self.blocks = nn.ModuleList(nn.Conv1d(channels, channels, 3, padding=1) for _ in range(3))
        self.output = nn.Conv1d(channels, CLASSES, 1)

    def forward(self, mel, speaker):
        x = self.input(mel.transpose(1, 2)) + self.speaker(speaker).transpose(1, 2)
        for block in self.blocks:
            x = x + block(F.gelu(x))
        return self.output(F.gelu(x)).transpose(1, 2)


def frame_labels(symbols, durations):
    return np.repeat([symbol_class(s) for s in symbols], durations)


def banded_viterbi(scores, starts, ends, band):
    """Re-time a fixed state sequence. ``scores`` is [frames, states]; state s
    originally covered ``starts[s]:ends[s]`` and may move by ``band`` frames."""
    frames, states = scores.shape
    low = np.maximum(np.asarray(starts) - band, 0)
    high = np.minimum(np.asarray(ends) + band, frames)
    low[0], high[-1] = 0, frames
    time = np.arange(frames)[:, None]
    scores = np.where((time >= low[None]) & (time < high[None]), scores, -np.inf)
    previous = np.full(states, -np.inf)
    previous[0] = scores[0, 0]
    advanced = np.zeros((frames, states), dtype=bool)
    for t in range(1, frames):
        shifted = np.concatenate(([-np.inf], previous[:-1]))
        advanced[t] = shifted > previous
        previous = np.where(advanced[t], shifted, previous) + scores[t]
    if not np.isfinite(previous[-1]):
        raise ValueError("no path inside the band")
    durations = np.zeros(states, dtype=np.int64)
    state = states - 1
    for t in range(frames - 1, -1, -1):
        durations[state] += 1
        if advanced[t, state]:
            state -= 1
    if state != 0 or np.any(durations < 1):
        raise ValueError("invalid refined path")
    return durations


def refine(items, config, device="cuda", iterations=2, band=4, steps=3000, batch=256, window=128,
           prior_scale=0.5, seed=20261113, log=print):
    """Refine durations for ``items``: dicts with ``symbols``, ``durations``,
    ``mel`` (float tensor [frames, n_mels]), ``speaker`` (int) and ``group``
    (hex text hash; both voices of a sentence share a fold).

    Returns one list of per-utterance duration lists per iteration.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    # Two-fold cross-fitting by sentence: a recording is re-timed only by a
    # classifier that never saw it (or the other voice's reading of its text),
    # so the classifier cannot simply reproduce the labels it was given.
    fold = np.array([int(str(item["group"])[:8], 16) % 2 for item in items])
    order = np.argsort(fold, kind="stable")
    items = [items[i] for i in order]
    fold = fold[order]
    lengths = [int(item["mel"].shape[0]) for item in items]
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    split = int(offsets[int((fold == 0).sum())])
    if min(split, offsets[-1] - split) <= 2 * window:
        raise ValueError("too few recordings for cross-fitted refinement")
    mel = ((torch.cat([item["mel"] for item in items]) - config.mel_mean) / config.mel_scale).to(device)
    speaker = torch.cat([torch.full((n,), int(item["speaker"]), dtype=torch.long) for item, n in zip(items, lengths)]).to(device)
    reference = [np.asarray(item["durations"], dtype=np.int64) for item in items]
    current = [d.copy() for d in reference]
    results = []
    for iteration in range(iterations):
        labels = torch.from_numpy(np.concatenate([frame_labels(item["symbols"], d) for item, d in zip(items, current)])).to(device)
        refined, moved, agree = [None] * len(items), [], []
        for held_out in (0, 1):
            low_train, high_train = (split, int(offsets[-1])) if held_out == 0 else (0, split)
            model = FrameClassifier(config.n_mels, len(config.speakers)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps)
            model.train()
            for step in range(steps):
                start = torch.randint(low_train, high_train - window, (batch,), generator=generator).to(device)
                index = start[:, None] + torch.arange(window, device=device)[None]
                loss = F.cross_entropy(model(mel[index], speaker[index]).flatten(0, 1), labels[index].flatten(), label_smoothing=0.05)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                if step % 1000 == 0 or step == steps - 1:
                    log(f"refine iteration={iteration + 1} held_out_fold={held_out} step={step + 1} loss={loss.item():.4f}")
            model.eval()
            prior = torch.bincount(labels[low_train:high_train], minlength=CLASSES).float().clamp_min(1)
            prior = (prior / prior.sum()).log()
            with torch.inference_mode():
                for number in np.flatnonzero(fold == held_out):
                    item, low, high = items[number], int(offsets[number]), int(offsets[number + 1])
                    original, now = reference[number], current[number]
                    logits = model(mel[low:high][None], speaker[low:high][None])[0]
                    agree.append(float((logits.argmax(-1) == labels[low:high]).float().mean()))
                    scores = (logits.log_softmax(-1) - prior_scale * prior).cpu().numpy().astype(np.float64)
                    active = np.flatnonzero(original > 0)
                    ends = np.cumsum(original)[active]
                    classes = [symbol_class(item["symbols"][i]) for i in active]
                    durations = np.zeros_like(original)
                    durations[active] = banded_viterbi(scores[:, classes], ends - original[active], ends, band)
                    refined[number] = durations
                    moved.append(np.abs(np.cumsum(durations)[active][:-1] - np.cumsum(now)[active][:-1]))
        shift = np.concatenate(moved) if moved else np.zeros(1)
        log(f"refine iteration={iteration + 1} held_out_frame_accuracy={np.mean(agree):.4f} "
            f"boundary_shift_mean={shift.mean():.3f} unchanged={np.mean(shift == 0):.3f} max={shift.max()}")
        current = refined
        restored = [None] * len(items)
        for position, source in enumerate(order):
            restored[source] = refined[position].tolist()
        results.append(restored)
    return results
