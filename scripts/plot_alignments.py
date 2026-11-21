"""Render alignment diagnostics for visual review. Cloud only (needs matplotlib).

One PNG per recording: log-mel spectrogram with token boundaries and labels,
and below it the frame energy, the pitch tracker's voicing and the pitch curve
the oscillator is trained on. Pause tokens are shaded.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from tiny_voca.align import frame_energy_db
from tiny_voca.config import Config
from tiny_voca.data import continuous_pitch
from tiny_voca.text import PHONE_SET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-voice", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = Config()
    rows = [json.loads(line) for line in (args.annotations / (args.split + ".jsonl")).read_text(encoding="utf-8").split("\n") if line.strip()]
    root = Path(json.loads((args.annotations / "annotations.json").read_text())["acoustic_root"])
    args.output.mkdir(parents=True, exist_ok=True)
    chosen = []
    for voice in c.speakers:
        mine = sorted((r for r in rows if r["speaker"] == voice), key=lambda r: (r["frames"], r["id"]))
        chosen += [mine[int(q * (len(mine) - 1))] for q in np.linspace(0.15, 0.7, args.per_voice)]
    index = []
    for row in chosen:
        stored = torch.load(root / row["path"], map_location="cpu", weights_only=True)
        frames = row["frames"]
        seconds = np.arange(frames) * c.hop / c.sample_rate
        symbols, durations = row["symbols"].split(), np.asarray(row["durations"])
        ends = np.cumsum(durations) * c.hop / c.sample_rate
        starts = ends - durations * c.hop / c.sample_rate
        pitch, trusted = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(max(12, frames / 22), 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        top.imshow(stored["mel"].numpy().T, origin="lower", aspect="auto", cmap="magma",
                   extent=(-0.5 * c.hop / c.sample_rate, (frames - 0.5) * c.hop / c.sample_rate, 0, c.n_mels), vmin=-11, vmax=2)
        for symbol, start, end, duration in zip(symbols, starts, ends, durations):
            if duration == 0:
                continue
            phone = symbol in PHONE_SET
            top.axvline(start - 0.5 * c.hop / c.sample_rate, color="white", linewidth=0.6, alpha=0.9)
            if not phone:
                top.axvspan(start - 0.5 * c.hop / c.sample_rate, end - 0.5 * c.hop / c.sample_rate, color="cyan", alpha=0.18)
                bottom.axvspan(start, end, color="tab:cyan", alpha=0.15)
            label = {"<bos>": "BOS", "<eos>": "EOS", "<sp>": "sp"}.get(symbol, symbol)
            top.text((start + end) / 2 - 0.5 * c.hop / c.sample_rate, c.n_mels + 1.5, label, ha="center", va="bottom",
                     fontsize=9, color="black" if phone else "teal")
        top.set_ylabel("mel bin")
        top.set_title(f"{row['id']}  ({row['speaker']})  {row['text'][:110]}", fontsize=10, pad=22)
        level = frame_energy_db(stored["wave"].numpy(), c.hop)
        bottom.plot(seconds, level, color="0.35", linewidth=0.9, label="level (dB)")
        bottom.set_ylabel("dB")
        bottom.set_ylim(-95, 0)
        voiced = stored["prosody"][:, 1].numpy() > 0.5
        axis = bottom.twinx()
        hertz = c.pitch_reference_hz * np.exp(pitch.numpy())
        axis.plot(seconds, np.where(voiced, hertz, np.nan), color="tab:red", linewidth=1.4, label="pitch (voiced)")
        axis.plot(seconds, np.where(voiced, np.nan, hertz), color="tab:red", linewidth=0.8, linestyle=":", label="interpolated")
        axis.set_ylabel("Hz")
        bottom.set_xlabel("seconds")
        figure.tight_layout()
        name = f"{row['id']}.png"
        figure.savefig(args.output / name, dpi=110)
        plt.close(figure)
        index.append({"id": row["id"], "speaker": row["speaker"], "text": row["text"], "image": name, "qc": row["qc"]})
    (args.output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": len(index), "directory": str(args.output)}))


if __name__ == "__main__":
    main()
