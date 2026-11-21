# tiny-voca-zh

A Mandarin Chinese fork of [tiny-voca](https://github.com/n4ze3m/tiny-voca): a
4.5M-parameter, two-voice Chinese text-to-speech model trained from scratch,
which runs in real time on a CPU or in the browser.

Same architecture, losses, two-stage training and GAN gradient scaling as the
original. The changes are in the text frontend (pypinyin) and the forced
alignment layer (character-level CTC, cross-platform). 中文说明见
[README_ZH.md](README_ZH.md)。

## Features

- **Small**: 4,564,555 parameters, 18 MB, waveform generator included.
- **Fast**: >10x real time on CPU (0.44 s for 5.9 s of audio).
- **Two voices**: ZH_F1 (female) and ZH_M1 (male), Mandarin, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, and ONNX Runtime Web.
- **Open**: code, weights, training recipe, annotations and evaluation report.
- **No human recordings**: the corpus was synthesized with Qwen3-TTS 1.7B
  VoiceDesign; the model distills real prosody (pitch/voicing/energy) extracted
  from the corpus audio with a teacher-forcing schedule.

## Install

```bash
git clone https://github.com/xiaoliukuailiu/tiny-voca-zh.git
cd tiny-voca-zh
conda create -n tinyvoca python=3.11 -y
conda activate tinyvoca
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx (Blackwell)
pip install -r requirements/zh-train.txt
pip install -e .
```

Weights: `runs/zh/inference.pt` (18 MB) is checked out in the repo.

## Usage

```bash
python -m tiny_voca.infer \
  --checkpoint runs/zh/inference.pt \
  --voice ZH_F1 \
  --text "你好，世界。这是中文语音合成。" \
  --output hello.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--voice` | `ZH_F1` | `ZH_F1` or `ZH_M1` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Raise or lower the pitch |
| `--pitch-range` | 1.0 | More or less pitch movement |

Long text is fine; it is read one sentence at a time and the audio is joined.
In Python, `tiny_voca.infer.speak_text` does the same.

## ONNX

```bash
python -m tiny_voca.export --checkpoint runs/zh/inference.pt --output tiny-voca-zh.onnx
```

The export path includes the same 5-frame voicing median filter / energy
smoothing used at inference (implemented with ONNX-17-safe ops).

## Web demo

The browser demo runs the model on ONNX Runtime Web; the Chinese phonemizer
(pypinyin + zh_text) is packed for Pyodide so all text stays on the device.

```bash
cd web
bash fetch_assets.sh /path/to/model.onnx
npx wrangler dev
```

## Training

Full pipeline (synthesize corpus → features → CTC align → annotate → train →
evaluate) is one command; see [README_ZH.md](README_ZH.md) for details.

```bash
python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/zh --config config/zh.json --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
python -m tiny_voca.evaluate --checkpoint runs/zh/inference.pt --annotations annotations --output eval/zh --tag zh
```

## This training run (Nov 17–18, 2026, 2x RTX 5090)

| Item | Value |
|---|---|
| Data | 20,000 sentences (~22 h) x 2 voices (Qwen3-TTS VoiceDesign); split train 37,672 / validation 464 / test 304 |
| Parameters | 4,564,555 inference (18 MB) + 2.69M training-only |
| Vocab | 218 tokens (initials / toned finals + punctuation) |
| Optimizer | AdamW, lr 3e-4, warmup 1000, half-life 100k |
| Batch | batch_frames 9000 / max_batch 32 |
| GAN | from step 8000, ratio 0.7 |
| Teacher (real prosody) | hold 3000 -> ramp 7000 -> 0 after step 10000 |
| Steps | 104,000 |
| GPU time | ~5.8 h on one RTX 5090 (4.8-5.2 steps/s); wall clock ~8.5 h incl. one interruption (resumed from step 59000) |
| Evaluation | faster-whisper small, CER, free-running generation |

### Results

| Set | Generated CER | Reference CER |
|---|---|---|
| validation | 17.2% (F1 12.3% / M1 22.1%) | 11.0% |
| test | 23.5% | 17.0% |
| novel sentences | 31.5% | — |

Prosody (validation): pitch error 1.31 semitones, voicing accuracy 96.6%,
duration ratio 0.93. Details: [eval/zh/report.json](eval/zh/report.json).

## Limits

- Mandarin only. Two fixed voices. No voice cloning.
- Multi-character words are phonemized with pypinyin; rare proper nouns may be
  read wrong.
- Tone sandhi applies within a whole sentence, which can over-apply across
  words.
- Audio quality ceiling = Qwen3-TTS. For more natural speech, retrain on human
  recordings rather than scaling the model up.

## License

Apache-2.0 for code and weights. The corpus is synthesized with Qwen3-TTS
(Apache-2.0) and contains no human recordings. The original tiny-voca project
is by [n4ze3m](https://github.com/n4ze3m).

## Citation

```bibtex
@software{tiny-voca-zh,
  title  = {tiny-voca-zh: a 4.5M-parameter two-voice Mandarin text-to-speech model},
  year   = {2026},
  url    = {https://github.com/xiaoliukuailiu/tiny-voca-zh}
}
```
