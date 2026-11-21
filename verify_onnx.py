"""Verify the exported ONNX speaks Chinese correctly (ORT on CPU)."""
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

os.environ.setdefault("TINY_VOCA_LANG", "zh")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tiny_voca.infer import load_model, phonemize

checkpoint = r"runs\zh\inference.pt"
onnx_path = r"web\public\assets\model.onnx"
text = "你好，这是一个很小的中文语音模型。它只有四百多万个参数。"

# 1) ONNX inference
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
phones = phonemize(text)
ids = np.array(phones.ids, dtype=np.int64)[None]
stress = np.array(phones.stress, dtype=np.int64)[None]
voice = np.array([0], dtype=np.int64)
one = np.array([1.0], dtype=np.float32)
zero = np.array([0.0], dtype=np.float32)
onnx_audio = session.run(
    None,
    {"tokens": ids, "stress": stress, "voice": voice, "speed": one, "pitch_shift": zero, "pitch_range": one},
)[0][0]

# 2) PyTorch inference (fixed model.py) as reference
model = load_model(checkpoint, "cpu")
tokens = torch.tensor([phones.ids])
stress_t = torch.tensor([phones.stress])
spk = torch.tensor([model.config.speakers.index("ZH_F1")])
audio, lengths, _ = model.synthesize(tokens, stress_t, torch.tensor([len(phones)]), spk, 1.0, 0.0, 0, 1.0)
torch_audio = audio[0, : int(lengths[0])].numpy()

# 3) Compare
n = min(len(onnx_audio), len(torch_audio))
print(f"onnx audio: {len(onnx_audio)} samples ({len(onnx_audio)/24000:.2f}s), peak {np.abs(onnx_audio).max():.3f}")
print(f"torch audio: {len(torch_audio)} samples ({len(torch_audio)/24000:.2f}s), peak {np.abs(torch_audio).max():.3f}")
print(f"max abs diff (first {n} samples, different noise): {np.abs(onnx_audio[:n] - torch_audio[:n]).max():.3f}")
print(f"onnx NaN/inf: {int(np.isnan(onnx_audio).sum() + np.isinf(onnx_audio).sum())}")
sf.write("onnx-check.wav", onnx_audio, 24000)
print("saved onnx-check.wav")

# 4) ASR sanity check
try:
    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8", download_root=r"E:\huggface_cache")
    segments, _ = asr.transcribe("onnx-check.wav", language="zh")
    heard = "".join(segment.text for segment in segments)
    print("whisper heard:", heard)
except Exception as error:
    print("ASR skipped:", error)
