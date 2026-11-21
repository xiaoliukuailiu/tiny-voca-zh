import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.export import Exportable, export
from tiny_voca.model import TinyVoca
from tiny_voca.text import PHONE_IDS


def small_model():
    torch.manual_seed(0)
    model = TinyVoca(Config()).eval()
    # A fresh model predicts arbitrary durations; pin them to a few frames per token.
    torch.nn.init.zeros_(model.duration.out.weight)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


def utterance(length=12):
    phones = sorted(PHONE_IDS)
    tokens = torch.tensor([[phones[(7 * i) % len(phones)] for i in range(length)]])
    return tokens, (torch.arange(length)[None] % 3)


class ExportTest(unittest.TestCase):
    def test_rewrite_matches_synthesize(self):
        model = small_model()
        tokens, stress = utterance()
        for voice in (0, 1):
            reference, lengths, details = model.synthesize(tokens, stress, torch.tensor([tokens.shape[1]]), torch.tensor([voice]),
                                                           speed=1.25, pitch_shift=0.1, pitch_range=1.3)
            noise = torch.randn(int(lengths[0]), generator=torch.Generator().manual_seed(0))[None]
            one = lambda value: torch.tensor([value])
            with torch.no_grad():
                audio, durations = Exportable(model, external_noise=True)(tokens, stress, torch.tensor([voice]), one(1.25), one(0.1), one(1.3), noise)
            self.assertTrue(torch.equal(durations, details["durations"]))
            self.assertEqual(audio.shape, reference.shape)
            self.assertLess(float((audio - reference).abs().max()), 1e-2 * float(reference.abs().max()))

    def test_onnx_runs_at_other_lengths(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        model = small_model()
        with tempfile.TemporaryDirectory() as directory:
            path = export(model, Path(directory) / "model.onnx", external_noise=True)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for length in (5, 40):
                tokens, stress = utterance(length)
                reference, lengths, _ = model.synthesize(tokens, stress, torch.tensor([length]), torch.tensor([1]))
                noise = np.zeros((1, model.config.hop * model.config.max_frames), dtype=np.float32)
                noise[0, :int(lengths[0])] = torch.randn(int(lengths[0]), generator=torch.Generator().manual_seed(0)).numpy()
                one = lambda value: np.array([value], dtype=np.float32)
                audio, _ = session.run(None, {"tokens": tokens.numpy(), "stress": stress.numpy(), "voice": np.array([1]),
                                              "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0), "noise": noise})
                self.assertEqual(audio.shape, tuple(reference.shape))
                self.assertLess(float(np.abs(audio - reference.numpy()).max()), 2e-2 * float(reference.abs().max()))


if __name__ == "__main__":
    unittest.main()
