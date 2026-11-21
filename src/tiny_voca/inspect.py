import argparse
import json
from pathlib import Path
from .config import Config
from .losses import Discriminators
from .train import TrainingSystem


def parameter_report(config):
    system = TrainingSystem(config)
    model = system.model
    groups = {
        "token, stress and voice embeddings": [model.tokens, model.stress, model.voices, model.frame_voices, model.frame_position],
        "text encoder": [model.encoder],
        "duration predictor": [model.duration],
        "frame transformer A + pitch/voicing/energy head": [model.prosody_stack, model.prosody_head, model.prosody_projection],
        "frame transformer B": [model.acoustic_stack],
        "source-filter waveform head": [model.vocoder],
    }
    parts = {name: sum(p.numel() for module in modules for p in module.parameters()) for name, modules in groups.items()}
    inference = sum(p.numel() for p in model.parameters())
    if sum(parts.values()) != inference:
        raise RuntimeError("parameter groups do not cover the inference model")
    training_only = {"auxiliary mel head": sum(p.numel() for p in system.mel_head.parameters()),
                     "waveform discriminators": sum(p.numel() for p in Discriminators().parameters())}
    return {"inference_parts": parts, "inference_parameters": inference, "training_only_parts": training_only,
            "fp32_weight_MB": round(inference * 4 / 1e6, 3), "fp16_weight_MB": round(inference * 2 / 1e6, 3),
            "under_5M": inference < 5_000_000}


def main():
    parser = argparse.ArgumentParser(description="Count all parameters without downloading weights")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    print(json.dumps(parameter_report(c), indent=2))


if __name__ == "__main__":
    main()
