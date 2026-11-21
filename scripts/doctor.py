"""Check that this machine can run the Mandarin pipeline.

Run it before anything else; it fails loudly instead of letting a missing
dependency surface three hours into synthesis.

    TINY_VOCA_LANG=zh python scripts/doctor.py --stage all

Stages: ``synth`` (Qwen3-TTS), ``align`` (MFA), ``train`` (torch + features),
``infer``. ``all`` checks everything and reports every problem at once.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"


def check(condition, name, detail="", level=FAIL):
    return {"name": name, "status": OK if condition else level, "detail": detail}


def module(name):
    return importlib.util.find_spec(name) is not None


def version_of(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "?"


def torch_report():
    results = []
    if not module("torch"):
        return [check(False, "torch", "not installed")]
    import torch
    results.append(check(True, "torch", torch.__version__))
    results.append(check(torch.cuda.is_available(), "cuda", f"{torch.version.cuda}" if torch.cuda.is_available() else "no GPU visible"))
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        results.append(check(True, "gpu", f"{name} (sm_{capability[0]}{capability[1]})"))
        # Blackwell (RTX 50xx) needs a CUDA 12.8 build; older wheels report an
        # unsupported architecture only when a kernel is first launched.
        if capability[0] >= 12 and tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 7):
            results.append(check(False, "blackwell support",
                                 "RTX 50xx needs torch>=2.7 built for CUDA 12.8; upgrade before training",
                                 level=FAIL))
    return results


def align_report():
    """The aligner needs torchaudio + transformers. MFA is optional (and has no
    Windows build); the in-process ctc aligner is the default there."""
    results = []
    for package in ("transformers", "torchaudio"):
        installed = module(package)
        results.append(check(installed, package,
                             version_of(package) if installed else "missing (the ctc aligner needs it)"))
    executable = shutil.which("mfa")
    results.append(check(bool(executable), "mfa",
                         "not on PATH; only needed for --aligner mfa (no Windows build, use ctc)",
                         level=WARN))
    return results


def qwen_report():
    results = [check(module("qwen_tts"), "qwen-tts", "pip install -U qwen-tts")]
    if module("qwen_tts"):
        for package in ("soundfile", "scipy", "numpy"):
            results.append(check(module(package), package, version_of(package) if module(package) else "missing"))
    return results


def corpus_report(root):
    texts = root / "data" / "zh_sentences.txt"
    if not texts.exists():
        return [check(False, "corpus", f"{texts} missing; run data/make_zh_corpus.py")]
    lines = [line for line in texts.read_text(encoding="utf-8").splitlines() if line.strip()]
    characters = sum(len(line) for line in lines)
    return [check(len(lines) > 1000, "corpus", f"{len(lines)} sentences, ~{characters / 5 / 3600:.1f} h of speech",
                  level=WARN)]


def disk_report(root, minimum_gb=60):
    free = shutil.disk_usage(root).free / 2 ** 30
    return [check(free >= minimum_gb, "disk", f"{free:.1f} GB free (need ~{minimum_gb} GB for 20 h of 24 kHz audio)",
                  level=WARN)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=("all", "synth", "align", "train", "infer"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    language = os.environ.get("TINY_VOCA_LANG", "en")
    results = [check(sys.version_info >= (3, 10), "python", sys.version.split()[0]),
               check(language == "zh", "TINY_VOCA_LANG", f"{language!r} (set it to zh)", level=WARN)]

    if args.stage in ("all", "train", "infer"):
        results += torch_report()
        results.append(check(module("pypinyin"), "pypinyin", version_of("pypinyin") if module("pypinyin") else "pip install pypinyin"))
        for package in ("soundfile", "librosa", "pyarrow"):
            results.append(check(module(package), package, version_of(package) if module(package) else "missing (prepare needs it)"))
        if args.stage in ("all", "infer"):
            results.append(check(module("faster_whisper"), "faster-whisper", "optional: evaluation and corpus screening", level=WARN))
    if args.stage in ("all", "synth"):
        results += qwen_report()
        results += corpus_report(args.root)
    if args.stage in ("all", "align"):
        results += align_report()
    if args.stage in ("all", "synth", "train"):
        results += disk_report(args.root)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    failures = [item for item in results if item["status"] == FAIL]
    warnings = [item for item in results if item["status"] == WARN]
    print(f"\n{len(results) - len(failures)} ok, {len(warnings)} warning, {len(failures)} failed", file=sys.stderr)
    if failures:
        print("Fix: " + "; ".join(item["name"] for item in failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
