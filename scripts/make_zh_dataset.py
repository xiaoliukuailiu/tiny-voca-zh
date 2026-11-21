"""Build the synthetic Mandarin corpus with Qwen3-TTS.

Two stages, matching how the English release was made:

1. **Design the voices.** ``Qwen3-TTS-12Hz-1.7B-VoiceDesign`` turns a written
   description into a reference clip. Two descriptions give the two voices
   (F and M). No human recording is involved.
2. **Read the corpus.** ``Qwen3-TTS-12Hz-1.7B-Base`` clones each designed
   reference and reads every sentence in that voice, batched.

Output is one 24 kHz wav per sentence plus a JSONL manifest that
``tiny_voca.prepare`` reads directly (``text``, ``audio``, ``voice_id``).

Everything is written as it is produced, so an interrupted run resumes by
skipping sentences whose id already exists.

    python scripts/make_zh_dataset.py \
        --texts data/zh_sentences.txt \
        --output data/zh-corpus \
        --voices 2 \
        --batch 8

Add ``--screen`` to score each clip with Whisper and drop bad ones (CER).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

SAMPLE_RATE = 24000

# Voice descriptions. Concrete acoustic attributes beat metaphors: the model
# follows pitch, pace and age far more reliably than "warm and friendly".
VOICE_DESIGNS = (
    {
        "voice_id": "ZH_F1",
        "instruct": "成年女性，二十多岁，中高音区，音色明亮清透，语速中等偏慢，"
                    "吐字清晰，语气自然平稳，适合朗读，不带明显情绪起伏。",
        "reference_text": "今天的天气很好，我们一起出去走走吧。",
    },
    {
        "voice_id": "ZH_M1",
        "instruct": "成年男性，三十岁左右，中低音区，音色厚实沉稳，语速中等，"
                    "吐字清晰有力，语气自然平稳，适合朗读，不带明显情绪起伏。",
        "reference_text": "会议将在下午三点开始，请大家提前十分钟到场。",
    },
)

# Sentences that exercise the parts of the front end a plain corpus misses:
# numbers, dates, money, questions, tone sandhi pairs and neutral tones.
TEMPLATES = (
    "{n}年{mon}月{day}日，公司发布了第{q}季度的财务报告。",
    "这件商品原价{yuan}元，现在打{zhe}折，实际支付{pay}元。",
    "从北京到上海大约需要{h}小时{min}分钟。",
    "请问，去最近的地铁站怎么走？",
    "他一句话也没说，转身就走了。",
    "我很好，你也很好，大家都很好。",
    "不好意思，我不是故意的。",
    "一个、两个、三个，一共是三个。",
    "这个方案不行，那个方案也不行。",
    "孩子们在操场上跑来跑去，笑得很开心。",
)


def log(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def read_texts(path, minimum=6, maximum=60):
    """Sentences from a UTF-8 file, one per line, de-duplicated and filtered."""
    seen, texts = set(), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if not minimum <= len(text) <= maximum:
            continue
        if not re.search(r"[\u4e00-\u9fff]", text):
            continue
        key = re.sub(r"\s+", "", text)
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts


def template_texts(count):
    """Deterministic number/date/money sentences, cycled until ``count``."""
    generated = []
    index = 0
    while len(generated) < count:
        template = TEMPLATES[index % len(TEMPLATES)]
        round_ = index // len(TEMPLATES)
        values = {
            "n": 2020 + round_ % 7,
            "mon": 1 + (index * 3) % 12,
            "day": 1 + (index * 7) % 28,
            "q": 1 + index % 4,
            "yuan": 10 + index % 90,
            "zhe": 5 + index % 5,
            "pay": 10 + index % 90,
            "h": 1 + index % 12,
            "min": 1 + index % 59,
        }
        try:
            generated.append(template.format(**values))
        except (KeyError, IndexError):
            generated.append(template)
        index += 1
    return generated


def resample(wave, sample_rate, target=SAMPLE_RATE):
    import numpy as np
    if sample_rate == target:
        return np.asarray(wave, dtype=np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    factor = gcd(target, sample_rate)
    return resample_poly(np.asarray(wave, dtype=np.float64),
                         target // factor, sample_rate // factor).astype(np.float32)


def design_references(model_name, output_dir, device, dtype):
    """Stage 1: create one reference clip per voice with VoiceDesign."""
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    log({"stage": "design", "model": model_name})
    model = Qwen3TTSModel.from_pretrained(model_name, device_map=device, dtype=dtype)
    references = []
    for design in VOICE_DESIGNS:
        path = output_dir / f"reference-{design['voice_id']}.wav"
        if not path.exists():
            wavs, sr = model.generate_voice_design(
                text=design["reference_text"], language="Chinese", instruct=design["instruct"])
            sf.write(path, resample(wavs[0], sr), SAMPLE_RATE)
            log({"designed": design["voice_id"], "seconds": round(len(wavs[0]) / sr, 2)})
        references.append({**design, "path": str(path)})
    return references


def synthesize(references, texts, output_dir, batch, device, dtype, model_name, limit=None,
               manifest_name="corpus.jsonl"):
    """Stage 2: read every sentence in each cloned voice, batched."""
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    log({"stage": "synthesize", "model": model_name, "voices": [r["voice_id"] for r in references]})
    model = Qwen3TTSModel.from_pretrained(model_name, device_map=device, dtype=dtype)
    audio_dir = output_dir / "wav"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / manifest_name

    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    with manifest.open("a", encoding="utf-8") as stream:
        for reference in references:
            prompt = model.create_voice_clone_prompt(
                ref_audio=reference["path"], ref_text=reference["reference_text"])
            pending = texts if limit is None else texts[:limit]
            for start in range(0, len(pending), batch):
                chunk = pending[start:start + batch]
                records = []
                for text in chunk:
                    identifier = hashlib.sha256(f"{reference['voice_id']}:{text}".encode()).hexdigest()[:16]
                    if identifier in done:
                        continue
                    records.append((identifier, text))
                if not records:
                    continue
                wavs, sr = model.generate_voice_clone(
                    text=[text for _, text in records],
                    language=["Chinese"] * len(records),
                    voice_clone_prompt=prompt,
                )
                for (identifier, text), wav in zip(records, wavs):
                    name = f"{reference['voice_id']}-{identifier}.wav"
                    path = audio_dir / name
                    samples = resample(wav, sr)
                    sf.write(path, samples, SAMPLE_RATE)
                    row = {
                        "id": identifier, "text": text,
                        "audio": str(path.relative_to(output_dir)).replace(os.sep, "/"),
                        "voice_id": reference["voice_id"],
                        "seconds": round(len(samples) / SAMPLE_RATE, 3),
                        "origin": "qwen3-tts-voice-clone",
                        "teacher_model": model_name,
                        "reference_audio": Path(reference["path"]).name,
                        "source_license": "synthetic",
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add(identifier)
                stream.flush()
                log({"voice": reference["voice_id"], "written": len(records), "of": len(pending)})
    return manifest


def character_error_rate(reference, hypothesis):
    reference = re.sub(r"[^\u4e00-\u9fff]", "", reference)
    hypothesis = re.sub(r"[^\u4e00-\u9fff]", "", hypothesis)
    if not reference:
        return 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1] / len(reference)


def screen(manifest, output_dir, model_name, threshold, device):
    """Optional: transcribe with Whisper and drop clips above ``threshold`` CER."""
    from faster_whisper import WhisperModel
    # faster-whisper wants "cuda", not "cuda:0".
    model = WhisperModel(model_name, device=device.split(":")[0], compute_type="float16")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, dropped, scores = [], [], []
    for row in rows:
        path = output_dir / row["audio"]
        segments, _ = model.transcribe(str(path), language="zh", beam_size=1)
        hypothesis = "".join(segment.text for segment in segments)
        score = character_error_rate(row["text"], hypothesis)
        scores.append(score)
        (kept if score <= threshold else dropped).append({**row, "transcript": hypothesis, "cer": round(score, 4)})
    scores.sort()
    summary = {"clips": len(rows), "kept": len(kept), "dropped": len(dropped), "threshold": threshold,
               "cer_median": round(scores[len(scores) // 2], 4) if scores else None,
               "cer_p90": round(scores[int(len(scores) * 0.9)], 4) if scores else None}
    (output_dir / "corpus.screened.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    (output_dir / "corpus.dropped.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dropped), encoding="utf-8")
    log({"stage": "screen", **summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, help="UTF-8 file, one sentence per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", type=int, default=200, help="append N generated number/date/money sentences")
    parser.add_argument("--limit", type=int, help="use only the first N sentences per voice")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    _hf_home = os.environ.get("HF_HOME", "")
    _design_default = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    _clone_default = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    if _hf_home:
        import glob
        _design_snap = glob.glob(os.path.join(_hf_home, "hub", "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign", "snapshots", "*"))
        _clone_snap = glob.glob(os.path.join(_hf_home, "hub", "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base", "snapshots", "*"))
        if _design_snap:
            _design_default = _design_snap[0]
        if _clone_snap:
            _clone_default = _clone_snap[0]
    parser.add_argument("--design-model", default=_design_default)
    parser.add_argument("--clone-model", default=_clone_default)
    parser.add_argument("--voices", nargs="+", help="synthesize only these voice ids (default: all)")
    parser.add_argument("--manifest-name", default="corpus.jsonl",
                        help="manifest file name inside --output (parallel voice runs)")
    parser.add_argument("--stage", default="all", choices=("all", "design", "synth", "screen"))
    parser.add_argument("--screen-model", default="large-v3")
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    args = parser.parse_args()

    import torch
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    args.output.mkdir(parents=True, exist_ok=True)

    reference_dir = args.output / "reference"
    reference_dir.mkdir(exist_ok=True)

    if args.stage in ("all", "design", "synth"):
        if not args.texts:
            parser.error("--texts is required for design/synth")
        texts = read_texts(args.texts)
        texts += template_texts(args.templates)
        log({"sentences": len(texts)})

    references = None
    if args.stage in ("all", "design"):
        references = design_references(args.design_model, reference_dir, args.device, dtype)
    if args.stage in ("all", "synth"):
        if references is None:
            references = []
            for design in VOICE_DESIGNS:
                path = reference_dir / f"reference-{design['voice_id']}.wav"
                if not path.exists():
                    sys.exit(f"missing reference {path}; run --stage design first")
                references.append({**design, "path": str(path)})
        if args.voices:
            references = [r for r in references if r["voice_id"] in args.voices]
        manifest = synthesize(references, texts, args.output, args.batch, args.device, dtype,
                              args.clone_model, args.limit, args.manifest_name)
        log({"manifest": str(manifest)})
    if args.stage == "screen":
        screen(args.output / "corpus.jsonl", args.output, args.screen_model,
               args.cer_threshold, args.device)


if __name__ == "__main__":
    main()
