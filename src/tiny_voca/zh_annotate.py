"""Mandarin annotation layer: phonemize text, read MFA alignments, write manifests.

This is the Chinese counterpart of ``annotate``. The English step aligns with a
CTC phoneme recogniser; Mandarin aligns with the Montreal Forced Aligner
(``mfa_align``), so the two stages are:

    # 1. turn acoustic features into an MFA corpus (<id>.wav + <id>.lab)
    python -m tiny_voca.zh_annotate --features features --export-corpus mfa-corpus

    # 2. align (run once per split; the dictionary and model come with MFA)
    mfa align mfa-corpus/train mandarin.dict mandarin.zip mfa-out/train --single_speaker

    # 3. write the manifests training reads
    python -m tiny_voca.zh_annotate --features features --textgrids mfa-out --output annotations

The output has exactly the fields ``tiny_voca.data.AnnotatedDataset`` checks:
tokens, stress, durations, frames, speaker, path, frontend, acoustic, annotation.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from .config import ANNOTATION_VERSION, Config
from .mfa_align import durations, parse_intervals
from .text import (LANGUAGE, PHONE_SET, SYMBOL_TO_ID, UNK, frontend_fingerprint,
                   frontend_identity, frontend_runtime, phonemize_with_owners, syllables_of)
from .zh_align_ctc import ALIGNER_MODEL as CTC_ALIGNER_MODEL
from .zh_align_ctc import CharacterAligner, durations_from_chars

SPLITS = ("train", "validation", "test")


def load_rows(features, split, config):
    metadata = features / (split + ".preparation.json")
    if not metadata.exists() or not json.loads(metadata.read_text()).get("complete"):
        raise ValueError(f"acoustic preparation for {split} is missing or incomplete")
    rows = [json.loads(line) for line in (features / (split + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("acoustic") != config.acoustic_fingerprint:
            raise ValueError("feature files were not produced with compatible audio/DSP settings")
        if row["speaker"] not in config.speakers:
            raise ValueError(f"unknown speaker {row['speaker']!r}")
    return rows


def export_corpus(features, rows, root):
    """Write ``<id>.wav`` and ``<id>.lab`` so MFA can align each sentence."""
    import soundfile as sf
    import torch
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        item = torch.load(features / row["path"], map_location="cpu", weights_only=True)
        sf.write(root / f"{row['id']}.wav", item["wave"].numpy(), 24000)
        (root / f"{row['id']}.lab").write_text(row["text"], encoding="utf-8")


def annotate_split(features, rows, textgrids, split, config, limits, aligner=None):
    accepted, rejected = [], []
    import numpy as np
    import torch
    for row in rows:
        try:
            phonemized, owners = phonemize_with_owners(row["text"])
            if aligner is not None:
                syllables = syllables_of(row["text"])
                if len(syllables) != max(owners) + 1:
                    raise ValueError("syllable count does not match token owners")
                item = torch.load(features / row["path"], map_location="cpu", weights_only=True)
                wave = item["wave"].numpy()
                voiced = np.asarray(item["prosody"][:, 1]) > 0.5
                spans = aligner.spans(wave, config.sample_rate, [s.char for s in syllables],
                                      config.hop, config.sample_rate)
                counts = durations_from_chars(phonemized.symbols, owners, spans, voiced,
                                             row["frames"], limits["min_pause_frames"])
            else:
                textgrid = textgrids / split / f"{row['id']}.TextGrid"
                if not textgrid.exists():
                    raise FileNotFoundError("missing_textgrid")
                intervals = parse_intervals(textgrid)
                counts = durations(intervals, phonemized.symbols, row["frames"], config.hop,
                                   config.sample_rate, limits["min_pause_frames"],
                                   strict=False, phone_set=PHONE_SET)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            rejected.append({"id": row["id"], "text": row["text"], "reasons": [f"{type(error).__name__}: {error}"]})
            continue
        reasons = []
        if phonemized.unknown:
            reasons.append("unknown_phoneme")
        speech = sum(count for symbol, count in zip(phonemized.symbols, counts)
                     if symbol in PHONE_SET or symbol == UNK)
        # The rate gate is in phones per second (as in the English align.py),
        # not in speech frames per second.
        phones = sum(1 for symbol, count in zip(phonemized.symbols, counts)
                     if (symbol in PHONE_SET or symbol == UNK) and count > 0)
        seconds = row["frames"] * config.hop / config.sample_rate
        rate = phones / max(seconds, 1e-6)
        if not limits["min_phones_per_second"] <= rate <= limits["max_phones_per_second"]:
            reasons.append("implausible_rate")
        entry = {
            "id": row["id"], "split": split, "speaker": row["speaker"], "text": row["text"],
            "text_hash": row["text_hash"], "path": row["path"], "frames": row["frames"],
            "symbols": " ".join(phonemized.symbols), "tokens": list(phonemized.ids),
            "stress": list(phonemized.stress), "durations": counts,
            "unknown": list(phonemized.unknown),
            "qc": {"phones_per_second": round(rate, 2), "speech_frames": speech,
                   "unknown": len(phonemized.unknown)},
            "provenance": row.get("provenance", {}),
            "acoustic": config.acoustic_fingerprint,
            "frontend": frontend_fingerprint(),
        }
        if sum(counts) != row["frames"] or len(counts) != len(phonemized.ids):
            reasons.append("duration_length_mismatch")
        if reasons:
            rejected.append({"id": row["id"], "text": row["text"], "reasons": reasons})
            continue
        accepted.append(entry)
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--textgrids", type=Path, help="directory holding <split>/<id>.TextGrid")
    parser.add_argument("--output", type=Path, help="annotation directory to write")
    parser.add_argument("--export-corpus", type=Path, help="write <split>/<id>.wav + .lab here and exit")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--aligner", choices=("mfa", "ctc"),
                        help="mfa reads TextGrids (Linux/macOS); ctc aligns in-process with a Chinese "
                             "wav2vec2 model and works on Windows (default: ctc on Windows, else mfa)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true", help="never download the aligner weights")
    parser.add_argument("--min-pause-frames", type=int, default=7)
    parser.add_argument("--min-phones-per-second", type=float, default=2.0)
    parser.add_argument("--max-phones-per-second", type=float, default=20.0)
    args = parser.parse_args()

    config = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    if LANGUAGE != "zh":
        raise SystemExit("zh_annotate needs TINY_VOCA_LANG=zh")
    limits = {"min_pause_frames": args.min_pause_frames,
              "min_phones_per_second": args.min_phones_per_second,
              "max_phones_per_second": args.max_phones_per_second}

    rows = {split: load_rows(args.features, split, config) for split in args.splits if (args.features / (split + ".jsonl")).exists()}
    if not rows:
        raise SystemExit(f"no <split>.jsonl found under {args.features}")
    args.splits = list(rows)
    if args.limit:
        rows = {split: value[:args.limit] for split, value in rows.items()}

    if args.export_corpus:
        for split in args.splits:
            export_corpus(args.features, rows[split], args.export_corpus / split)
        print(json.dumps({"exported": {split: len(rows[split]) for split in args.splits}}))
        return

    if not args.output:
        parser.error("--output is required unless --export-corpus is given")

    aligner_name = args.aligner or ("ctc" if os.name == "nt" else "mfa")
    if aligner_name == "mfa" and not args.textgrids:
        parser.error("--aligner mfa needs --textgrids (or use --aligner ctc, which needs no external binary)")
    aligner = None
    if aligner_name == "ctc":
        aligner = CharacterAligner(args.device, local_files_only=args.local_files_only)

    identity = {
        "annotation_version": ANNOTATION_VERSION, "language": LANGUAGE,
        "acoustic": config.acoustic_fingerprint, "frontend": frontend_fingerprint(),
        **frontend_runtime(),
        "aligner": {"kind": "mfa"} if aligner_name == "mfa" else {"kind": "ctc", "model": CTC_ALIGNER_MODEL},
        "limits": limits,
        "sources": {split: hashlib.sha256((args.features / (split + ".jsonl")).read_bytes()).hexdigest()
                    for split in args.splits},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "annotations.json").write_text(json.dumps(
        {"identity": identity, "fingerprint": fingerprint, "complete": not args.limit,
         "acoustic_root": str(args.features.resolve()), "limits": limits,
         "frontend": frontend_identity()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {}
    for split in args.splits:
        accepted, rejected = annotate_split(args.features, rows[split], args.textgrids, split, config, limits, aligner)
        for row in accepted:
            row["annotation"] = fingerprint
        (args.output / (split + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
        (args.output / (split + ".rejected.jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected), encoding="utf-8")
        summary[split] = {
            "accepted": len(accepted), "rejected": len(rejected),
            "hours": round(sum(row["frames"] for row in accepted) * config.hop / config.sample_rate / 3600, 3),
            "rejection_reasons": dict(Counter(reason.split(":")[0] for row in rejected for reason in row["reasons"])),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
