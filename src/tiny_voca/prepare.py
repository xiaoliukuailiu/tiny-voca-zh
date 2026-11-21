"""Cloud-only ACOUSTIC feature preparation from local JSONL or HF-style Parquet shards.

Stores the waveform, log-mel, pitch/voicing and energy of each recording. It
does not tokenize text: phoneme tokens and durations are a separate, versioned
annotation layer (``tiny_voca.annotate``), so a front-end or aligner change
never forces this expensive step to run again.

No network access: use the HF CLI separately to obtain a pinned dataset snapshot.
"""

import argparse
import hashlib
import io
import json
import librosa
import math
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .config import Config
from .journal import append, atomic_json, exclusive_lock, file_hash, json_safe, recover

MIN_FRAMES = 32


def records(source):
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        yield {"_input_error":str(error)}
    elif source.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            yield from batch.to_pylist()
    else:
        raise ValueError("input must be .jsonl or .parquet")


def extract(wave, config):
    c = config
    if wave.ndim != 1 or not np.isfinite(wave).all() or not wave.size:
        raise ValueError("expected finite mono audio")
    frames = math.ceil(len(wave)/c.hop)
    if frames < 1:
        raise ValueError("empty audio")
    if np.max(np.abs(wave)) > 1.01 or np.sqrt(np.mean(wave**2)) < 1e-5:
        raise ValueError("audio is clipped/out of range or effectively silent")
    wave = np.pad(wave, (0, frames*c.hop-len(wave))).astype(np.float32)
    spectrum = librosa.stft(wave, n_fft=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[:, :frames]
    filters = librosa.filters.mel(sr=c.sample_rate, n_fft=c.n_fft, n_mels=c.n_mels)
    mel = np.log(np.maximum(filters @ np.abs(spectrum)**2, 1e-5)).T
    f0, voiced, _ = librosa.pyin(wave, fmin=c.f0_min, fmax=c.f0_max, sr=c.sample_rate,
                                frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")
    f0, voiced = f0[:frames], voiced[:frames]
    pitch = np.where(voiced, np.log(np.maximum(np.nan_to_num(f0, nan=200.0), 1)/200.0), 0)
    rms = librosa.feature.rms(y=wave, frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[0, :frames]
    prosody = np.stack((pitch, voiced.astype(np.float32), np.log(np.maximum(rms, 1e-5))), axis=-1)
    return {"wave": torch.from_numpy(wave), "mel": torch.from_numpy(mel.astype(np.float32)),
            "prosody": torch.from_numpy(prosody.astype(np.float32))}


_WORKER = {}


def _worker_init(feature_dir, config, text_column, speaker_column, split):
    """Per-worker setup: silence BLAS oversubscription before any compute."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _WORKER.update(feature_dir=Path(feature_dir), config=config, text_column=text_column,
                   speaker_column=speaker_column, split=split)


def _prepare_row(source, source_index, index, row, c, text_column, speaker_column, split, feature_dir):
    """Validate one row, extract its features and durably save the .pt file."""
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    if "_input_error" in row:
        raise ValueError(row["_input_error"])
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("row must have a stable nonempty string id")
    if row.get("split", split) != split:
        raise ValueError("row split differs from --split")
    text, speaker = row[text_column], row[speaker_column]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if speaker not in c.speakers:
        raise ValueError(f"unknown speaker {speaker!r}; expected {c.speakers}")
    provenance = json_safe({k:row[k] for k in ("text_id", "origin", "source_id", "source_license", "source_url", "teacher_model", "reference_audio") if k in row})
    audio = row["audio"]
    if isinstance(audio, dict):
        audio = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else source.parent / audio["path"]
    else:
        audio = source.parent / audio
    wave, sr = sf.read(audio, dtype="float32")
    if sr != c.sample_rate:
        raise ValueError(f"expected {c.sample_rate} Hz; got {sr}. Resample explicitly before preparation")
    frames = math.ceil(len(wave)/c.hop)
    if not MIN_FRAMES <= frames <= c.max_frames:
        raise ValueError("audio duration outside admission limits")
    name = hashlib.sha256(identifier.encode()).hexdigest()+".pt"
    entry = {"record_key":f"{source_index}:{index}", "id":identifier, "path":f"{split}/{name}", "acoustic":c.acoustic_fingerprint,
             "text_hash":hashlib.sha256(text.strip().encode()).hexdigest(), "text":text,
             "speaker":speaker, "source_file":str(source), "frames":frames, "provenance":provenance}
    json.dumps(entry, ensure_ascii=False, allow_nan=False)  # Validate before expensive DSP.
    sample = extract(wave, c)
    sample.update(speaker=c.speakers.index(speaker))
    temporary = feature_dir/(name+".tmp")
    with temporary.open("wb") as stream:
        torch.save(sample, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, feature_dir/name)
    return entry


def _worker_prepare(task):
    """Run one row inside a worker process; journals stay in the main process."""
    source_path, source_index, index, row = task
    try:
        entry = _prepare_row(Path(source_path), source_index, index, row, _WORKER["config"],
                             _WORKER["text_column"], _WORKER["speaker_column"],
                             _WORKER["split"], _WORKER["feature_dir"])
        return {"record_key":f"{source_index}:{index}", "entry":entry, "reason":None,
                "identifier":entry["id"], "source":source_path}
    except Exception as error:
        identifier = row.get("id") if isinstance(row, dict) else None
        return {"record_key":f"{source_index}:{index}", "entry":None,
                "reason":f"{type(error).__name__}: {error}", "identifier":str(identifier), "source":source_path}


def _sync(accepted_stream, rejected_stream):
    accepted_stream.flush()
    os.fsync(accepted_stream.fileno())
    rejected_stream.flush()
    os.fsync(rejected_stream.fileno())


def _progress(accepted_count, total_rows, split):
    if accepted_count % 100 == 0:
        print(json.dumps({"prepared":accepted_count, "of":total_rows, "split":split}), flush=True)


def _prepare_serial(args, c, tasks, total_rows, accepted_stream, rejected_stream,
                    accepted_count, rejected_count, seen, feature_dir):
    for source_path, source_index, index, row in tasks:
        source = Path(source_path)
        record_key = f"{source_index}:{index}"
        try:
            entry = _prepare_row(source, source_index, index, row, c, args.text_column,
                                 args.speaker_column, args.split, feature_dir)
            if entry["id"] in seen:
                raise ValueError("duplicate id")
        except Exception as error:
            identifier = row.get("id") if isinstance(row, dict) else None
            append(rejected_stream, {"record_key":record_key, "id":str(identifier), "source":str(source), "reason":f"{type(error).__name__}: {error}"}, sync=False)
            rejected_count += 1
        else:
            # Journal write failures must abort, not masquerade as bad source rows.
            append(accepted_stream, entry, sync=False)
            seen.add(entry["id"])
            accepted_count += 1
            _progress(accepted_count, total_rows, args.split)
    _sync(accepted_stream, rejected_stream)
    return accepted_count, rejected_count


def _prepare_parallel(args, c, tasks, total_rows, accepted_stream, rejected_stream,
                      accepted_count, rejected_count, seen, feature_dir):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                   initializer=_worker_init,
                                   initargs=(str(feature_dir), c, args.text_column, args.speaker_column, args.split))
    try:
        futures = {executor.submit(_worker_prepare, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result["reason"] is not None:
                append(rejected_stream, {"record_key":result["record_key"], "id":result["identifier"],
                                         "source":result["source"], "reason":result["reason"]}, sync=False)
                rejected_count += 1
            elif result["entry"]["id"] in seen:
                append(rejected_stream, {"record_key":result["record_key"], "id":result["entry"]["id"],
                                         "source":result["source"], "reason":"ValueError: duplicate id"}, sync=False)
                rejected_count += 1
            else:
                # Journal write failures must abort, not masquerade as bad source rows.
                append(accepted_stream, result["entry"], sync=False)
                seen.add(result["entry"]["id"])
                accepted_count += 1
                _progress(accepted_count, total_rows, args.split)
    finally:
        executor.shutdown(wait=True)
    _sync(accepted_stream, rejected_stream)
    return accepted_count, rejected_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--speaker-column", default="voice_id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--resume", action="store_true", help="Skip durably accepted/rejected records from these same inputs")
    parser.add_argument("--workers", type=int, default=1, help="Parallel DSP worker processes; 0/1 means serial")
    args = parser.parse_args()
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    args.output.mkdir(parents=True, exist_ok=True)
    # Lock is held through completion or an exception. Different splits may run separately.
    with exclusive_lock(args.output/(args.split+".lock")):
        prepare_split(args, c)


def prepare_split(args, c):
    manifest = args.output / (args.split+".jsonl")
    rejects = args.output / (args.split+".rejected.jsonl")
    metadata_path = args.output/(args.split+".preparation.json")
    identity = {"acoustic":c.acoustic_fingerprint,
                "sources":[{"path":str(p.resolve()), "sha256":file_hash(p)} for p in args.inputs],
                "text_column":args.text_column, "speaker_column":args.speaker_column, "speakers":list(c.speakers),
                # Admission limits affect which rows finish as rejects, not tensor contents.
                "admission":{"max_frames":c.max_frames, "min_frames":MIN_FRAMES}}
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError("preparation exists; use --resume or a new directory")
        if json.loads(metadata_path.read_text())["identity"] != identity:
            raise ValueError("resume inputs, feature schema or admission limits changed")
    elif manifest.exists() or rejects.exists():
        raise ValueError("manifest has no compatible preparation metadata; use a new directory")
    feature_schema = args.output/"features.json"
    if feature_schema.exists() and json.loads(feature_schema.read_text())["fingerprint"] != c.acoustic_fingerprint:
        raise ValueError("feature schema differs from other splits in this directory")
    atomic_json(feature_schema, {"fingerprint":c.acoustic_fingerprint, "fields":c.acoustic_config})
    atomic_json(metadata_path, {"identity":identity, "complete":False})
    feature_dir = args.output / args.split
    feature_dir.mkdir(exist_ok=True)
    entries, rejected = recover(manifest), recover(rejects)
    done = {r["record_key"] for r in entries+rejected}
    seen = {r["id"] for r in entries}
    for entry in entries:
        if not (args.output/entry["path"]).is_file():
            raise ValueError(f"committed feature file is missing: {entry['path']}")
    accepted_count, rejected_count = len(entries), len(rejected)
    tasks = []
    for source_index, source in enumerate(args.inputs):
        for index, row in enumerate(records(source)):
            record_key = f"{source_index}:{index}"
            if record_key not in done:
                tasks.append((str(source.resolve()), source_index, index, row))
    total_rows = accepted_count + rejected_count + len(tasks)
    print(json.dumps({"split":args.split, "pending":len(tasks), "of":total_rows, "workers":args.workers}), flush=True)
    with manifest.open("a", encoding="utf-8") as accepted_stream, rejects.open("a", encoding="utf-8") as rejected_stream:
        if args.workers > 1 and tasks:
            accepted_count, rejected_count = _prepare_parallel(args, c, tasks, total_rows, accepted_stream, rejected_stream,
                                                               accepted_count, rejected_count, seen, feature_dir)
        else:
            accepted_count, rejected_count = _prepare_serial(args, c, tasks, total_rows, accepted_stream, rejected_stream,
                                                             accepted_count, rejected_count, seen, feature_dir)
    atomic_json(metadata_path, {"identity":identity, "complete":True, "accepted":accepted_count, "rejected":rejected_count})
    print(json.dumps({"accepted":accepted_count, "rejected":rejected_count, "manifest":str(manifest)}))
    if not accepted_count:
        raise SystemExit("no usable records; inspect rejected.jsonl")


if __name__ == "__main__":
    main()
