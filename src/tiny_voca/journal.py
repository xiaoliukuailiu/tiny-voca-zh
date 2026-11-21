"""Durable JSONL preparation journals. A partial final write is recoverable."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path


@contextmanager
def exclusive_lock(path):
    """Hold an exclusive lock on ``path`` for the duration of the block.

    POSIX uses ``fcntl.flock``; Windows has no fcntl, so it locks one byte with
    ``msvcrt.locking``. Both are non-blocking: a second preparation of the same
    split fails immediately instead of waiting.
    """
    path = Path(path)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_bytes":len(value), "sha256":hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(k):json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    raise TypeError(f"unsupported provenance type: {type(value).__name__}")


def recover(path):
    """Read physical LF-delimited records and truncate only an incomplete tail.

    A malformed complete line is corruption, not an interrupted append: fail.
    The caller must hold the split's exclusive preparation lock.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.seek(position)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            records.append(json.loads(line))
    return records


def append(stream, value, sync=True):
    """Append one record. ``sync=False`` batches the fsync (the caller must end
    with a synced record); a torn tail is still recoverable either way."""
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n"
    stream.write(payload)
    stream.flush()
    if sync:
        os.fsync(stream.fileno())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        append(stream, value)
    os.replace(temporary, path)
