"""Split the synthesized corpus into train / validation / test manifests.

The annotation layer refuses a split if any sentence appears in two of them, so
the split happens on **text**, not on rows: with two voices every sentence has
two recordings, and those two must stay on the same side of the split.

    python scripts/split_zh_corpus.py --corpus data/zh-corpus/corpus.screened.jsonl

Writes ``train.jsonl`` / ``validation.jsonl`` / ``test.jsonl`` next to the input,
with row keys ``tiny_voca.prepare`` already understands.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

SPLITS = (("train", 0.98), ("validation", 0.012), ("test", 0.008))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path, help="directory for the manifests (default: next to the corpus)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("corpus is empty")

    by_text = {}
    for row in rows:
        key = hashlib.sha256(row["text"].strip().encode()).hexdigest()
        by_text.setdefault(key, []).append(row)

    keys = sorted(by_text)
    random.Random(args.seed).shuffle(keys)
    total = len(keys)

    # Compute split counts; for small datasets, guarantee each split >= 1
    # Process non-train splits first, then train gets the remainder
    counts = {}
    remaining = total
    for index, (name, share) in enumerate(SPLITS):
        if index == 0:
            continue  # train computed last
        c = max(1, round(total * share)) if total >= len(SPLITS) else (1 if total > 0 else 0)
        c = min(c, remaining)
        counts[name] = c
        remaining -= c
    counts[SPLITS[0][0]] = remaining

    bounds, start = {}, 0
    for name, _ in SPLITS:
        count = counts[name]
        bounds[name] = keys[start:start + count]
        start += count

    output = args.output or args.corpus.parent
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, chosen in bounds.items():
        selected = [row for key in chosen for row in by_text[key]]
        path = output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        summary[name] = {"sentences": len(chosen), "clips": len(selected),
                         "hours": round(sum(row.get("seconds", 0) for row in selected) / 3600, 2)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
