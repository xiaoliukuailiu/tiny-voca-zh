"""Agreement of the phonemizer with eSpeak on the corpus sentences. Cloud only.

    python compare_espeak.py --reference data/corpus/espeak_reference.jsonl \
        --data built --output out/agreement

The reference is the eSpeak token sequence stored with the corpus annotations
(one row per distinct sentence). eSpeak is only the yardstick here: nothing
from this report goes back into the dictionary or the model.

Reported, over all sentences:
  * tokens      every token (phones, word boundaries, punctuation), no stress
  * phones      phones only
  * phones+stress  phones with their stress value
  * stress      stress agreement on the vowels both sides agree on
and where the two differ: phone confusions, and the words behind them.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dizione.convert import to_text                             # noqa: E402
from dizione.engine import Phonemizer                           # noqa: E402
from dizione.sentence import groups                             # noqa: E402
from tiny_voca.text import PHONE_SET, VOWEL_SET, tokenize       # noqa: E402


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def alignment(a, b):
    """Pairs ``(i, j)`` aligning sequence ``a`` to ``b``; ``None`` marks a gap."""
    rows, columns = len(a), len(b)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i
    for j in range(1, columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs, i, j = [], rows, columns
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


_WEAK = frozenset(("ə", "ɐ", "ɪ", "ᵻ", "i", "ʊ", "ɚ"))
_WRITTEN_OUT = re.compile(r"[0-9$£€%&@#+=/]")


def kind_of(ours, theirs):
    """What sort of difference one phone edit is."""
    pair = {ours, theirs}
    if "-" in pair:
        return "r added or dropped" if "ɹ" in pair else "phone added or dropped"
    if pair <= {"ɔːɹ", "oːɹ", "ɔː", "oː"}:
        return "north or force vowel (ɔːɹ, oːɹ)"
    if pair <= {"ɑː", "ɔ", "ɔː"}:
        return "lot, cloth or thought vowel (ɑː, ɔ, ɔː)"
    if pair <= {"t", "ɾ", "ʔ", "d"}:
        return "t, flap or glottal stop"
    if ours in VOWEL_SET and theirs in VOWEL_SET:
        return "weak vowel quality (ə, ɐ, ɪ, ᵻ, i)" if pair & _WEAK else "other vowel"
    return "consonant" if ours not in VOWEL_SET and theirs not in VOWEL_SET else "vowel for consonant"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-phrases", action="store_true", help="Keep a boundary between all words")
    parser.add_argument("--show", type=int, default=60)
    args = parser.parse_args()

    phonemizer = Phonemizer(args.data, join_phrases=not args.no_phrases)
    rows = [json.loads(line) for line in args.reference.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    totals = Counter()
    confusions, words, by_source, failures = Counter(), Counter(), defaultdict(Counter), []
    kinds, split = defaultdict(Counter), defaultdict(Counter)
    examples = {}
    for row in rows:
        reference = list(zip(row["symbols"], row["stress"]))
        part = "numbers_and_symbols" if _WRITTEN_OUT.search(row["text"]) else "plain_prose"
        try:
            analysis = phonemizer.analyse(row["text"])
            ours = tokenize([(True, value) if kind == "punct" else (False, groups(value)) for kind, value in analysis])
        except Exception as error:                          # a sentence we cannot phonemize counts fully against us
            failures.append({"text": row["text"], "error": f"{type(error).__name__}: {error}"})
            totals["tokens"] += len(reference)
            totals["token_edits"] += len(reference)
            phones = sum(s in PHONE_SET for s, _ in reference)
            totals["phones"] += phones
            totals["phone_edits"] += phones
            totals["stressed_edits"] += phones
            totals["sentences"] += 1
            split[part].update(sentences=1, tokens=len(reference), token_edits=len(reference), phones=phones, phone_edits=phones)
            continue
        mine = list(zip(ours.symbols, ours.stress))
        owners = [unit for kind, value in analysis if kind == "clause" for unit in value for _ in unit.phones]
        ref_phones = [(s, x) for s, x in reference if s in PHONE_SET]
        our_phones = [(s, x) for s, x in mine if s in PHONE_SET]
        token_edits = edit_distance([s for s, _ in mine], [s for s, _ in reference])
        pairs = alignment([s for s, _ in our_phones], [s for s, _ in ref_phones])
        phone_edits = sum(1 for i, j in pairs if i is None or j is None or our_phones[i][0] != ref_phones[j][0])
        stressed_edits = edit_distance(our_phones, ref_phones)
        totals["sentences"] += 1
        totals["tokens"] += len(reference)
        totals["token_edits"] += token_edits
        totals["phones"] += len(ref_phones)
        totals["phone_edits"] += phone_edits
        totals["stressed_edits"] += stressed_edits
        totals["exact_tokens"] += token_edits == 0
        totals["exact_phones"] += phone_edits == 0
        totals["exact_all"] += mine == reference
        totals["boundaries_ours"] += ours.symbols.count("<sp>")
        totals["boundaries_reference"] += row["symbols"].count("<sp>")
        split[part].update(sentences=1, tokens=len(reference), token_edits=token_edits, phones=len(ref_phones),
                           phone_edits=phone_edits, identical=int(mine == reference))
        spans, current, source = defaultdict(list), None, "?"   # reference phones aligned to each of our units
        for i, j in pairs:
            if i is not None:
                current, source = id(owners[i]), owners[i].source
            edit = None
            if i is not None and j is not None:
                a, b = our_phones[i], ref_phones[j]
                if a[0] == b[0] and a[0] in VOWEL_SET:
                    totals["vowels"] += 1
                    totals["vowel_stress_same"] += a[1] == b[1]
                    if a[1] != b[1]:
                        totals[f"stress_{a[1]}_for_{b[1]}"] += 1
                        if part == "plain_prose":
                            kinds["stress only"][source] += 1
                elif a[0] != b[0]:
                    edit = (a[0], b[0])
            elif i is None:
                edit = ("-", ref_phones[j][0])
            else:
                edit = (our_phones[i][0], "-")
            if edit:
                confusions[edit] += 1
                if part == "plain_prose":
                    kinds[kind_of(*edit)][source] += 1
            if j is not None:
                key = current if current is not None else id(owners[0]) if owners else None
                spans[key].append(ref_phones[j])
        for kind, value in analysis:
            if kind != "clause":
                continue
            for unit in value:
                theirs = to_text(spans.get(id(unit), []))
                same = to_text(unit.phones) == theirs
                by_source[unit.source]["units"] += 1
                by_source[unit.source]["different"] += not same
                if not same:
                    key = (unit.word, unit.source, to_text(unit.phones), theirs)
                    words[key] += 1
                    examples.setdefault(key, row["text"])

    n = totals
    summary = {
        "sentences": n["sentences"], "failed": len(failures), "join_phrases": not args.no_phrases,
        "token_agreement": round(1 - n["token_edits"] / n["tokens"], 5),
        "phone_agreement": round(1 - n["phone_edits"] / n["phones"], 5),
        "phone_and_stress_agreement": round(1 - n["stressed_edits"] / n["phones"], 5),
        "stress_agreement_on_shared_vowels": round(n["vowel_stress_same"] / max(n["vowels"], 1), 5),
        "sentences_identical": round(n["exact_all"] / n["sentences"], 4),
        "sentences_identical_ignoring_stress": round(n["exact_tokens"] / n["sentences"], 4),
        "sentences_identical_phones_only": round(n["exact_phones"] / n["sentences"], 4),
        "reference_tokens": n["tokens"], "reference_phones": n["phones"],
        "word_boundaries": {"ours": n["boundaries_ours"], "reference": n["boundaries_reference"]},
        "stress_differences": {key: value for key, value in sorted(n.items()) if key.startswith("stress_")},
        "units_by_source": {source: {"units": c["units"], "differ": round(c["different"] / c["units"], 4)} for source, c in sorted(by_source.items())},
        # Sentences with digits or symbols are read out differently on purpose (years, money, times),
        # so the kinds of difference below are counted on plain prose only.
        "by_kind_of_sentence": {name: {"sentences": c["sentences"], "token_agreement": round(1 - c["token_edits"] / c["tokens"], 5),
                                       "phone_agreement": round(1 - c["phone_edits"] / c["phones"], 5),
                                       "identical": round(c["identical"] / c["sentences"], 4)} for name, c in sorted(split.items())},
        "differences_in_plain_prose": {kind: {"phones": sum(c.values()), **dict(c.most_common())}
                                       for kind, c in sorted(kinds.items(), key=lambda item: -sum(item[1].values()))},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.output / "confusions.tsv").write_text("".join(f"{count}\t{ours}\t{theirs}\n" for (ours, theirs), count in confusions.most_common()), encoding="utf-8")
    (args.output / "words.tsv").write_text("".join(
        f"{count}\t{word}\t{source}\t{ours}\t{theirs}\t{examples[(word, source, ours, theirs)]}\n"
        for (word, source, ours, theirs), count in words.most_common()), encoding="utf-8")
    (args.output / "failures.json").write_text(json.dumps(failures, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("confusions (ours -> eSpeak):", "  ".join(f"{a}>{b}:{c}" for (a, b), c in confusions.most_common(args.show)))
    for (word, source, ours, theirs), count in words.most_common(args.show):
        print(f"{count:4d}  {word} [{source}]  {ours}  ||  {theirs}")
    if failures:
        print("failures:", json.dumps(failures[:5], ensure_ascii=False))


if __name__ == "__main__":
    main()
