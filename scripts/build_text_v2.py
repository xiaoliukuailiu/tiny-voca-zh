"""Build the script list for the second audio batch. Cloud only.

Sources, both free of third-party licence obligations:
* Mozilla Common Voice sentence files (CC0-1.0): ``sentence-collector.txt`` and
  ``wiki.en.txt`` from the common-voice repository, given as local files.
* Template sentences written for this project (numbers, dates, money, names,
  questions, lists), released with the project.

Every script is new (not in the first batch), passes the front end without
unknown phonemes, and goes to the training split only, so the existing
validation and test sentences stay untouched and comparable.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from tiny_voca.text import phonemize

OK = re.compile(r"^[A-Z][A-Za-z ,;'\-]+[.?!]$")
FIRST = ("Nazeem Maya Liam Olivia Noah Ava Ethan Sofia Lucas Amara Diego Priya Kenji Fatima Omar Chloe Mateo Aisha Ravi Elena "
         "Jamal Ingrid Tariq Naomi Felix Zara Hugo Leila Andre Mei Samir Clara Idris Hana Marco Yusuf Nora Dmitri Anika Caleb").split()
LAST = "Johnson Patel Garcia Kim Nguyen Okafor Rossi Haddad Tanaka Silva Brooks Novak Ibrahim Larsen Mendoza Clarke".split()
PLACES = ("Seattle Austin Denver Chicago Boston Phoenix Atlanta Portland Nashville Miami Toronto Dublin Nairobi Mumbai Lisbon "
          "the library the airport the pharmacy the bakery the train station the farmers market").split("  ")
PLACES = ["Seattle", "Austin", "Denver", "Chicago", "Boston", "Phoenix", "Atlanta", "Portland", "Nashville", "Miami", "Toronto",
          "Dublin", "Nairobi", "Mumbai", "Lisbon", "the library", "the airport", "the pharmacy", "the bakery", "the train station"]
ITEMS = ["apples", "batteries", "notebooks", "tickets", "blankets", "sandwiches", "umbrellas", "candles", "chairs", "postcards"]
ONES = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
MONTHS = "January February March April May June July August September October November December".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
            "twentieth twenty first twenty second twenty third thirtieth").split("  ")
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
            "fifteenth", "twentieth", "twenty first", "twenty third", "thirtieth"]


def number(rng, low=1, high=999):
    n = rng.randint(low, high)
    words = []
    if n >= 100:
        words += [ONES[n // 100 - 1], "hundred"]
        n %= 100
        if n:
            words.append("and")
    if n >= 20:
        words.append(TENS[n // 10 - 2] + (" " + ONES[n % 10 - 1] if n % 10 else ""))
    elif n:
        words.append(ONES[n - 1])
    return " ".join(words)


def clock(rng):
    hour, minute = ONES[rng.randint(0, 11)], rng.choice(["o'clock", "fifteen", "thirty", "forty five", "oh five", "twenty"])
    return f"{hour} {minute} {rng.choice(['in the morning', 'in the afternoon', 'at night', 'sharp'])}"


def templates(rng, count):
    name = lambda: rng.choice(FIRST)
    full = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    patterns = [
        lambda: f"Hello {name()}, welcome back! How was your day?",
        lambda: f"{name()}, could you call {full()} before {clock(rng)}?",
        lambda: f"The meeting with {full()} moved to {rng.choice(DAYS)}, {rng.choice(MONTHS)} {rng.choice(ORDINALS)}.",
        lambda: f"It costs {number(rng)} dollars and {number(rng, 1, 99)} cents, including tax.",
        lambda: f"We need {number(rng, 2, 60)} {rng.choice(ITEMS)}, {number(rng, 2, 60)} {rng.choice(ITEMS)}, and {number(rng, 2, 60)} {rng.choice(ITEMS)}.",
        lambda: f"Is the flight to {rng.choice(PLACES[:15])} leaving at {clock(rng)}, or was it delayed again?",
        lambda: f"My name is {full()}, and I live about {number(rng, 2, 90)} miles from {rng.choice(PLACES[:15])}.",
        lambda: f"Did {name()} really finish all {number(rng, 3, 40)} chapters in {number(rng, 2, 12)} days?",
        lambda: f"Please bring {number(rng, 2, 20)} {rng.choice(ITEMS)} to {rng.choice(PLACES)} by {rng.choice(DAYS)}.",
        lambda: f"On {rng.choice(MONTHS)} {rng.choice(ORDINALS)}, {name()} and {name()} are driving to {rng.choice(PLACES[:15])}.",
        lambda: f"That's {number(rng, 100, 999)} steps, give or take {number(rng, 2, 30)}!",
        lambda: f"Who told {name()} that the store closes at {clock(rng)}?",
        lambda: f"First we stop at {rng.choice(PLACES)}, then {rng.choice(PLACES)}, and finally {rng.choice(PLACES)}.",
        lambda: f"{full()} scored {number(rng, 40, 100)} percent, which is {number(rng, 2, 30)} points higher than last time.",
        lambda: f"Wait, are you saying {name()} left {number(rng, 2, 50)} minutes ago?",
        lambda: f"I asked {name()}, {name()}, and {name()} to meet us at {rng.choice(PLACES)}.",
        lambda: f"The recipe needs {number(rng, 2, 12)} cups of flour, {number(rng, 2, 6)} eggs, and a pinch of salt.",
        lambda: f"Honestly, {name()}, I never expected {number(rng, 50, 900)} people to show up!",
        lambda: f"Can you remind me whether {full()} prefers {rng.choice(DAYS)} or {rng.choice(DAYS)}?",
        lambda: f"It rained for {number(rng, 2, 14)} days in {rng.choice(PLACES[:15])}, so we stayed inside and read.",
    ]
    out = set()
    while len(out) < count:
        out.add(rng.choice(patterns)())
    return sorted(out)


def usable(line):
    line = re.sub(r"\s+", " ", line.strip().replace("’", "'"))
    if not OK.match(line) or not 28 <= len(line) <= 170 or not 5 <= len(line.split()) <= 30:
        return None
    if re.search(r"\b[A-Z]{2,}\b|--|\b[b-hj-z]\b", line):          # acronyms, dashes, stray single letters
        return None
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--existing", type=Path, nargs="+", required=True, help="First-batch text files (jsonl with 'text')")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collector-count", type=int, default=2700)
    parser.add_argument("--wiki-count", type=int, default=900)
    parser.add_argument("--template-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    seen = {json.loads(line)["text"].strip().lower() for path in args.existing for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = []

    def take(lines, count, origin, url, category):
        pool = sorted({s for s in (usable(l) for l in lines) if s and s.lower() not in seen})
        rng.shuffle(pool)
        # A quarter questions and a tenth exclamations (the first batch was short of
        # both); the rest statements. Longer sentences first within each kind.
        quota = {"?": round(0.25 * count), "!": round(0.10 * count)}
        quota["."] = count - sum(quota.values())
        pool.sort(key=lambda s: len(s) < 45)
        taken = {"?": 0, "!": 0, ".": 0}
        kept = 0
        for text in pool:
            if kept == count:
                break
            if taken[text[-1]] >= quota[text[-1]]:
                continue
            taken[text[-1]] += 1
            try:
                phones = phonemize(text)
            except Exception:
                continue
            if phones.unknown or len(phones) > 300:
                continue
            seen.add(text.lower())
            identifier = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append({"id": identifier, "text": text, "original_text": text, "category": category, "origin": origin, "source_id": identifier,
                         "source_url": url, "source_license": "cc0-1.0" if origin.startswith("common_voice") else "project-own",
                         "group_id": origin + ":" + identifier, "split": "train", "llm_model": None, "llm_modified": False,
                         "word_count": len(text.split()), "character_count": len(text), "audio_status": "not_generated"})
            kept += 1
        return kept
    repo = "https://github.com/common-voice/common-voice/tree/main/server/data/en"
    counts = {"collector": take(args.collector.read_text(encoding="utf-8", errors="ignore").splitlines(), args.collector_count, "common_voice_sentence_collector", repo, "conversation"),
              "wiki": take(rng.sample(args.wiki.read_text(encoding="utf-8", errors="ignore").splitlines(), 60000), args.wiki_count, "common_voice_wiki", repo, "reading"),
              "templates": take(templates(rng, args.template_count * 2), args.template_count, "tiny_voca_templates", "https://huggingface.co/datasets/n4ze3m/tiny-voca", "pronunciation")}
    rows.sort(key=lambda r: (r["character_count"], r["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    questions = sum(r["text"].endswith("?") for r in rows)
    print(json.dumps({"scripts": len(rows), **counts, "questions": questions, "exclamations": sum(r["text"].endswith("!") for r in rows),
                      "characters": sum(r["character_count"] for r in rows), "median_characters": sorted(r["character_count"] for r in rows)[len(rows) // 2]}))


if __name__ == "__main__":
    main()
