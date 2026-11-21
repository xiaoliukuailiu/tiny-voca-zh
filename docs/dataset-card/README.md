---
language: en
task_categories:
- text-to-speech
license: cc-by-4.0
pretty_name: tiny-voca
configs:
- config_name: batch1
  default: true
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
- config_name: batch2
  data_files:
  - split: train
    path: batch2/data/train-*.parquet
---
# tiny-voca dataset

A synthetic American English speech corpus with two voices, F2 (female) and M2
(male), made to train the tiny-voca text-to-speech model. Every sentence is
read by both voices.

| | |
|---|---|
| Clips | 17,371 |
| Duration | 19.1 hours |
| Voices | F2, M2 |
| Audio | 24 kHz mono, 16-bit FLAC, stored inside the Parquet files |
| Language | English (US) |
| License | CC BY 4.0 |

## Usage

```python
from datasets import load_dataset

batch1 = load_dataset("n4ze3m/tiny-voca", "batch1")
batch2 = load_dataset("n4ze3m/tiny-voca", "batch2")

row = batch1["train"][0]
print(row["voice_id"], row["text"])
audio = row["audio"]["array"]
```

## Contents

| Config | Split | Clips | Hours | Text |
|---|---|---:|---:|---|
| `batch1` | train | 8,226 | | LibriTTS-R transcripts and LLM-written sentences |
| `batch1` | validation | 718 | | same |
| `batch1` | test | 596 | | same |
| `batch2` | train | 7,831 | 7.6 | Mozilla Common Voice sentences and project templates |

The three `batch1` splits total 11.5 hours. Splits are made by sentence: both
readings of a sentence are always in the same split. `batch2` adds training
data only, so the validation and test sentences are the same for every model
trained on this corpus.

## How it was made

1. The two voices were created from text descriptions with
   Qwen3-TTS 1.7B VoiceDesign. They are not recordings or imitations of real
   people. The reference clips are in `references/`, and the descriptions and
   seeds are in `voices.json`.
2. Each sentence was synthesized in both voices with Qwen3-TTS 1.7B Base,
   using the reference clips as the voice prompt.
3. Each clip was transcribed with Whisper small.en. Clips above 15% word error
   rate, or with implausible duration, speaking rate or silence, were generated
   a second time. Clips that failed twice were left out: 270 of 9,810 in
   `batch1` and 327 of 8,158 in `batch2`. They are listed in
   `generation/batch1/review.jsonl` and `generation/batch2/review.jsonl`.
4. Accepted clips were peak-normalized to 0.95.

The generation scripts and their settings are in `generation/`.

The automatic screening catches wrong or missing words. It does not measure
naturalness, and the corpus has not been checked by listeners clip by clip.

## Text sources

The `origin`, `source_id`, `source_license` and `source_url` columns record the
source of every row.

| Config | Source | Sentences | License of the text |
|---|---|---:|---|
| `batch1` | LibriTTS-R transcripts (book sentences) | | CC BY 4.0 |
| `batch1` | Written for this project with an LLM (DeepSeek), mostly conversational | | |
| `batch2` | Mozilla Common Voice sentence collector | 2,700 | CC0 |
| `batch2` | Mozilla Common Voice Wikipedia sentences | 779 | CC0 |
| `batch2` | Templates written for this project: numbers, dates, money, names, questions | 600 | |

`batch1` has 4,905 sentences in total. A quarter of the `batch2` sentences are
questions and a tenth are exclamations, which `batch1` was short of.

## Columns

| Column | Description |
|---|---|
| `id` | Clip ID, `<text_id>_<voice_id>` |
| `text_id`, `voice_id` | Sentence and voice |
| `text` | The sentence that was read |
| `split` | train, validation or test |
| `category`, `origin` | Kind of sentence and where it came from |
| `source_id`, `source_license`, `source_url` | Source attribution |
| `audio` | 24 kHz audio |
| `duration_seconds`, `sample_rate` | Audio length and rate |
| `asr_transcript`, `normalized_wer` | Whisper transcript and its word error rate against `text` |
| `attempt`, `seed`, `batch_id` | Generation attempt and random seed |
| `teacher_model`, `reference_audio` | Model and voice prompt used |

`batch1` also has `requested_style` (a style hint, not a verified label) and
`qc_reasons` (always empty for accepted clips).

## Annotations

`annotations/` holds the phoneme-level annotations used for training:
phonemes, stress, and the duration of each phoneme in 10.7 ms frames, from
forced alignment with `facebook/wav2vec2-lv-60-espeak-cv-ft`. Clips that
failed the alignment checks are listed with a reason in the `*.rejected.jsonl`
files.

| Directory | Covers | Phonemes from | Accepted clips |
|---|---|---|---:|
| `annotations/v3` | `batch1` | eSpeak NG | 9,490 of 9,540 |
| `annotations/v4` | `batch1` and `batch2` | dizione | 17,326 of 17,371 |

## Limitations

- The speech is synthetic. It carries the pronunciation habits and errors of
  the model that generated it.
- Two voices, one accent, read speech only.
- Digits, abbreviations and symbols are rare: most numbers in the text are
  written out in words.

## License

CC BY 4.0. Please credit "tiny-voca dataset, Muhammed Nazeem" and link to this
page.

```bibtex
@misc{nazeem2026tinyvocadata,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca dataset: synthetic two-voice American English speech},
  year   = {2026},
  url    = {https://huggingface.co/datasets/n4ze3m/tiny-voca}
}
```

Author: Muhammed Nazeem ([ORCID 0009-0002-9991-0217](https://orcid.org/0009-0002-9991-0217)).

The LibriTTS-R transcripts are used under CC BY 4.0: Koizumi et al.,
"LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", Interspeech 2023,
https://www.openslr.org/141/. The Common Voice sentences are CC0, from
https://github.com/common-voice/common-voice. The audio was generated with
Qwen3-TTS (Apache-2.0). See `LICENSE.md`.
