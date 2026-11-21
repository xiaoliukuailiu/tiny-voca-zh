# tiny-voca annotations v3: phoneme tokens and validated durations

Annotation layer for the recordings in this dataset at revision
`93dfa15bf40ec278a3ea1b418294c5620d850a8f`. It adds, for every accepted
recording, the phoneme token sequence and the number of acoustic frames each
token lasts. The audio shards are untouched; nothing here replaces them.

The repository history was squashed before release, so that revision can no
longer be fetched. The batch 1 audio shards in `data/` are the same files in
the current revision.

Annotation fingerprint: `ebed758f71eb2af6d241a1badd9ee951ceb05e7605ad719507d77468a6341bdd`
(training checkpoints record it and refuse manifests from another run).

## What was wrong before

* The old front end piped text to `espeak-ng --stdin` without a final newline.
  eSpeak NG 1.51 then drops the last character of the input, so the last
  phonemes of every clause were wrong ("lips" became `l ɪ p`, "forever" became
  `f ɔː ɹ v`). 294 of 300 sampled rows were affected.
* Tokens were the UTF-8 bytes of the IPA string: 2.5 byte tokens per phoneme,
  3.3 frames per token, and each byte (including stress marks, length marks and
  spaces) had to own at least one frame.
* Durations came from a learned aligner that collapsed in all three attempts.

## How these annotations were made

1. **Text to phonemes.** eSpeak NG 1.51, voice `en-us`, `--ipa --sep=_`, input
   terminated by a newline. One token per eSpeak phoneme (59 phones), stress as
   a separate 0/1/2 feature, explicit tokens for utterance edges, word
   boundaries and punctuation. 76 symbols in total; no unknown phoneme occurs
   in the corpus.
2. **Forced alignment.** A pretrained phoneme recogniser,
   `facebook/wav2vec2-lv-60-espeak-cv-ft` (revision
   `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`, Apache-2.0), aligned each phoneme
   sequence to its recording with CTC Viterbi. **It was used only to annotate
   this data. None of its weights or outputs are part of the TTS model.** Its
   label inventory is the same eSpeak phoneme set, so one aligned label is one
   token.
3. **Boundaries.** This recogniser emits each label at the start of the phone.
   Measured against the pitch tracker's voicing flips on 300 recordings, a
   phone starts 19 ms before the start of its first emission frame (both
   voices). A phone lasts from its onset to the next phone's onset.
4. **Pauses.** Silence (frame level below the recording's loud level minus
   45 dB) between two phones is given to the pause-capable token there: BOS,
   EOS, punctuation, or the word boundary. At least 75 ms of silence is needed,
   otherwise the gap stays with the phones (stop closures).
5. **Frame grid.** Durations are counted in 256-sample frames at 24 kHz, frame
   `i` centred on sample `i * 256`, the same grid as the stored mel and pitch
   features. They sum to the frame count exactly and every phoneme has at least
   one frame.

A second pass that re-timed boundaries with a small classifier trained from
scratch on these labels was evaluated (`refined-1`, `refined-2`). It did not
improve the timing probe below, so it is **not** used. The files are kept for
reference.

## Quality checks

| Split | Accepted | Rejected | Hours accepted |
|---|---:|---:|---:|
| train | 8,186 | 40 | 10.02 |
| validation | 714 | 4 | 0.74 |
| test | 590 | 6 | 0.70 |

Rejected rows are listed with reasons in `<split>.rejected.jsonl`: 30 where the
pitch tracker found voicing in under half of the vowel frames (creaky or
breathy recordings), 19 with more than a quarter of phonemes weakly matched and
2 with low mean confidence. The weak matches are mostly numbers that eSpeak and
the teacher TTS read differently ("1-800-555-0147", "In 1681", "12/09/2026").
Sentence-level split isolation is unchanged: no text occurs in two splits.

Label-free timing probe over all accepted rows: at vowel to voiceless-consonant
and voiceless-fricative to vowel boundaries the voicing must flip.

| Voice | Boundary type | Boundaries | Flip within 1 frame (10.7 ms) | within 2 frames |
|---|---|---:|---:|---:|
| F2 | vowel to voiceless | 29,597 | 63.8% | 85.6% |
| F2 | fricative to vowel | 10,625 | 66.3% | 89.0% |
| M2 | vowel to voiceless | 29,617 | 58.0% | 81.2% |
| M2 | fricative to vowel | 10,685 | 65.9% | 85.6% |

The probe includes the pitch tracker's own jitter, so these are lower bounds on
boundary accuracy, not a comparison with hand labels (there are none).

Other statistics (`summary.json` has all of them, per voice and per phoneme):
mean phone length 7.5 frames (F2) and 7.0 frames (M2); 0.2% of phones last one
frame (97 to 99% did in the collapsed runs); a word boundary carries a pause in
0.8% of cases, a comma in 47%, a semicolon in 76%.

## Files

| File | Content |
|---|---|
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | accepted rows: `id`, `speaker`, `text`, `symbols`, `tokens`, `stress`, `durations`, `frames`, `qc`, `provenance`, fingerprints |
| `*.rejected.jsonl` | rejected rows with reasons |
| `annotations.json` | identity of this run (front end, aligner revision, settings, source hashes), completion flag |
| `summary.json` | QC quantiles, timing probe, duration tables |
| `phonemes.jsonl` | front-end output per distinct text |
| `raw/` | per-recording alignment journal (resumable) |
| `refined-1/`, `refined-2/` | the evaluated, unused refinement passes |

`path` in a row is relative to the acoustic feature directory on the training
machine. Provenance fields (`text_id`, `origin`, `source_id`, `source_license`,
`source_url`, `teacher_model`, `reference_audio`) are copied from the source
rows; the dataset's `LICENSE.md` continues to apply.

## Limits

* No hand-labelled boundaries exist for this corpus; accuracy is estimated from
  the voicing probe and from whether a TTS model trained on these durations
  speaks intelligibly.
* A pause inside an eSpeak word group has no token and stays with a phone.
* Where eSpeak and the teacher TTS pronounce a word differently (heteronyms,
  some numbers), the token is eSpeak's and the audio is the teacher's.

Reproduce on the training machine:

```bash
python -m tiny_voca.annotate --features <acoustic feature dir> \
  --output <new directory> --dataset-revision 93dfa15bf40ec278a3ea1b418294c5620d850a8f
```
