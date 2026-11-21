---
license: apache-2.0
language:
- en
pipeline_tag: text-to-speech
tags:
- text-to-speech
- onnx
datasets:
- n4ze3m/tiny-voca
---
# tiny-voca

tiny-voca is a small English text-to-speech model with two voices, F2 (female)
and M2 (male). The complete model, including the waveform generator, has 4.5M
parameters and was trained from scratch on about 18 hours of synthetic speech.
It runs about 70x faster than real time on four CPU threads.

<audio controls src="https://huggingface.co/n4ze3m/tiny-voca/resolve/main/samples/F2-1.wav"></audio>

- Code: https://github.com/n4ze3m/tiny-voca
- Phonemizer: https://github.com/n4ze3m/dizione
- Dataset: https://huggingface.co/datasets/n4ze3m/tiny-voca

| | |
|---|---|
| Parameters | 4,537,291 |
| Voices | F2, M2 (American English) |
| Output | 24 kHz mono |
| Input | Text, converted to phonemes with dizione |
| Pretrained weights in the model | None |
| License | Apache-2.0 |

## Files

| File | Size | Contents |
|---|---:|---|
| `tiny-voca.pt` | 18.2 MB | PyTorch weights, float32 |
| `onnx/model.onnx` | 21.4 MB | ONNX export, float32, opset 17 |
| `config.json` | | Model settings, phoneme vocabulary, voices |
| `phonemizer/` | 5.4 MB | Data files for dizione |
| `samples/` | | The audio on this page |

SHA256:

- `tiny-voca.pt`: `SHA_PT`
- `onnx/model.onnx`: `SHA_ONNX`

## Samples

| Text | F2 | M2 |
|---|---|---|
SAMPLE_ROWS

## Usage

### PyTorch

```bash
pip install git+https://github.com/n4ze3m/dizione git+https://github.com/n4ze3m/tiny-voca
hf download n4ze3m/tiny-voca --local-dir tiny-voca
export DIZIONE_DATA=tiny-voca/phonemizer

python -m tiny_voca.infer \
  --checkpoint tiny-voca/tiny-voca.pt \
  --voice F2 \
  --text "Hello! This is tiny voca." \
  --output hello.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text`. |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Added to log F0. 0.058 is about one semitone. |
| `--pitch-range` | 1.0 | Scales pitch movement around the utterance mean. |

Text of any length works. The command reads it one sentence at a time and
joins the audio. A text that ends without a mark gets a full stop. A sentence
with nothing to read, such as one in another script, is skipped. The saved
audio never peaks above 0.99.

### ONNX

The graph takes phoneme ids and stress values and returns the waveform. Text is
converted to ids outside the graph with dizione; the ids are the positions in
`vocab` in `config.json`.

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
import dizione   # needs DIZIONE_DATA=tiny-voca/phonemizer

phones = dizione.phonemize("Hello! This is tiny voca.")
one = lambda value: np.array([value], dtype=np.float32)

session = ort.InferenceSession("tiny-voca/onnx/model.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = F2, 1 = M2
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
})
audio = audio[0]
audio = audio * min(1.0, 0.99 / np.abs(audio).max())    # the model can peak above 1.0
sf.write("hello.wav", audio, 24000)
```

| Input | Type | Shape | Meaning |
|---|---|---|---|
| `tokens` | int64 | [1, N] | Phoneme ids, including the start and end tokens |
| `stress` | int64 | [1, N] | 0 none, 1 primary, 2 secondary |
| `voice` | int64 | [1] | 0 = F2, 1 = M2 |
| `speed` | float32 | [1] | 1.0 as predicted; larger is faster |
| `pitch_shift` | float32 | [1] | Added to log F0 |
| `pitch_range` | float32 | [1] | Scales pitch movement |

| Output | Type | Shape | Meaning |
|---|---|---|---|
| `audio` | float32 | [1, samples] | Waveform at 24 kHz |
| `durations` | int64 | [1, N] | Length of each token in frames of 256 samples |

The model reads one utterance per call, up to 512 tokens and 32 seconds. Split
longer text into sentences; `tiny_voca.infer.speak_text` does this in Python.
End each utterance with a mark such as a full stop: without one the last word
breaks off.
The audio can peak a little above 1.0 (about one sentence in twenty), so scale
it down before you save 16-bit samples, as above. The same file runs in
onnxruntime-web (WebAssembly). Its audio matches the PyTorch model to within
0.6% of the peak amplitude, apart from the random noise signal, which each
runtime draws itself.

## Model

Text is converted to phonemes with a stress value for each. A transformer
encoder (4 blocks) reads the phonemes, a duration predictor gives each one a
length in frames, and two frame-level transformers (2 and 5 blocks) predict
pitch, voicing and energy and then the parameters of the waveform generator.
All blocks use 192 channels, 4 heads and rotary position embeddings.

The waveform is produced by a source-filter synthesizer instead of a neural
vocoder. A band-limited pulse train at the predicted pitch and a noise signal
are each shaped by a time-varying filter, whose cepstrum the network predicts
once per frame (256 samples, 10.7 ms). This part has 25,862 parameters.

| Component | Parameters |
|---|---:|
| Embeddings (phoneme, stress, voice) | 16,704 |
| Text encoder | 1,588,864 |
| Duration and pause predictor | 123,906 |
| Frame transformer A, prosody head | 795,971 |
| Frame transformer B | 1,985,984 |
| Source-filter waveform head | 25,862 |
| Total | 4,537,291 |

## Training

| | |
|---|---|
| Data | [n4ze3m/tiny-voca](https://huggingface.co/datasets/n4ze3m/tiny-voca): 16,018 clips, 17.6 hours, two voices |
| Steps | 104,737 |
| Hardware | One RTX 4090, 7.7 hours |
| Losses | Multi-resolution spectral losses; adversarial loss with multi-period and multi-resolution discriminators after step 8,000 |

The training audio is synthetic. Both voices were created from text
descriptions with Qwen3-TTS VoiceDesign, and the recordings were generated with
Qwen3-TTS 1.7B Base. No recording of a real speaker is involved. Phoneme
durations come from forced alignment with
`facebook/wav2vec2-lv-60-espeak-cv-ft`, which is used only to label the
training data. No pretrained weights are part of the model.

## Evaluation

Word error rate of synthesized speech, transcribed with Whisper small.en. The
recordings of the same sentences are scored the same way for reference.

| Sentences | Clips | WER, tiny-voca | WER, recordings |
|---|---:|---:|---:|
| Validation | 8 | 0.6% | 0.6% |
| Test | 8 | 0.9% | 0.9% |
| New sentences, not in the dataset | 16 | 0.0% | |

These sets are small and the scores are a sanity check, not a benchmark. No
formal listening test has been run.

## Limitations

- English only, two fixed voices, no voice cloning.
- The voice is clear but can sound slightly synthetic.
- The F2 voice sounds younger than intended. It was designed as an adult voice
  (the prompt asked for an adult American woman in her mid twenties; see
  voices.json in the dataset). A more mature voice is planned.
- The same text is always read the same way. There is no sampling of prosody.
- Speech is slightly faster than the training recordings at the default speed.
- Foreign names and unusual spellings are often mispronounced. Numbers,
  abbreviations and homographs depend on the phonemizer.
- No streaming.
- The training data is itself synthetic, so the model cannot be more natural
  than the model that generated it.

## Cost

The whole project cost $9.56 of GPU time on one rented RTX 4090. That is the
dataset, the failed tries and the final training run.

![Cost breakdown](https://huggingface.co/n4ze3m/tiny-voca/resolve/main/cost.png)

## Support

It cost less than a pizza to make. If you like it, you can buy me one:
https://ko-fi.com/n4ze3m

## License

The weights and code are released under Apache-2.0. The files in `phonemizer/`
are derived from the CMU Pronouncing Dictionary and other sources; their
licenses are in `phonemizer/LICENSE.cmudict` and
`phonemizer/LICENSE.homograph-data`.

## Citation

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```

Author: Muhammed Nazeem ([ORCID 0009-0002-9991-0217](https://orcid.org/0009-0002-9991-0217)).

## Acknowledgements

The design builds on FastSpeech 2 (explicit duration and prosody prediction),
DDSP and the neural homomorphic vocoder (source-filter synthesis with predicted
filters), and HiFi-GAN and UnivNet (discriminators). No code or weights from
these projects are used.
