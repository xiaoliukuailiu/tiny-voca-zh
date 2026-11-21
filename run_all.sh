#!/usr/bin/env bash
# End-to-end Mandarin pipeline.
#
#   bash run_all.sh              # everything
#   bash run_all.sh smoke        # ~20 sentences, 2k steps: proves the wiring
#   bash run_all.sh synth        # one stage: corpus|synth|split|prepare|align|annotate|train|eval|speak
#
# Every stage is resumable: re-running skips what already exists.
set -euo pipefail

export TINY_VOCA_LANG=zh
export PYTHONPATH="${PYTHONPATH:-}:src"
PY="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA="${DATA:-data/zh-corpus}"
FEATURES="${FEATURES:-features}"
MFA_OUT="${MFA_OUT:-mfa-out}"
MFA_CORPUS="${MFA_CORPUS:-mfa-corpus}"
ANNOT="${ANNOT:-annotations}"
RUN="${RUN:-runs/zh}"
CONFIG="${CONFIG:-config/zh.json}"
MFA_DICT="${MFA_DICT:-mandarin_mfa.dict}"
MFA_ACOUSTIC="${MFA_ACOUSTIC:-mandarin_mfa.zip}"
JOBS="${JOBS:-8}"
# ctc: in-process character aligner, works everywhere. mfa: needs the binary.
ALIGNER="${ALIGNER:-ctc}"

STAGE="${1:-all}"
STEPS="${STEPS:-104000}"
GAN_START="${GAN_START:-8000}"
LIMIT="${LIMIT:-0}"

if [ "$STAGE" = "smoke" ]; then
  STAGE=all
  LIMIT=20
  STEPS=2000
  GAN_START=500
  DATA=data/zh-corpus-smoke
  FEATURES=features-smoke
  MFA_OUT=mfa-out-smoke
  MFA_CORPUS=mfa-corpus-smoke
  ANNOT=annotations-smoke
  RUN=runs/zh-smoke
  echo "== smoke mode: 20 sentences, $STEPS steps, separate output directories =="
fi

limit_args=()
if [ "$LIMIT" -gt 0 ]; then limit_args=(--limit "$LIMIT"); fi

section() { printf '\n== %s ==\n' "$1"; }

section "0. environment"
"$PY" scripts/doctor.py --stage all

if [ "$STAGE" = "all" ] || [ "$STAGE" = "corpus" ]; then
  section "1. text corpus"
  [ -f data/zh_sentences.txt ] || "$PY" data/make_zh_corpus.py --count 20000 --output data/zh_sentences.txt
  wc -l data/zh_sentences.txt
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "synth" ]; then
  section "2. synthesize with Qwen3-TTS"
  mkdir -p "$DATA"
  "$PY" scripts/make_zh_dataset.py --texts data/zh_sentences.txt --output "$DATA" \
      --templates 400 --batch 8 "${limit_args[@]}"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "synth" ] || [ "$STAGE" = "split" ]; then
  section "3. split train/validation/test on text"
  CORPUS="$DATA/corpus.jsonl"
  [ -f "$DATA/corpus.screened.jsonl" ] && CORPUS="$DATA/corpus.screened.jsonl"
  "$PY" scripts/split_zh_corpus.py --corpus "$CORPUS"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "prepare" ]; then
  section "4. acoustic features"
  for split in train validation test; do
    if [ -f "$DATA/$split.jsonl" ]; then
      "$PY" -m tiny_voca.prepare --inputs "$DATA/$split.jsonl" --split "$split" \
          --output "$FEATURES" --config "$CONFIG" --resume
    fi
  done
fi

if [ "$ALIGNER" = "mfa" ] && { [ "$STAGE" = "all" ] || [ "$STAGE" = "align" ]; }; then
  section "5. export MFA corpus and align"
  "$PY" -m tiny_voca.zh_annotate --features "$FEATURES" --export-corpus "$MFA_CORPUS" \
      --config "$CONFIG"
  for split in train validation test; do
    if [ -d "$MFA_CORPUS/$split" ]; then
      mfa align "$MFA_CORPUS/$split" "$MFA_DICT" "$MFA_ACOUSTIC" "$MFA_OUT/$split" \
          --single_speaker --jobs "$JOBS" --output_format long_textgrid --clean
    fi
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "annotate" ]; then
  section "6. annotation manifests ($ALIGNER)"
  if [ "$ALIGNER" = "mfa" ]; then
    "$PY" -m tiny_voca.zh_annotate --features "$FEATURES" --textgrids "$MFA_OUT" \
        --output "$ANNOT" --config "$CONFIG" --aligner mfa
  else
    "$PY" -m tiny_voca.zh_annotate --features "$FEATURES" \
        --output "$ANNOT" --config "$CONFIG" --aligner ctc
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
  section "7. train"
  "$PY" -m tiny_voca.train --train "$ANNOT/train.jsonl" --validation "$ANNOT/validation.jsonl" \
      --output "$RUN" --config "$CONFIG" --steps "$STEPS" --gan-start "$GAN_START" \
      --gan-ratio 0.7 --teacher-final 0 --device cuda
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
  section "8. evaluate"
  if [ -f "$RUN/inference.pt" ]; then
    "$PY" -m tiny_voca.evaluate --checkpoint "$RUN/inference.pt" --annotations "$ANNOT" \
        --output "eval/zh" --tag zh --device cuda --asr-model small
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "speak" ]; then
  section "9. demo"
  if [ -f "$RUN/inference.pt" ]; then
    "$PY" -m tiny_voca.infer --checkpoint "$RUN/inference.pt" --voice ZH_F1 \
        --text "你好，这是一个很小的中文语音模型。它只有四百多万个参数。" \
        --output demo-zh.wav --device cpu
    echo "wrote demo-zh.wav"
  fi
fi

echo
echo "done. checkpoint: $RUN/inference.pt"
