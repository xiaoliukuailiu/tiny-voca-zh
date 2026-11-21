#!/usr/bin/env bash
# Fetch the files the demo serves into public/assets (they are not stored in git).
#
#   bash fetch_assets.sh /path/to/dizione
#
# Downloads the ONNX model, its config and the phonemizer data from the Hugging
# Face repository, and zips the dizione package for Pyodide. While the model
# repository is private, the token of the logged-in Hugging Face user is used.
set -euo pipefail
cd "$(dirname "$0")"
DIZIONE="${1:?path to the dizione repository}"
BASE="https://huggingface.co/n4ze3m/tiny-voca/resolve/main"
TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
mkdir -p public/assets/phonemizer
get() {
  if [ -f "$TOKEN_FILE" ]; then
    curl -fsSL -H "Authorization: Bearer $(cat "$TOKEN_FILE")" -o "$2" "$BASE/$1"
  else
    curl -fsSL -o "$2" "$BASE/$1"
  fi
}
get onnx/model.onnx public/assets/model.onnx
get config.json public/assets/config.json
for name in lexicon.tsv align.json g2p.json g2p.npz homographs.json LICENSE.cmudict LICENSE.homograph-data; do
  get "phonemizer/$name" "public/assets/phonemizer/$name"
done
rm -f public/assets/dizione.zip
(cd "$DIZIONE/src" && zip -q -r - dizione -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' '*/built/*') > public/assets/dizione.zip
ls -l public/assets public/assets/phonemizer
