"""Try the ONNX export and print the real error."""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TINY_VOCA_LANG", "zh")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tiny_voca.export import export
from tiny_voca.infer import load_model, match_frontend

match_frontend(r"runs\zh\inference.pt")
model = load_model(r"runs\zh\inference.pt", "cpu")
try:
    export(model, Path(r"web\public\assets\model.onnx"))
    print("EXPORT OK")
except Exception as error:
    print("ERROR:", type(error).__name__)
    print(str(error)[:2000])
