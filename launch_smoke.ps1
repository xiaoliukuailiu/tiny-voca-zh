# Launch script for smoke test with correct Python and offline mode
$env:PYTHON = "C:\Users\admin405\.conda\envs\qwen_audio\python.exe"
$env:HF_HOME = "E:\huggface_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TINY_VOCA_LANG = "zh"
$env:PYTHONPATH = "src"

Set-Location "D:\Desktop\tiny-voca-zh\tiny-voca"
.\run_all.ps1 smoke
