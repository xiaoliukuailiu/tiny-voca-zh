# Native Windows pipeline (PowerShell).
#
#   .\run_all.ps1 smoke        # 20 sentences, 2k steps: proves the wiring in ~15 min
#   .\run_all.ps1              # everything
#   .\run_all.ps1 train        # one stage
#
# Unlike run_all.sh this does NOT need MFA or WSL: annotation uses the built-in
# character-level CTC aligner (--aligner ctc), which is pure PyTorch.
#
# Every stage is resumable; re-running skips what already exists.

param(
    [ValidateSet("all", "smoke", "corpus", "synth", "split", "prepare", "align", "annotate", "train", "eval", "speak")]
    [string]$Stage = "all",
    [int]$Limit = 0,
    [int]$Steps = 104000,
    [int]$GanStart = 8000,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$env:TINY_VOCA_LANG = "zh"
$env:PYTHONPATH = "src"

$py = "python"
if ($env:PYTHON) { $py = $env:PYTHON }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Data     = "data/zh-corpus"
$Features = "features"
$Annot    = "annotations"
$Run      = "runs/zh"
$Config   = "config/zh.json"

if ($Stage -eq "smoke") {
    $Stage = "all"
    $Limit = 20
    $Steps = 2000
    $GanStart = 500
    $Data = "data/zh-corpus-smoke"
    $Features = "features-smoke"
    $Annot = "annotations-smoke"
    $Run = "runs/zh-smoke"
    Write-Host "== smoke mode: 20 sentences, $Steps steps, separate output directories =="
}

if ($Limit -gt 0) { $limitArgs = @("--limit", "$Limit") } else { $limitArgs = @() }

function Step($Name) { Write-Host "`n== $Name ==" }

function Invoke-Python([string[]]$Arguments) {
    # PowerShell 5.1: a native command's stderr line becomes a NativeCommandError
    # record, and with $ErrorActionPreference = "Stop" it aborts the script even
    # when python exits 0. Lower it around the call; real failures still throw via
    # $LASTEXITCODE below.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py @Arguments
    $ErrorActionPreference = $previous
    if ($LASTEXITCODE -ne 0) { throw "python failed with exit code $LASTEXITCODE : $($Arguments -join ' ')" }
}

Step "0. environment"
Invoke-Python @("scripts/doctor.py", "--stage", "all")

if ($Stage -in @("all", "corpus")) {
    Step "1. text corpus"
    if (-not (Test-Path "data/zh_sentences.txt")) {
        Invoke-Python @("data/make_zh_corpus.py", "--count", "20000", "--output", "data/zh_sentences.txt")
    }
    $lines = (Get-Content "data/zh_sentences.txt").Count
    Write-Host "$lines sentences"
}

if ($Stage -in @("all", "synth")) {
    Step "2. synthesize with Qwen3-TTS"
    New-Item -ItemType Directory -Force -Path $Data | Out-Null
    Invoke-Python (@("scripts/make_zh_dataset.py", "--texts", "data/zh_sentences.txt",
                     "--output", $Data, "--templates", "400", "--batch", "32") + $limitArgs)
}

if ($Stage -in @("all", "synth", "split")) {
    Step "3. split train/validation/test on text"
    $corpus = Join-Path $Data "corpus.jsonl"
    $screened = Join-Path $Data "corpus.screened.jsonl"
    if (Test-Path $screened) { $corpus = $screened }
    Invoke-Python @("scripts/split_zh_corpus.py", "--corpus", $corpus)
}

if ($Stage -in @("all", "prepare")) {
    Step "4. acoustic features"
    foreach ($split in @("train", "validation", "test")) {
        $manifest = Join-Path $Data "$split.jsonl"
        if (Test-Path $manifest) {
            Invoke-Python @("-m", "tiny_voca.prepare", "--inputs", $manifest, "--split", $split,
                            "--output", $Features, "--config", $Config, "--resume", "--workers", "16")
        }
    }
}

if ($Stage -eq "align") {
    Step "5. alignment"
    Write-Host "The ctc aligner runs inside the annotation step; there is nothing to do here."
}

if ($Stage -in @("all", "annotate")) {
    Step "6. annotation manifests (ctc aligner)"
    Invoke-Python @("-m", "tiny_voca.zh_annotate", "--features", $Features,
                    "--output", $Annot, "--config", $Config, "--aligner", "ctc", "--device", $Device)
}

if ($Stage -in @("all", "train")) {
    Step "7. train"
    Invoke-Python @("-m", "tiny_voca.train",
                    "--train", (Join-Path $Annot "train.jsonl"),
                    "--validation", (Join-Path $Annot "validation.jsonl"),
                    "--output", $Run, "--config", $Config,
                    "--steps", "$Steps", "--gan-start", "$GanStart",
                    "--gan-ratio", "0.7", "--teacher-final", "0",
                    "--device", $Device)
}

if ($Stage -in @("all", "eval")) {
    Step "8. evaluate"
    $ckpt = Join-Path $Run "inference.pt"
    if (Test-Path $ckpt) {
        Invoke-Python @("-m", "tiny_voca.evaluate", "--checkpoint", $ckpt, "--annotations", $Annot,
                        "--output", "eval/zh", "--tag", "zh", "--device", $Device, "--asr-model", "small")
    }
}

if ($Stage -in @("all", "speak")) {
    Step "9. demo"
    $ckpt = Join-Path $Run "inference.pt"
    if (Test-Path $ckpt) {
        Invoke-Python @("-m", "tiny_voca.infer", "--checkpoint", $ckpt, "--voice", "ZH_F1",
                        "--text", "你好，这是一个很小的中文语音模型。它只有四百多万个参数。",
                        "--output", "demo-zh.wav", "--device", "cpu")
        Write-Host "wrote demo-zh.wav"
    }
}

Write-Host "`ndone. checkpoint: $Run/inference.pt"
