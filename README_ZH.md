# tiny-voca 中文版（Mandarin fork）

一个 4.5M 参数的**双音色中文 TTS**，从零训练，CPU 上也能实时合成。
这是 [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca) 的中文移植：模型结构、损失函数、两阶段训练与 GAN 梯度定标一行未改，改动集中在文本前端与强制对齐层。

- 端到端可跑：合成语料 → 特征 → 对齐 → 标注 → 训练 → 评测 → 推理，一条命令。
- 无需外部语料：`data/make_zh_corpus.py` 生成 2 万句、约 22 小时的中文文本。
- 语料由 Qwen3-TTS 合成，不涉及任何真人录音。

---

## 1. 在 5090 上跑之前必须知道的四件事

**⓪ Windows 原生可以直接跑。** 用 `run_all.ps1`，对齐默认走内置的**字符级 CTC 对齐器**（`--aligner ctc`，纯 PyTorch），不需要 MFA，也不需要 WSL 或 Docker。原来的 `fcntl` 文件锁已换成跨平台实现（Windows 用 `msvcrt.locking`）。

**① RTX 50 系是 Blackwell（sm_120），必须用 CUDA 12.8 的 PyTorch。**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # torch >= 2.7
```
用 cu121 / cu124 的轮子会报 `no kernel image is available for execution on the device`，而且只在第一个 kernel 启动时才报，看起来像别的问题。`scripts/doctor.py` 会提前检查这一项。

**② 不要给 Qwen3-TTS 开 FlashAttention-2。** 官方示例里的 `attn_implementation="flash_attention_2"` 在 Blackwell 上经常没有可用轮子。本仓库的合成本脚本没有开它，直接用默认注意力即可，1.7B 模型 bf16 只占约 4 GB。

**③ 合成和训练用两个 conda 环境。** `qwen-tts` 会拉自己的 torch 版本，和训练环境混在一起很容易把 CUDA 版本搞坏。

---

## 2. 目录

```
tiny-voca/
├── run_all.sh                    # 一键流水线
├── config/zh.json                # 中文模型配置（vocab 218, ZH_F1/ZH_M1）
├── data/
│   ├── make_zh_corpus.py         # 文本语料生成器
│   └── zh_sentences.txt          # 20000 句，约 22 小时
├── requirements/
│   ├── zh-synth.txt              # 合成环境
│   └── zh-train.txt              # 训练环境
├── scripts/
│   ├── doctor.py                 # 环境自检
│   ├── make_zh_dataset.py        # Qwen3-TTS 双音色合成 + 语料质检
│   └── split_zh_corpus.py        # 按文本切分 train/val/test（无泄漏）
└── src/tiny_voca/
    ├── zh_text.py                # 汉字 → 声母/韵母(带调)
    ├── text.py                   # 语言选择器（TINY_VOCA_LANG）
    ├── mfa_align.py              # TextGrid → 声学帧时长
    ├── zh_annotate.py            # 中文标注入口
    ├── train.py / model.py / losses.py   # 未改动
    └── infer.py / evaluate.py    # 已适配中文
```

---

## 3. 环境

### A. 合成（GPU）

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements/zh-synth.txt

# 国内用 ModelScope 下模型，避免 HuggingFace 超时
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local_dir ./Base
```

### B. 对齐

**Windows 用户跳过这一节。** 默认的 `ctc` 对齐器只需要 `torch` + `transformers` + `torchaudio`，`pip install -e .` 之后就能用；第一次运行会下载约 1.2 GB 的 wav2vec2 中文模型。

MFA 只在 Linux/macOS 上需要（字符边界更准），且没有 Windows 版本：

```bash
conda create -n mfa python=3.11 -y
conda activate mfa
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_mfa
```

### C. 训练（GPU）

```bash
conda create -n tinyvoca python=3.11 -y
conda activate tinyvoca
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/zh-train.txt
pip install -e .
```

---

## 4. 跑起来

### Windows（PowerShell，原生）

```powershell
conda activate tinyvoca
cd tiny-voca
.\run_all.ps1 smoke     # 20 句 / 2000 步，十几分钟
.\run_all.ps1           # 完整流程
```

只有 `corpus / synth / split / prepare / annotate / train / eval / speak` 这些阶段，**没有单独的 align 阶段**——ctc 对齐在 annotate 内部完成。`.\run_all.ps1 align` 会提示无需操作。

可选参数：`-Limit 20`、`-Steps 2000`、`-Device cuda`。

如果 PowerShell 拦截脚本执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS

先跑 smoke：20 句、2000 步，验证整条链路通不通，大约十几分钟。

```bash
conda activate tinyvoca
export TINY_VOCA_LANG=zh
bash run_all.sh smoke
```

通过后再跑完整流程（约 22 小时语料，合成 + 训练合计大半天到一天）：

```bash
bash run_all.sh
```

也可以只跑某一段，每段都可重复执行（已完成的会跳过）：

```bash
bash run_all.sh synth      # 只合成
bash run_all.sh prepare    # 只提特征
bash run_all.sh align      # 只做 MFA 对齐（ctc 无需此步）
bash run_all.sh annotate   # 只写标注（含 ctc 对齐）
bash run_all.sh train      # 只训练
```

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIMIT` | 0 | >0 时每音色只合成 N 句 |
| `STEPS` | 104000 | 训练步数 |
| `JOBS` | 8 | MFA 并行进程数 |
| `ALIGNER` | ctc | `ctc`（跨平台）或 `mfa`（需 MFA 二进制，仅 Linux/macOS） |
| `MFA_DICT` / `MFA_ACOUSTIC` | `mandarin_mfa.dict` / `.zip` | MFA 模型路径 |
| `CUDA_VISIBLE_DEVICES` | — | 单卡跑时设 `0` |

### 分步命令（等价于 run_all.sh 内部）

```bash
export TINY_VOCA_LANG=zh PYTHONPATH=src

python scripts/doctor.py --stage all                       # 0 环境自检
python data/make_zh_corpus.py --count 20000 --output data/zh_sentences.txt
python scripts/make_zh_dataset.py --texts data/zh_sentences.txt --output data/zh-corpus --templates 400
python scripts/split_zh_corpus.py --corpus data/zh-corpus/corpus.jsonl

for s in train validation test; do
  python -m tiny_voca.prepare --inputs data/zh-corpus/$s.jsonl --split $s --output features --config config/zh.json
done

python -m tiny_voca.zh_annotate --features features --export-corpus mfa-corpus --config config/zh.json

# 仅 Linux/macOS：MFA 对齐（Windows 跳过，直接用下面的 ctc）
for s in train validation test; do
  mfa align mfa-corpus/$s mandarin_mfa.dict mandarin_mfa.zip mfa-out/$s --single_speaker --jobs 8 --output_format long_textgrid --clean
done

# 对齐并写标注：ctc 跨平台，mfa 需 --textgrids
python -m tiny_voca.zh_annotate --features features --output annotations --config config/zh.json --aligner ctc
# python -m tiny_voca.zh_annotate --features features --textgrids mfa-out --output annotations --config config/zh.json --aligner mfa

python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/zh --config config/zh.json --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0

python -m tiny_voca.evaluate --checkpoint runs/zh/inference.pt --annotations annotations --output eval/zh --tag zh
python -m tiny_voca.infer --checkpoint runs/zh/inference.pt --voice ZH_F1 --text "你好，世界。" --output hello.wav
```

---

## 5. 各阶段产物与耗时（单卡 5090）

| 阶段 | 产物 | 估计耗时 | 占用 |
|---|---|---|---|
| 合成 20000 句 × 2 音色 | `data/zh-corpus/wav/*.wav` + `corpus.jsonl` | 2–5 h | 3.8 GB |
| 特征提取 | `features/{train,val,test}.jsonl` + `.pt` | 1–2 h | ~10 GB |
| 对齐 | ctc：在 annotate 内完成；MFA：`mfa-out/<split>/<id>.TextGrid` | ctc 10–30 min（GPU），MFA 1–3 h（CPU） | — |
| 标注 | `annotations/*.jsonl` | 数分钟 | 小 |
| 训练 | `runs/zh/inference.pt` | 8–12 h | 小 |
| 评测 | `eval/zh/report.json` + 音频 | 20 min | 小 |

MFA 是 CPU 密集的，`--jobs` 按物理核数给；它和 GPU 训练可以并行跑。

---

## 6. 质量检查点

合成后建议先跑质检，丢掉读错的片段：

```bash
python scripts/make_zh_dataset.py --output data/zh-corpus --stage screen --cer-threshold 0.15
```

它会用 Whisper 转录并按**字错误率（CER）**打分，输出 `corpus.screened.jsonl`。中文没有词边界，用 CER 不用 WER；评测模块同理。

训练中看 `runs/zh/metrics.jsonl`：`validation_predicted_mel` 应该单调下降，`pitch_error_semitones` 和 `voicing_accuracy` 是韵律是否学到的信号。

---

## 7. 已知限制

- **声调编码**：调值并入韵母 token（`a1`…`a5`），不单独预测。这是刻意的——声调是词汇信息，文本给定即可。
- **变调只在标点内生效**：三声连读按整句扫描，跨词时可能过度变调。要更准需接分词（jieba）把作用域限制在词内。
- **儿化未合并**：`儿` 是独立的 `er5` token。
- **拉丁字母被丢弃**：语料里混英文会漏读，合成前应过滤。
- **多音字**依赖 pypinyin 的短语词典，生僻专名仍会错。
- **ctc 对齐的声韵母切分是近似的**：字符级时间戳是准的，但一个字内部声母/韵母的边界靠"清浊翻转"或按类别给固定比例估计。MFA 能直接给出声韵母边界，质量更好；两者训练出来的模型都自洽，因为标注与训练用的是同一套约定。
- **MFA 对合成高音女声**不一定最优。若大量样本被判 `missing_textgrid` 或 `implausible_rate`，先抽查几个 TextGrid，再调 `--min-pause-frames`。
- **音质上限 = Qwen3-TTS 的上限**。想更自然，应换真人语料重训，而不是放大模型。

---

## 8. 许可

代码与权重沿用 Apache-2.0。语料由 Qwen3-TTS 合成（Apache-2.0），不包含真人录音。
