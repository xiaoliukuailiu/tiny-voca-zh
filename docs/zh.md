# Mandarin (中文) 版本改造说明

把 tiny-voca 从美式英语移植到普通话。模型结构、损失函数、两阶段训练与 GAN 梯度定标**一行未改**；改动集中在文本前端和对齐层。

## 1. 为什么不能直接复用英文管线

英文用 `facebook/wav2vec2-lv-60-espeak-cv-ft` 做 CTC 强制对齐。它的 388 个标签里确实有 84 个带声调的中文标签，但实际覆盖是残缺的：

- **三声 0 个标签**（普通话里最常见的一类调）
- **所有 `-n` / `-ng` 鼻韵母缺失**（`an`/`ən`/`aŋ`/`əŋ`/`in`/`iŋ`/`uan`/`yn` 全部没有），只有 `onɡ` 且只覆盖二声和五声
- 只有 30 个韵母基式，覆盖不了普通话音节

所以对齐器必须换掉。现在有两条路：

- **`--aligner ctc`（默认，跨平台，Windows 可用）**：用中文 wav2vec2 做字符级 CTC 对齐，一个汉字就是一个音节，再把字内时间按"清浊翻转/声母类别比例"切给声母和韵母。纯 PyTorch，无外部二进制。
- **`--aligner mfa`（Linux/macOS）**：Montreal Forced Aligner + `mandarin_mfa`，直接给出声韵母边界，质量更好。

两者产出的标注都自洽——训练和推理用的是同一套前端约定。运行方式见仓库根目录 `README_ZH.md`。

## 2. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `src/tiny_voca/zh_text.py` | **新增**。汉字→声母/韵母(带调)音素化；数字、日期、百分比、金额规范化；三声变调、一/不变调；轻声 |
| `src/tiny_voca/text.py` | 改为前端选择器，由 `TINY_VOCA_LANG` 在 `en_text` / `zh_text` 间切换 |
| `src/tiny_voca/en_text.py` | 原 `text.py` 重命名，逻辑未动 |
| `src/tiny_voca/config.py` | `vocab_size` 和 `speakers` 改为按语言取默认值；英文仍是 76 / `F2,M2`，中文 218 / `ZH_F1,ZH_M1` |
| `src/tiny_voca/mfa_align.py` | **新增**。TextGrid 解析 + 声素区间→声学帧时长，含停顿归属 |
| `src/tiny_voca/zh_annotate.py` | **新增**。中文标注入口，产出 `train.py` 能直接读的 manifest |
| `src/tiny_voca/__init__.py` | `TinyVoca` 改为懒加载，前端和配置可以在没装 torch 的环境里测试 |
| `scripts/make_zh_dataset.py` | **新增**。Qwen3-TTS 双音色合成 + Whisper CER 质检 |
| `tests/test_zh_frontend.py` `tests/test_zh_align.py` | **新增**，47 个用例 |

`model.py` / `train.py` / `losses.py` / `prepare.py` 未改。

## 3. 符号表设计

```
specials(5) + 标点(12) + 声母(21) + 韵母×5调(35×5=175) = 218
```

- **声调并入韵母**：`a1 a2 a3 a4 a5`、`ong2`、`er5`（5=轻声）。声调是词汇信息，由文本给定，不需要模型预测；同时保证"一个 token = 一个音素 = 一个对齐标签"。
- 声母独立成 token（21 个），韵母保持在 35 个，而不是几百个整音节。
- `ü` 在内部写作 `v`：`lv4`（绿）、`nv3`（女）、`jv4`（句，j/q/x 后的 u 归一为 v）。
- 标点折叠到 ASCII：`，`→`,`、`。`→`.`、`、`→`,`。

## 4. 环境

三个环境分开，避免依赖打架。

```bash
# A. 合成（GPU，CUDA）
conda create -n qwen3-tts python=3.12 -y && conda activate qwen3-tts
pip install -U qwen-tts soundfile scipy
pip install -U faster-whisper            # 只有质检需要
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local_dir ./Base

# B. 对齐
conda create -n mfa python=3.11 -y && conda activate mfa
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_mfa

# C. 训练
pip install -e ".[own]"    # 仓库根目录
pip install pypinyin
```

## 5. 全流程

```bash
export TINY_VOCA_LANG=zh

# 1) 合成语料：两个音色 × 全部句子，输出 24kHz wav + corpus.jsonl
python scripts/make_zh_dataset.py \
    --texts data/zh_sentences.txt --output data/zh-corpus \
    --templates 200 --batch 8

# 2) 质检（可选但强烈建议）：Whisper 转录，CER>15% 丢弃
python scripts/make_zh_dataset.py --output data/zh-corpus --stage screen --cer-threshold 0.15
# 之后用 data/zh-corpus/corpus.screened.jsonl 作为输入

# 3) 声学特征
python -m tiny_voca.prepare --inputs data/zh-corpus/corpus.screened.jsonl \
    --split train --output features
# validation / test 同理

# 4) 导出 MFA 语料并对齐
python -m tiny_voca.zh_annotate --features features --export-corpus mfa-corpus
for s in train validation test; do
  mfa align mfa-corpus/$s mandarin_mfa.dict mandarin_mfa.zip mfa-out/$s --single_speaker --jobs 8
done

# 5) 写标注 manifest
python -m tiny_voca.zh_annotate --features features --textgrids mfa-out --output annotations

# 6) 训练（超参与英文版一致）
python -m tiny_voca.train --train annotations/train.jsonl \
    --validation annotations/validation.jsonl --output runs/zh \
    --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0

# 7) 推理
python -m tiny_voca.infer --checkpoint runs/zh/inference.pt --voice ZH_F1 \
    --text "你好，这是一个很小的中文语音模型。" --output hello.wav
```

## 6. 规模预估

| 项 | 英文版 | 中文版 |
|---|---|---|
| 音素表 | 76 | 218 |
| 推理参数 | 4,537,291 | 约 4.56M（embedding 大 1.7 万） |
| 训练步数 | 104,737 | 建议持平或略多 |
| 单卡 4090 时长 | 7.7 h | 约 8–10 h（音素更多、句子更长） |

参数量仍在 `train.py` 的 5,000,000 预算内。

## 7. 已知限制

- **变调只在标点内生效**：没有做分词，三声连读跨词时可能过度变调。要更准需接 jieba 限制在词内。
- **儿化未合并**：`儿` 作为独立的 `er5` token，没有并进前一个韵母。
- **拉丁字母/英文单词被丢弃**，不计入 token。语料里混英文会漏字，合成阶段应先过滤。
- **多音字依赖 pypinyin 的短语词典**，生僻专名仍会读错。
- **MFA 的 `mandarin_mfa` 声学模型对合成音（尤其高音女声）不一定最优**，若大量对齐被拒（`implausible_rate` / 缺失 TextGrid），先抽查 TextGrid 再决定是否换 `mandarin_mfa` 以外的模型或调 `--min-pause-frames`。
- **合成数据的音质上限就是 Qwen3-TTS 的上限**：模型学的是"这个 TTS 怎么说话"，包括它的合成痕迹。
