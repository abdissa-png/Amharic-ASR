# Amharic ASR

Offline Amharic speech recognition.
The test speakers and the test prompts are held out of training. That split is
the primary benchmark.

Two acoustic models share the same data pipeline:

- A **Conformer CTC** model, aimed at roughly 20–50M parameters, with character,
  SentencePiece, or phoneme outputs.
- A **streaming Zipformer-S** transducer (RNN-T + CTC) vendored from icefall.
  The live chunk is 32 embed frames (640 ms) with 128 frames of left context.

Decoding is greedy CTC, a prefix beam, a word KenLM (pyctcdecode), or a
character / morpheme KenLM. The Conformer and the Zipformer both export to
ONNX. A small local demo serves a microphone and file upload.

The released graphs are
[abdissa-png/amharic-asr-small](https://huggingface.co/abdissa-png/amharic-asr-small).
[Amharic ASR](https://huggingface.co/spaces/abdissa-png/amharic-asr) runs them
in the browser.


Training text is normalised conservatively (NFC, whitespace, quotes; punctuation
becomes a space). Ge'ez homophones are **not** folded in training. Scoring uses
`evaluation/geez_eval.py`.

Checkpoints, audio, and generated jsonl stay out of git. See `.gitignore`.

## Layout

| Path | Role |
| --- | --- |
| `prepare_dataset.py`, `data/` | Dataset.ET plus FLEURS, ALFFA, BDU, and Leyu; pooling, augmentation, audit |
| `tokenizer/` | Character, SentencePiece, phoneme, and HornMorpho segmentation |
| `phonology/` | G2P, lexicon, morphological decomposition |
| `models/` | Log-Mel features, Conformer CTC, distillation, icefall Zipformer wrapper |
| `train.py` | Conformer CTC training |
| `train_zipformer.py` | Streaming Zipformer training (needs `k2` and `lhotse`) |
| `train_teacher.py` | Fine-tune the large CTC teacher |
| `decoding/` | Greedy, beam, word KenLM, character/morpheme KenLM, Zipformer streaming and ONNX |
| `evaluation/` | WER, CER, homophone fold, seen/unseen words, robustness, failure inspection |
| `evaluate.py` | Score a Conformer checkpoint or an ONNX graph |
| `eval_zipformer.py`, `eval_zipformer_stream.py`, `eval_leyu.py`, `eval_youtube_timed.py` | Zipformer and long-form scoring |
| `deployment/` | ONNX export, INT8 quantization, ONNX Runtime inference, RTF |
| `demo/` | Local mic / upload server (`python -m demo.app`) |
| `baselines/` | Pretrained Amharic and multilingual models on the same test set |
| `configs/` | One YAML per experiment |
| `third_party/icefall/` | Zipformer modules pinned in `third_party/icefall/SOURCE` |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conformer training pulls datasets through Hugging Face `datasets`. The hub
cache is reused. Set `HF_TOKEN` only if the hub asks you to log in.
`train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when that
variable is unset.

Zipformer training needs extra packages. `k2` wheels are tied to the Python
version, the torch build, and CUDA. If `import k2` fails, the vendored
`icefall.utils` prints the matching wheel URL.

```bash
pip install -r requirements-zipformer.txt
```

Pretrained baselines (not needed for training):

```bash
pip install -r requirements-baselines.txt
# Omnilingual ASR is separate: https://github.com/facebookresearch/omnilingual-asr
```

The word and character LMs need the KenLM Python package (in
`requirements.txt`) and the `lmplz` binary from
[kpu/kenlm](https://github.com/kpu/kenlm). Do not put that C++ checkout at the
repository root or at `third_party/kenlm/`; both paths shadow `import kenlm`
and are gitignored.

## Data

Audio under `data/` and jsonl under `metadata/` are produced locally and are
not in git.

```bash
python prepare_dataset.py                 # Dataset.ET FLAC + jsonl
python prepare_dataset.py --metadata-only # jsonl only
python -m data.audit
python -m data.oov

python -m data.prepare_fleurs --splits train validation test
python -m data.prepare_alffa
python -m data.prepare_bdu
python -m data.prepare_leyu
python -m data.prepare_aug_assets
python -m data.prepare_pool              # Dataset.ET dev/test stay the comparable splits
python -m data.prepare_pool_leyu         # screened pool train plus Leyu train
```

`data/screen_pairs.py` flags transcript pairs that should not be trained on.

## Conformer CTC

```bash
python -m models.conformer --config configs/20m_char.yaml
python train.py --config configs/20m_char.yaml --overfit-batch
python train.py --config configs/20m_char.yaml
python train.py --config configs/20m_char_4layer.yaml --resume
```

The default character model (`configs/20m_char.yaml`) is 12 layers, width 256,
4 heads, and an 80-bin log-Mel input. Other YAMLs cover 4-layer and 12-layer
variants, relative attention, multi-kernel convolution, a 20 ms front end,
augmentation, the pooled corpora, SentencePiece sizes, a phoneme output, and
the ablations named `configs/ablation_4layer_*.yaml`.

```bash
python evaluate.py --config configs/20m_char.yaml --ckpt exp/20m_char/best.pt --split test
python evaluate.py --config configs/20m_char_12layer.yaml \
    --ckpt exp/20m_char_12layer/best.pt \
    --jsonl metadata/fleurs_am_et/test.jsonl --decoder kenlm
python evaluate.py --config configs/20m_char_12layer.yaml \
    --onnx exp/20m_char_12layer/model.int8.onnx \
    --jsonl metadata/fleurs_am_et/test.jsonl --decoder greedy
```

`--decoder` is `greedy`, `beam`, or `kenlm`. Reports include WER, CER, MER,
seen/unseen-word error, short/long subsets, RTF, RAM, parameter count, and
checkpoint size.

```bash
python -m decoding.train_lm --out-dir exp/20m_char_12layer/lm
python -m decoding.train_lm --unit char --lm-only --order 6 \
    --src-text exp/20m_char_12layer/lm/train.txt \
    --out-dir exp/zipformer_s_streaming_pool/lm_gpac_char
```

Word n-grams are scored by pyctcdecode. Character and morpheme n-grams are
scored by `decoding/unit_lm.py`. Do not retune language-model weights on a test
split.

```bash
pip install -r requirements-baselines.txt
python -m baselines.run --model ethio-asr --split test
python -m baselines.run --model dvoice --split test
python -m baselines.run --model omnilingual --split test
python -m baselines.run --model whisper --split test
```

```bash
python train_teacher.py
```

`models/distillation.py` mixes teacher and ground-truth CTC targets.

## Streaming Zipformer

`third_party/icefall/SOURCE` records the icefall commit these files came from.

```bash
python train_zipformer.py --config configs/zipformer_s_streaming_et.yaml
python train_zipformer.py --config configs/zipformer_s_streaming_pool.yaml
python train_zipformer.py \
    --config configs/zipformer_s_streaming_pool_leyu.yaml \
    --resume exp/zipformer_s_streaming_pool/best.pt
```

The Leyu continuation keeps Dataset.ET as the dev set. Score Leyu itself with
`eval_leyu.py`. `save_live_zipformer.py` copies a running training process to
`emergency.pt` without allocating another copy of the weights on the GPU.

```bash
python eval_zipformer.py \
    --config configs/zipformer_s_streaming_pool.yaml \
    --ckpt exp/zipformer_s_streaming_pool/best.pt \
    --jsonl metadata/fleurs_am_et/test.jsonl
python eval_zipformer_stream.py \
    --config configs/zipformer_s_streaming_pool.yaml \
    --ckpt exp/zipformer_s_streaming_pool/best.pt \
    --jsonl metadata/fleurs_am_et/test.jsonl
python eval_leyu.py --test
python eval_youtube_timed.py --url 'https://www.youtube.com/watch?v=YiB33Piqh20'
```

`eval_zipformer.py` runs CTC, greedy transducer, word KenLM, and the character
or morpheme LM. `eval_zipformer_stream.py` pumps audio the way the microphone
demo does. Failure dumps and frame inspections live in
`evaluation/diagnose_failures.py`, `inspect_internals.py`, and
`inspect_pipeline.py`.

`bench_zipformer.py`, `bench_amp.py`, and `diag_fp16.py` check throughput and
mixed precision. They are not part of the training loop.

## ONNX and the demo

```bash
python -m deployment.export_onnx \
    --config configs/20m_char_12layer.yaml \
    --ckpt exp/20m_char_12layer/best.pt
python -m deployment.quantize --onnx exp/20m_char_12layer/model.onnx
python -m deployment.benchmark --onnx exp/20m_char_12layer/model.onnx --split test --limit 50

python -m deployment.export_zipformer_streaming
```

The Zipformer export writes encoder, decoder, and joiner graphs for chunk 32 /
left 128, plus an INT8 copy, under the experiment `onnx/` directory. ONNX
Runtime uses CUDA when the process can see a GPU, and CPU otherwise.
`onnxruntime` in `requirements.txt` is the CPU wheel. For a GPU, install
`onnxruntime-gpu` in place of it.

```bash
python -m demo.app
```

Open `http://127.0.0.1:7860`. The process also speaks HTTPS on that port so a
phone on the same network can use the microphone. The checkpoint menu lists
the PyTorch and ONNX models whose files are present. Weights are not in git, so
a fresh clone shows only the presets that exist on disk. The same graphs are
on [abdissa-png/amharic-asr-small](https://huggingface.co/abdissa-png/amharic-asr-small),
and the browser page is
[Amharic ASR](https://huggingface.co/spaces/abdissa-png/amharic-asr).

## Tests

```bash
pytest
```
