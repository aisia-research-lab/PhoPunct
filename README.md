# PhoPunct — Vietnamese Punctuation Restoration Benchmark

Code to reproduce the experiments of *PhoPunct: A Large-Scale Benchmark of Transformer Architectures for Vietnamese Punctuation Restoration* (PACLIC 2026).

We compare four multilingual / Vietnamese encoders (mBERT, vELECTRA, ViBERT, XLM-R)
with and without a Bi-LSTM and with a softmax or CRF output layer (16 baselines),
plus the proposed **PhoPunct** (PhoBERT-large + Bi-LSTM + CRF), on two domains
(**News**, **Novels**). All 17 configurations share one data pipeline, one model
class, and one training budget, so differences come from the architecture only.

## Repository layout

```
├── train.py                 # train one configuration (+ test-set evaluation of the best epoch)
├── evaluate.py              # score a checkpoint on valid/test
├── predict.py               # punctuate raw text with a checkpoint
├── phopunct/
│   ├── data.py              # word-level data pipeline (chunking, sub-word alignment)
│   ├── models.py            # WordTagger: backbone [+ Bi-LSTM] + linear [+ CRF]
│   └── metrics.py           # P/R/F1 over the 6 punctuation classes
├── scripts/
│   ├── run_all.sh           # all 17 configurations x 2 domains (paper setting)
│   ├── smoke_test.sh        # quick CPU sanity check
│   ├── collect_results.py   # build the results table from logs/ or outputs/
│   └── make_smoke_subset.py
├── data/README.md           # data format, statistics, checksums
└── logs/                    # original training logs of the 34 runs reported in the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Paper runs used a single NVIDIA RTX 3090 / 3090 Ti (24 GB) with fp16.

## Data

Put the corpus under `data/punctuation/{News,Novels}/{train,valid,test}.txt`
(one `<word> <label>` per line). See [`data/README.md`](data/README.md) for the
format, label statistics and SHA-256 checksums of the exact files used.

## Models

| # | Configuration | `train.py` flags |
|---|---|---|
| 1–4 | {mBERT, vELECTRA, ViBERT, XLM-R} | `--backbone {mbert,velectra,vibert,xlmr}` |
| 5–8 | … + Bi-LSTM | `… --use_bilstm` |
| 9–12 | … + CRF | `… --use_crf` |
| 13–16 | … + Bi-LSTM + CRF | `… --use_bilstm --use_crf` |
| 17 | **PhoPunct**: PhoBERT-large + Bi-LSTM + CRF | `--backbone phobert --use_bilstm --use_crf` |

| `--backbone` | Hugging Face checkpoint |
|---|---|
| `mbert` | `bert-base-multilingual-cased` |
| `velectra` | `FPTAI/velectra-base-discriminator-cased` |
| `vibert` | `NlpHUST/vibert4news-base-cased` |
| `xlmr` | `xlm-roberta-base` |
| `phobert` | `vinai/phobert-large` |

Architecture (identical for all configurations):

```
words -> sub-word tokenizer -> encoder
      -> first sub-word of each word (gathered BEFORE the Bi-LSTM, so the
         Bi-LSTM / CRF always operate on the word sequence)
      -> [Bi-LSTM, 128 hidden/direction, 1 layer] -> dropout 0.2 -> linear (7 labels)
      -> CRF (Viterbi)  |  softmax (masked cross-entropy)
```

Task: word-level tagging with 7 labels `O, PERIOD, COMMA, COLON, QMARK, EXCLAM, SEMICOLON`.
Input streams are cut into chunks of ≤128 words, only after a sentence-final label.

## Training configuration (all runs)

| Hyper-parameter | Value |
|---|---|
| max sub-word length | 256 (hard limit for PhoBERT) |
| batch size (train / eval) | 16 / 32 |
| epochs | 8 |
| optimizer | AdamW (PyTorch defaults: weight decay 0.01), lr 2e-5 for all parameters |
| schedule | linear, 6% warm-up |
| gradient clipping | 1.0 |
| precision | fp16 autocast (logits cast to fp32 before the CRF) |
| seed | 42 |
| model selection | best validation macro F1 over the 8 epochs |

These are the defaults of `train.py`.

## Reproducing the paper

**1. Smoke test (CPU, a few minutes)**

```bash
bash scripts/smoke_test.sh
```

**2. All 34 runs (GPU)**

```bash
bash scripts/run_all.sh            # or a subset: bash scripts/run_all.sh 12 16 17
python scripts/collect_results.py --outputs outputs
```

Each run writes to `outputs/<run>_<domain>/`: `best_checkpoint.pt`,
`valid_metrics.json`, `test_metrics.json` (best-validation epoch scored on
`test.txt`), `eval_results.txt`, `args.json`.
Approximate time per run on an RTX 3090: Novels ≤ 1 h; News 1.5–9 h
(PhoPunct is the slowest, ~1 h per epoch on News).

**3. A single run**

```bash
python train.py --backbone phobert --use_bilstm --use_crf \
    --data_dir data/punctuation/News --output_dir outputs/phopunct_news --device cuda --fp16
```

**4. Evaluate an existing checkpoint / punctuate text**

```bash
python evaluate.py --checkpoint outputs/phopunct_news/best_checkpoint.pt \
    --data_dir data/punctuation/News --split test
python predict.py --checkpoint outputs/phopunct_news/best_checkpoint.pt \
    --text "bạn có khỏe không tôi rất nhớ bạn"
# e.g. -> Bạn có khỏe không? Tôi rất nhớ bạn!
```

`evaluate.py` / `predict.py` also accept checkpoints produced by the original
training scripts (configuration is inferred from the weights).

## Results

Macro F1 over the six punctuation classes (`O` excluded), **validation set**,
best epoch, parsed from [`logs/`](logs) with
`python scripts/collect_results.py --logs logs`:

| # | Model | News | Novels |
|---|---|---|---|
| 1 | mBERT | 0.5371 | 0.4384 |
| 2 | vELECTRA | 0.5380 | 0.4070 |
| 3 | ViBERT | 0.5747 | 0.4982 |
| 4 | XLM-R | 0.5825 | 0.4979 |
| 5 | mBERT + Bi-LSTM | 0.5390 | 0.4256 |
| 6 | vELECTRA + Bi-LSTM | 0.5342 | 0.4046 |
| 7 | ViBERT + Bi-LSTM | 0.5774 | 0.4587 |
| 8 | XLM-R + Bi-LSTM | 0.5807 | 0.4683 |
| 9 | mBERT + CRF | 0.5376 | 0.4339 |
| 10 | vELECTRA + CRF | 0.5384 | 0.4064 |
| 11 | ViBERT + CRF | 0.5797 | 0.4978 |
| 12 | XLM-R + CRF | 0.5862 | 0.5018 |
| 13 | mBERT + Bi-LSTM + CRF | 0.5354 | 0.4237 |
| 14 | vELECTRA + Bi-LSTM + CRF | 0.5345 | 0.4022 |
| 15 | ViBERT + Bi-LSTM + CRF | 0.5766 | 0.4591 |
| 16 | XLM-R + Bi-LSTM + CRF | 0.5789 | 0.4690 |
| 17 | **PhoPunct** | **0.6135** | 0.4897 |

<!-- TODO: add the test-set table produced by `collect_results.py --outputs` / evaluate.py. -->

Per-class reports for every epoch are in the logs. The log of PhoPunct-News
ends during the epoch-8 evaluation (the instance stopped); its best epoch (7)
is complete.

## Reproducibility notes

- The code was refactored from per-family scripts into one package for release.
  With the same seed, `train.py` reproduces the original scripts **bit-for-bit**
  (all per-step losses and validation scores were compared for the softmax,
  CRF and PhoPunct families on a CPU subset).
- GPU runs with fp16 and cuDNN LSTM kernels are not fully deterministic;
  re-runs may differ slightly from the logged scores.
- COLON / SEMICOLON are very rare in Novels (6 SEMICOLON tokens in valid),
  so per-class and macro scores on Novels are noisy.
- `NlpHUST/vibert4news-base-cased` lacks `model_type`/`tokenizer_class`
  metadata, so mBERT and ViBERT are loaded with `BertModel`/`BertTokenizer`.

## Citation

If you use the data, code, or other resources provided in this repository in your research or work, please cite the following paper:

```bibtex
@inproceedings{nguyen2026phopunct,
  title     = {{PhoPunct: A Large-Scale Benchmark of Transformer Architectures for Vietnamese Punctuation Restoration}},
  author    = {Nguyen, Khoa D. and Huynh, Thai Bao and Nguyen, Binh T.},
  booktitle = {Proceedings of the 40th Pacific Asia Conference on Language, Information and Computation (PACLIC 40)},
  year      = {2026}
}
```
We appreciate proper attribution when using or building upon the resources released with this repository.

## License

This project is licensed under the terms of the [MIT License](LICENSE). See the `LICENSE` file for more details.
