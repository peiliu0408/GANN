# GANN — Global-Attention Neural Captioner

Image captioning model: Transformer encoder-decoder with a **global attention**
mechanism in every decoder layer, a caption-object multi-label head, and
two-stage training:

- **Stage 1 (XE)**: joint label + cross-entropy loss, noam schedule
- **Stage 2 (SCST)**: self-critical sequence training on CIDEr-D reward with
  multi-sample variance reduction (`train_sample_n = 5`)

Trained on MS-COCO (Karpathy split), single GPU.

---

## 1. Quick start

```bash
# 0) download the data archive (too large for git) and unpack it HERE:
#      data.tar          (~85 GiB)  -> data/
#    link: see §6
tar -xf data.tar

# 1) install
pip install -r requirements.txt
#    NOTE: METEOR / SPICE metrics need a Java runtime (JRE 8+):
#      apt install default-jre     (or: conda install openjdk)
#    Without Java they are skipped automatically; BLEU/ROUGE-L/CIDEr-D work.

# 2) train (two stages, best recipe, ~4 h on one A800)
CUDA_VISIBLE_DEVICES=0 bash train.sh

# 3) test  (bundled checkpoint dir, test split, greedy decoding)
bash test.sh ckpt test                          # final test numbers
```

The `ckpt/` folder in this repo contains the best model's `opt.json` plus
ready-made prediction files — but **not** the weights (`model-best.pth`,
286 MB, too large for git). Download `model-best.pth` from the shared folder
in §6 and put it into `ckpt/`:

```
ckpt/
├── model-best.pth    <- downloaded from §6
└── opt.json
```

## 2. Environment

```bash
pip install -r requirements.txt        # torch >= 1.10 (CUDA build), numpy, tqdm
```

| Item | Requirement |
|---|---|
| GPU | 1× ≥ 40 GB memory (80 GB recommended — SCST decodes 5 samples/step) |
| Python | 3.8+ |
| **Java** | **needed only for METEOR / SPICE metrics** (`apt install default-jre` or `conda install openjdk`); `java -version` must work. All other metrics are pure Python. |

## 3. Data layout (`data.tar`)

Unpack `data.tar` at the repo root — it creates exactly this tree:

```
data/
├── captions/
│   └── dataset_coco.json        # Karpathy split (images + sentences + splits)
├── features/
│   ├── cocobu_att/              # one <cocoid>.npz per image, key 'feat',
│   │   ...                      # shape (n, 2048), n = 10..100 regions
│   └── cocobu_att_packed/
│       ├── data.f32             # all features packed into one float32 memmap
│       └── packed_meta.json
└── processed/
    ├── vocab.json               # word vocabulary (threshold 5)
    ├── feat_index_packed.json   # cocoid -> [row_start, row_end]
    ├── labels.json              # 207 caption-object labels
    ├── labels_{train,val,test}.npz
    ├── coco-train-idxs.pkl      # CIDEr-D df cache for the SCST reward
    └── ...
```

If you only have raw per-image npz files and no packed file, rebuild it:

```bash
python3 scripts/build_vocab_labels.py
python3 scripts/build_feature_index.py
python3 scripts/pack_features.py
python3 scripts/prepro_ngram_cache.py
```

## 4. Training

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

Two stages run back-to-back; outputs go to `log_xe_<arch>_seed<N>/` and
`log_scst_<arch>_seed<N>/`. Every hyperparameter is env-overridable:

```bash
ARCH=paper SEED=44 bash train.sh                                   # new arch/seed
RESUME_XE=ckpt/model-best.pth bash train.sh                        # skip stage 1
```

Best recipe (found by systematic search):

| Stage | Key settings |
|---|---|
| 1 — XE | `bert` init, label smoothing 0.1, weight tying, EMA 0.999, noam warmup 20000, λ = 0.2, 30-epoch cap, early stop 5 |
| 2 — SCST | fixed lr 2.5e-5 (noam disabled), `train_sample_n` = 5, 25-epoch cap, early stop 8 |

Timing reference (one A800, medium arch): XE ≈ 2 h (24-30 ep), SCST ≈ 1.5 min/ep
× epochs.

## 5. Testing & results

```bash
bash test.sh ckpt test                          # test split, greedy (bundled ckpt)
bash test.sh <ckpt_dir> test beam 5 1.5         # custom beam / length penalty
```

Metrics: BLEU-1..4, ROUGE-L, CIDEr-D (+ METEOR/SPICE if Java is present).
Predictions are saved to `<ckpt_dir>/preds_<split>_<method>.json`.

**Rescore an existing predictions JSON** — if you already have a predictions
file (gts + preds), you can recompute all metrics directly from it, without
loading the checkpoint or a GPU (runs in seconds):

```bash
python3 scripts/rescore.py ckpt/preds_test_greedy.json
python3 scripts/rescore.py ckpt/preds_test_scst.json
# optionally save the metrics to a file:
python3 scripts/rescore.py <preds.json> --out score.json
```

The script checks that gts/preds id sets match, prints the recomputed metrics
next to any metrics recorded in the file's `info`, and (with `--out`) writes
them to a standalone JSON.

**Decoding note**: this model is trained with SCST, whose reward is the
CIDEr of *greedy*-decoded captions — so **greedy decoding scores highest on
CIDEr-D**; beam search mainly helps BLEU-1.

Reference results (MS-COCO Karpathy test, this exact recipe, seed 43):

| Predictions file | Decoding | B-1 | B-4 | ROUGE-L | CIDEr-D |
|---|---|---|---|---|---|
| `ckpt/preds_test_greedy.json` | greedy | 80.02 | 38.56 | 58.50 | 123.97 |
| `ckpt/preds_test_scst.json` | greedy | 80.61 | 40.05 | 59.24 | **129.52** |

## 6. Large-file downloads

`data.tar` and the model weights (`model-best.pth`) are too large for git —
download both from the shared OneDrive folder:

**[GANN — OneDrive shared folder](https://1drv.ms/f/c/94a9f528230586fe/IgA0wgMW6I_FTZL2rxXbhsrYAaq9-gD2C-cIE1k5crT6lqA?e=fo7OUY)**

| File | Size | Where to put it |
|---|---|---|
| `data.tar` | 85.2 GB | repo root, then `tar -xf data.tar` (creates `data/`, see §3) |
| `model-best.pth` | 286 MB | `ckpt/model-best.pth` |

## 7. Repository layout

```
├── README.md                 this file
├── train.sh                  one-command two-stage training
├── test.sh                   one-command evaluation
├── requirements.txt          python deps (+ Java note for METEOR/SPICE)
├── .gitignore                excludes data/, weights, logs
├── scripts/                  train.py, eval.py, rescore.py, data-prep scripts
├── src/gann/                 model, dataloader, SCST reward, metric wrapper
├── third_party/              vendored BLEU/ROUGE/CIDEr code (pycocoevalcap)
├── ckpt/                     best model: opt.json + predictions
│                             (model-best.pth via §6 download)
├── data/                     (from data.tar, see §3)
└── log_*/                    training outputs (created by train.sh, gitignored)
```
