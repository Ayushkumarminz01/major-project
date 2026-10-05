# DCAE-XGB Hybrid IDS — CIC IoT 2023 (local version)

Trains a Denoising Convolutional Autoencoder (DCAE) unsupervised on benign traffic
(matching the EADL-IDS base-paper methodology: Gaussian-noise input corruption, MSE
reconstruction loss), then uses its latent embedding + reconstruction error as extra
features for a supervised XGBoost multi-class classifier trained on the real CIC-IoT-2023
labels. See `src/train.py` for the full pipeline and comments.

## 1. Install

Python 3.10+ recommended. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The TensorFlow DCAE uses CPU on native Windows for TensorFlow 2.11 and newer;
use WSL2 with a compatible CUDA setup to train the DCAE on an NVIDIA GPU. The
XGBoost classifier can use CUDA separately with `--xgb_device cuda`.

## 2. Get the data

Download the CIC IoT 2023 CSVs (e.g. from
https://www.unb.ca/cic/datasets/iotdataset-2023.html) and put the `.csv` files
anywhere on disk — by default the script looks in `./data`, but you can point it
anywhere with `--data_dir`. Subfolders are searched recursively.

## 3. Run

```bash
python src/train.py --data_dir ./data --output_dir ./outputs
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `./data` | Folder to search recursively for `*.csv` |
| `--output_dir` | `./outputs` | Where models, plots, and reports are saved |
| `--max_files` | `10` | Cap on number of CSV shards processed (use `-1` for all) |
| `--sample_frac` | `1.0` | Randomly keep this fraction of rows from each CSV, e.g. `0.2` = 20%. With rare-class preservation, only DDoS/DoS are sampled |
| `--granularity` | `8class` | `8class` (grouped) or `raw` (all ~34 labels) |
| `--epochs` | `15` | Maximum DCAE epochs; early stopping may finish sooner |
| `--batch_size` | `2048` | DCAE batch size |
| `--max_dcae_train_rows` | `1000000` | Maximum benign training rows sampled for DCAE fitting; `-1` uses all |
| `--dcae_val_rows` | `200000` | Maximum benign validation rows evaluated per DCAE epoch |
| `--data_workers` | `4` | Parallel TensorFlow input-map calls for streaming memmap batches |
| `--chunk_size` | `100000` | Rows processed per CSV/scaling/feature-extraction batch |
| `--max_train_rows` | `500000` | Maximum stratified rows kept in RAM for XGBoost training; `-1` uses the full train split |
| `--max_val_rows` | `100000` | Maximum stratified rows kept in RAM for XGBoost validation; `-1` uses the full validation split |
| `--roc_sample_rows` | `200000` | Maximum stratified test rows used for ROC-AUC and ROC plots |
| `--xgb_device` | `cuda` | XGBoost device; use `cpu` if CUDA is unavailable |
| `--run_raw_ablation` | off | Also train the slower raw-feature-only XGBoost comparison |
| `--seed` | `42` | Random seed |

The DCAE uses a 64-dimensional latent layer. XGBoost uses `multi:softprob`,
histogram trees, depth 8, learning rate 0.05, `max_delta_step=1`, subsample and
column sample rates of 0.8, and up to 1,000 trees with 30-round early stopping.
Training rows remain capped by `--max_train_rows` by default to
limit classifier memory use; increase that cap only when the available RAM/VRAM
can support it. For a CPU-only run, pass `--xgb_device cpu`.

DCAE training uses the same noisy-input/clean-target MSE objective for training
and validation. Its validation set is limited to 200,000 benign rows; early
stopping monitors validation MSE with patience 3 and restores the best weights.
The scaler is fitted incrementally on training rows only and the same fitted
instance transforms train, validation, and test rows.

Example, quick smoke test on a handful of files:

```bash
python src/train.py --data_dir ./data --max_files 3 --epochs 3
```

### Memory and disk use

CSV input is read in chunks. Cleaned features, scaled features, fused features,
split indices, and held-out probabilities are stored as NumPy memmaps under
`outputs/memmap_cache`, so the complete dataset is not copied into RAM. Ensure
the output drive has enough free space for several float32 copies of the selected
features (roughly 30 GB for the full 46.7-million-row dataset). XGBoost trains on
stratified samples bounded by `--max_train_rows` and `--max_val_rows`; inference
and classification metrics cover the held-out split in chunks, while ROC-AUC and
ROC plots use a stratified sample bounded by `--roc_sample_rows`.

To reduce processing time and disk use, set `--max_files` or `--sample_frac`,
for example `--max_files 30 --sample_frac 0.25`.

### Class imbalance

By default the classifier uses balanced per-row sample weights, avoiding
in-memory SMOTE expansion. XGBoost's `scale_pos_weight` is binary-only and does
not correctly express weights for this `multi:softprob` classifier. Validation
probabilities tune class thresholds from precision-recall candidates, then
refine them to improve multiclass validation Macro F1. Final predictions select
the class with the largest probability-minus-threshold score; thresholds are
saved in `preproc.joblib`.

```bash
python src/train.py --data_dir ./data --output_dir ./outputs
```

Use `--max_train_rows` to cap classifier RAM use, and add `--run_raw_ablation`
only when the additional raw-only model comparison is needed.

## 4. Outputs

Written to `--output_dir`:
- `dcae.keras`, `dcae_encoder.keras` — the trained autoencoder and its encoder half
- `xgb_classifier.json` — the trained supervised classifier
- `preproc.joblib` — scaler, class list, feature columns (needed to reuse the models)
- `dcae_training_loss.png`, `confusion_matrix.png`, `evaluation_metrics.png`,
  `overall_metrics.png`, `roc_auc.png`
- `evaluation_metrics.csv` — per-class, macro-average, and weighted-average
  precision, accuracy, recall, F1, ROC-AUC, and support
- `classification_report.txt` — per-class classification results and
  macro/weighted metrics; the raw-features-versus-DCAE-fused ablation is optional

## Folder layout

```
ciciot2023_dcae_hybrid/
├── README.md
├── requirements.txt
├── data/            # put your CIC-IoT-2023 CSVs here (or point --data_dir elsewhere)
├── outputs/         # results land here
└── src/
    ├── config.py    # tunable defaults, the 8-class label grouping
    ├── data.py      # loading, cleaning, label grouping, scaling, image reshaping
    ├── dcae.py      # the Denoising Convolutional Autoencoder architecture
    └── train.py     # main entry point — run this
```
