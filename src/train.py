"""DCAE-XGB Hybrid IDS on CIC IoT 2023 — local runner.

Stage 1 (unchanged base-paper methodology): a Denoising Convolutional Autoencoder
is trained unsupervised on BENIGN traffic only.

Stage 2 (new): the trained encoder is used as a feature extractor over every row
(benign + attack); its latent embedding + reconstruction error are fused with the
raw engineered features.

Stage 3 (new): a supervised XGBoost multi-class classifier is trained on the fused
features using the real CIC-IoT-2023 labels, replacing the base paper's label-free
K-means + LLM-RAG explanation stage (which exists only because that paper assumed
no ground truth was available).

Run:  python train.py --data_dir ../data --output_dir ../outputs
"""
import argparse
import os
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe; plots are still saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              precision_recall_curve, recall_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import callbacks, optimizers
from xgboost import XGBClassifier

import data as D
from config import BENIGN_LABEL, LATENT_DIM, NOISE_STD, SEED, USE_LOG1P
from dcae import build_dcae


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="../data", help="Folder searched recursively for *.csv")
    p.add_argument("--output_dir", default="../outputs", help="Where models/plots/reports are saved")
    p.add_argument("--max_files", type=int, default=10, help="-1 to load every CSV found")
    p.add_argument("--sample_frac", type=float, default=1.0,
                    help="Randomly keep this fraction of rows from each CSV (e.g. 0.2 = 20%%). "
                         "Lower this if you run out of RAM instead of lowering --max_files. "
                         "By default only applies to the huge DDoS/DoS rows (see "
                         "--no_preserve_rare) so it doesn't shrink already-rare classes further.")
    p.add_argument("--no_preserve_rare", action="store_true",
                    help="Apply --sample_frac uniformly to every class, including rare ones "
                         "(old behavior). Default is to always keep every BruteForce/Web/"
                         "Recon/Spoofing/Mirai/Benign row and only subsample DDoS/DoS.")
    p.add_argument("--granularity", choices=["8class", "raw"], default="8class")
    p.add_argument("--epochs", type=int, default=15, help="Maximum DCAE training epochs")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_dcae_train_rows", type=int, default=1_000_000,
                    help="Maximum benign training rows sampled for DCAE fitting; -1 uses all")
    p.add_argument("--dcae_val_rows", type=int, default=200_000,
                    help="Maximum benign validation rows evaluated per DCAE epoch")
    p.add_argument("--data_workers", type=int, default=4,
                    help="Parallel TensorFlow input-map calls for memmap batch loading")
    p.add_argument("--chunk_size", type=int, default=100_000,
                    help="Rows processed at a time while reading, scaling, and extracting features")
    p.add_argument("--max_train_rows", type=int, default=500_000,
                    help="Maximum stratified rows held in RAM for classifier training; -1 uses all")
    p.add_argument("--max_val_rows", type=int, default=100_000,
                    help="Maximum stratified rows held in RAM for classifier validation; -1 uses all")
    p.add_argument("--roc_sample_rows", type=int, default=200_000,
                    help="Maximum stratified test rows used for ROC-AUC and ROC plots")
    p.add_argument("--run_raw_ablation", action="store_true",
                    help="Train and evaluate the extra raw-feature XGBoost model")
    p.add_argument("--xgb_device", choices=["cuda", "cpu"], default="cuda",
                    help="XGBoost execution device (use cpu if CUDA is unavailable)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def create_split_memmap(labels, path, chunk_size, seed):
    splits = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=labels.shape)
    rng = np.random.default_rng(seed)
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        chunk_labels = labels[start:end]
        chunk_splits = np.empty(len(chunk_labels), dtype=np.uint8)
        for class_id in np.unique(chunk_labels):
            positions = np.flatnonzero(chunk_labels == class_id)
            draws = rng.random(len(positions))
            chunk_splits[positions] = np.where(draws < 0.70, 0,
                                                np.where(draws < 0.80, 1, 2))
        splits[start:end] = chunk_splits
    splits.flush()
    return splits


def make_indices(labels, splits, split_id, path, chunk_size, class_id=None):
    count = 0
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        mask = splits[start:end] == split_id
        if class_id is not None:
            mask &= labels[start:end] == class_id
        count += int(np.count_nonzero(mask))
    indices = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(count,))
    offset = 0
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        mask = splits[start:end] == split_id
        if class_id is not None:
            mask &= labels[start:end] == class_id
        local = np.flatnonzero(mask) + start
        indices[offset:offset + len(local)] = local
        offset += len(local)
    indices.flush()
    return indices


def fit_training_scaler(features, splits, chunk_size):
    scaler = MinMaxScaler()
    train_count = 0
    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        train_mask = splits[start:end] == 0
        if np.any(train_mask):
            train_rows = features[start:end][train_mask]
            scaler.partial_fit(train_rows)
            train_count += len(train_rows)
    if train_count == 0:
        raise ValueError("No training rows were available to fit the feature scaler.")
    return scaler, train_count


def tune_class_thresholds(y_true, probabilities, num_classes):
    thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    candidate_thresholds = []
    for class_id in range(num_classes):
        binary_truth = y_true == class_id
        if not np.any(binary_truth) or np.all(binary_truth):
            candidate_thresholds.append(np.array([thresholds[class_id]]))
            continue
        precision, recall, candidates = precision_recall_curve(
            binary_truth, probabilities[:, class_id])
        if not len(candidates):
            candidate_thresholds.append(np.array([thresholds[class_id]]))
            continue
        f1_values = (2 * precision[:-1] * recall[:-1] /
                     np.maximum(precision[:-1] + recall[:-1], 1e-12))
        best_candidate = candidates[int(np.argmax(f1_values))]
        thresholds[class_id] = best_candidate
        sample_indices = np.linspace(0, len(candidates) - 1,
                                     min(31, len(candidates)), dtype=int)
        candidate_thresholds.append(np.unique(np.append(candidates[sample_indices],
                                                         best_candidate)))

    best_score = f1_score(y_true, predict_with_thresholds(probabilities, thresholds),
                          labels=range(num_classes), average="macro", zero_division=0)
    for _ in range(2):
        for class_id, candidates in enumerate(candidate_thresholds):
            for candidate in candidates:
                trial = thresholds.copy()
                trial[class_id] = candidate
                score = f1_score(
                    y_true, predict_with_thresholds(probabilities, trial),
                    labels=range(num_classes), average="macro", zero_division=0)
                if score > best_score:
                    thresholds[class_id] = candidate
                    best_score = score
    return thresholds


def predict_with_thresholds(probabilities, thresholds):
    return np.argmax(probabilities - thresholds[None, :], axis=1).astype(np.int32)


def sample_indices(labels, splits, split_id, limit, chunk_size, seed, class_id=None):
    class_counts = Counter()
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        mask = splits[start:end] == split_id
        if class_id is not None:
            mask &= labels[start:end] == class_id
        chunk_counts = np.bincount(np.asarray(labels[start:end][mask], dtype=np.int64))
        for label_id, count in enumerate(chunk_counts):
            if count:
                class_counts[label_id] += int(count)
    if limit < 0 or sum(class_counts.values()) <= limit:
        return make_indices(labels, splits, split_id,
                    os.path.join(os.path.dirname(labels.filename),
                         f"all_split_{split_id}_{class_id}.npy"), chunk_size,
                         class_id=class_id)

    total_count = sum(class_counts.values())
    ideal_quotas = {class_id: limit * count / total_count
                    for class_id, count in class_counts.items()}
    quotas = {class_id: max(1, int(ideal_quotas[class_id]))
              for class_id in class_counts}
    while sum(quotas.values()) > limit:
        largest = max(quotas, key=lambda class_id: quotas[class_id] - ideal_quotas[class_id])
        if quotas[largest] <= 1:
            break
        quotas[largest] -= 1
    while sum(quotas.values()) < limit:
        available = [class_id for class_id, count in class_counts.items()
                     if quotas[class_id] < count]
        if not available:
            break
        next_class = max(available,
                         key=lambda class_id: ideal_quotas[class_id] - quotas[class_id])
        quotas[next_class] += 1
    rng = np.random.default_rng(seed)
    reservoirs = {class_id: np.empty(0, dtype=np.int64) for class_id in quotas}
    reservoir_keys = {class_id: np.empty(0, dtype=np.float64) for class_id in quotas}
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        chunk_labels = labels[start:end]
        chunk_splits = splits[start:end]
        for class_id, quota in quotas.items():
            candidates = np.flatnonzero((chunk_labels == class_id) &
                                        (chunk_splits == split_id)) + start
            if len(candidates):
                keys = np.concatenate((reservoir_keys[class_id], rng.random(len(candidates))))
                combined = np.concatenate((reservoirs[class_id], candidates))
                if len(combined) > quota:
                    selected = np.argpartition(keys, -quota)[-quota:]
                    reservoirs[class_id] = combined[selected]
                    reservoir_keys[class_id] = keys[selected]
                else:
                    reservoirs[class_id] = combined
                    reservoir_keys[class_id] = keys
    return np.concatenate(list(reservoirs.values()))


def make_memmap_dataset(scaled, indices, side, batch_size, noise_std,
                        workers, seed, shuffle):
    batch_count = (len(indices) + batch_size - 1) // batch_size
    rng = np.random.default_rng(seed)

    def batch_index_generator():
        batch_order = np.arange(batch_count)
        if shuffle:
            rng.shuffle(batch_order)
        for batch_number in batch_order:
            row_start = batch_number * batch_size
            yield np.asarray(indices[row_start:row_start + batch_size], dtype=np.int64)

    dataset = tf.data.Dataset.from_generator(
        batch_index_generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ).apply(tf.data.experimental.assert_cardinality(batch_count))

    def load_batch(row_ids):
        def read_images(batch_row_ids):
            rows = np.asarray(scaled[batch_row_ids], dtype=np.float32)
            return D.to_img(rows)[0].astype(np.float32, copy=False)

        clean_images = tf.numpy_function(read_images, [row_ids], tf.float32)
        clean_images.set_shape((None, side, side, 1))
        noisy_images = clean_images + tf.random.normal(
            tf.shape(clean_images), stddev=noise_std, seed=seed)
        return noisy_images, clean_images

    return dataset.map(load_batch, num_parallel_calls=workers).prefetch(workers)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Chunked load to disk-backed arrays ---
    files = D.find_csvs(args.data_dir)
    if not files:
        raise SystemExit(f"No CSV files found under {args.data_dir!r}. "
                          f"Point --data_dir at your CIC-IoT-2023 CSVs.")
    if args.max_files != -1:
        files = files[:args.max_files]
    cache_dir = os.path.join(args.output_dir, "memmap_cache")
    print(f"Preparing {len(files)} CSV file(s) in chunks of {args.chunk_size:,} rows...")
    features, labels, feat_cols, classes = D.prepare_memmaps(
        files, cache_dir, chunk_size=args.chunk_size, sample_frac=args.sample_frac,
        seed=args.seed, preserve_rare=not args.no_preserve_rare,
        granularity=args.granularity)
    print(f"Rows: {len(labels):,} | classes: {classes} | features: {len(feat_cols)}")

    splits = create_split_memmap(labels, os.path.join(cache_dir, "splits.npy"),
                                 args.chunk_size, args.seed)
    train_indices = sample_indices(labels, splits, 0, args.max_train_rows,
                                   args.chunk_size, args.seed)
    val_indices = sample_indices(labels, splits, 1, args.max_val_rows,
                                 args.chunk_size, args.seed + 1)
    test_indices = make_indices(labels, splits, 2,
                                 os.path.join(cache_dir, "test_indices.npy"), args.chunk_size)

    benign_name = "Benign" if "Benign" in classes else BENIGN_LABEL
    if benign_name not in classes:
        raise ValueError(f"Benign class {benign_name!r} is missing from the processed data.")
    benign_id = classes.index(benign_name)
    scaler, scaler_train_count = fit_training_scaler(features, splits, args.chunk_size)

    scaled = np.lib.format.open_memmap(
        os.path.join(cache_dir, "scaled.npy"), mode="w+", dtype=np.float32,
        shape=features.shape)
    for start in range(0, len(features), args.chunk_size):
        end = min(start + args.chunk_size, len(features))
        scaled[start:end] = np.clip(scaler.transform(features[start:end]), 0, 1)
    scaled.flush()
    side = int(np.ceil(np.sqrt(len(feat_cols))))
    benign_train_indices = sample_indices(
        labels, splits, 0, args.max_dcae_train_rows, args.chunk_size, args.seed,
        class_id=benign_id)
    benign_val_indices = sample_indices(
        labels, splits, 1, args.dcae_val_rows, args.chunk_size, args.seed + 1,
        class_id=benign_id)
    print(f"image shape: ({side}, {side}, 1) | benign-train rows for DCAE: {len(benign_train_indices):,} "
          f"| scaler fit rows: {scaler_train_count:,}")

    # --- Stage 1: train the DCAE, unsupervised, benign-only ---
    dcae, encoder = build_dcae(side, LATENT_DIM)
    dcae.compile(optimizer=optimizers.Adam(1e-3),
                 loss=tf.keras.losses.MeanSquaredError())
    dcae_train_data = make_memmap_dataset(
        scaled, benign_train_indices, side, args.batch_size, NOISE_STD,
        args.data_workers, args.seed, shuffle=True)
    dcae_val_data = make_memmap_dataset(
        scaled, benign_val_indices, side, args.batch_size, NOISE_STD,
        args.data_workers, args.seed + 1, shuffle=False)
    hist = dcae.fit(
        dcae_train_data,
        validation_data=dcae_val_data,
        epochs=args.epochs,
        shuffle=False,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min",
                           restore_best_weights=True)],
        verbose=2,
    )
    plt.figure()
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("DCAE training loss (benign only)")
    plt.savefig(os.path.join(args.output_dir, "dcae_training_loss.png"), bbox_inches="tight")
    plt.close()

    # --- Stage 2: write fused features in bounded batches ---
    fused = np.lib.format.open_memmap(
        os.path.join(cache_dir, "fused.npy"), mode="w+", dtype=np.float32,
        shape=(len(features), len(feat_cols) + LATENT_DIM + 1))
    for start in range(0, len(scaled), args.chunk_size):
        end = min(start + args.chunk_size, len(scaled))
        rows = np.asarray(scaled[start:end], dtype=np.float32)
        images, _ = D.to_img(rows)
        latent = encoder.predict(images, batch_size=args.batch_size, verbose=0)
        reconstruction = dcae.predict(images, batch_size=args.batch_size, verbose=0)
        error = np.mean((images - reconstruction) ** 2, axis=(1, 2, 3))[:, None]
        fused[start:end] = np.concatenate((rows, latent, error), axis=1)
    fused.flush()

    Xtr = np.asarray(fused[train_indices], dtype=np.float32)
    ytr = np.asarray(labels[train_indices], dtype=np.int32)
    ytr_orig = ytr.copy()
    Xva = np.asarray(fused[val_indices], dtype=np.float32)
    yva = np.asarray(labels[val_indices], dtype=np.int32)
    print("classifier train/validation:", Xtr.shape, Xva.shape,
          "| full test rows:", len(test_indices))

    # --- Stage 3: supervised classifier on the fused features ---
    print("Train-split class counts:", Counter(ytr))
    sample_w = compute_sample_weight("balanced", ytr)

    clf = XGBClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, max_delta_step=1,
        objective="multi:softprob", num_class=len(classes),
        eval_metric="mlogloss", tree_method="hist", device=args.xgb_device,
        early_stopping_rounds=30, random_state=args.seed, n_jobs=-1,
    )
    # For multi:softprob, per-row balanced weights support every class; scale_pos_weight is binary-only.
    clf.fit(Xtr, ytr, sample_weight=sample_w, eval_set=[(Xva, yva)], verbose=False)
    validation_probabilities = clf.predict_proba(Xva)
    class_thresholds = tune_class_thresholds(yva, validation_probabilities, len(classes))
    validation_predictions = predict_with_thresholds(validation_probabilities, class_thresholds)
    validation_macro_f1 = f1_score(yva, validation_predictions, labels=range(len(classes)),
                                   average="macro", zero_division=0)
    print("Validation class thresholds:", dict(zip(classes, class_thresholds.tolist())))
    print(f"Threshold-tuned validation macro F1: {validation_macro_f1:.4f}")

    # --- Evaluate the full test split with disk-backed predictions ---
    test_count = len(test_indices)
    yte = np.lib.format.open_memmap(os.path.join(cache_dir, "y_test.npy"), mode="w+",
                                    dtype=np.int32, shape=(test_count,))
    y_pred = np.lib.format.open_memmap(os.path.join(cache_dir, "y_pred.npy"), mode="w+",
                                       dtype=np.int32, shape=(test_count,))
    y_proba = np.lib.format.open_memmap(
        os.path.join(cache_dir, "y_proba.npy"), mode="w+", dtype=np.float32,
        shape=(test_count, len(classes)))
    for start in range(0, test_count, args.chunk_size):
        end = min(start + args.chunk_size, test_count)
        row_ids = test_indices[start:end]
        batch = np.asarray(fused[row_ids], dtype=np.float32)
        yte[start:end] = labels[row_ids]
        probabilities = clf.predict_proba(batch)
        y_pred[start:end] = predict_with_thresholds(probabilities, class_thresholds)
        y_proba[start:end] = probabilities
    yte.flush(); y_pred.flush(); y_proba.flush()

    report = classification_report(yte, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    acc = accuracy_score(yte, y_pred)
    precision_macro = precision_score(yte, y_pred, labels=range(len(classes)),
                                      average="macro", zero_division=0)
    precision_weighted = precision_score(yte, y_pred, labels=range(len(classes)),
                                         average="weighted", zero_division=0)
    recall_macro = recall_score(yte, y_pred, labels=range(len(classes)),
                                average="macro", zero_division=0)
    recall_weighted = recall_score(yte, y_pred, labels=range(len(classes)),
                                   average="weighted", zero_division=0)
    f1_macro = f1_score(yte, y_pred, labels=range(len(classes)), average="macro",
                        zero_division=0)
    f1_weighted = f1_score(yte, y_pred, labels=range(len(classes)), average="weighted",
                           zero_division=0)
    cm_counts = confusion_matrix(yte, y_pred, labels=range(len(classes)))
    support = cm_counts.sum(axis=1)
    class_auc = []
    roc_data = []
    rng = np.random.default_rng(args.seed)
    per_class_roc_rows = max(1, args.roc_sample_rows // len(classes))
    roc_plot_indices = []
    for class_id in range(len(classes)):
        class_positions = np.flatnonzero(yte == class_id)
        take = min(len(class_positions), per_class_roc_rows)
        if take:
            roc_plot_indices.append(rng.choice(class_positions, size=take, replace=False))
    roc_plot_indices = (np.concatenate(roc_plot_indices) if roc_plot_indices
                        else np.empty(0, dtype=np.int64))
    for class_id, class_name in enumerate(classes):
        sampled_truth = yte[roc_plot_indices] == class_id
        sampled_scores = y_proba[roc_plot_indices, class_id]
        if np.any(sampled_truth) and np.any(~sampled_truth):
            auc_value = roc_auc_score(sampled_truth, sampled_scores)
            fpr, tpr, _ = roc_curve(sampled_truth, sampled_scores)
            roc_data.append((class_name, fpr, tpr, auc_value))
        else:
            auc_value = float("nan")
        class_auc.append(auc_value)
    valid_auc = np.isfinite(class_auc) & (support > 0)
    auc_macro = float(np.mean(np.asarray(class_auc)[valid_auc])) if np.any(valid_auc) else float("nan")
    auc_weighted = (float(np.average(np.asarray(class_auc)[valid_auc], weights=support[valid_auc]))
                    if np.any(valid_auc) else float("nan"))

    summary = (
        f"\nAccuracy            : {acc:.4f}\n"
        f"Precision (macro)   : {precision_macro:.4f}\n"
        f"Precision (weighted): {precision_weighted:.4f}\n"
        f"Recall (macro)      : {recall_macro:.4f}\n"
        f"Recall (weighted)   : {recall_weighted:.4f}\n"
        f"F1 (macro)          : {f1_macro:.4f}\n"
        f"F1 (weighted)       : {f1_weighted:.4f}\n"
        f"ROC-AUC (macro OvR) : {auc_macro:.4f}\n"
        f"ROC-AUC (weighted)  : {auc_weighted:.4f}\n"
    )
    print(report); print(summary)

    # --- Confusion matrix plot ---
    cm = confusion_matrix(yte, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Normalized confusion matrix")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metric_macro = [acc, precision_macro, recall_macro, f1_macro, auc_macro]
    metric_weighted = [acc, precision_weighted, recall_weighted, f1_weighted, auc_weighted]
    x = np.arange(len(metric_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    macro_bars = ax.bar(x - width / 2, metric_macro, width, label="Macro")
    weighted_bars = ax.bar(x + width / 2, metric_weighted, width, label="Weighted")
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Evaluation metrics")
    ax.bar_label(macro_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(weighted_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "overall_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for class_name, fpr, tpr, auc_value in roc_data:
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (sampled plot, full-test AUC)")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "roc_auc.png"), bbox_inches="tight")
    plt.close(fig)

    true_positives = np.diag(cm_counts)
    precision_by_class = true_positives / np.maximum(cm_counts.sum(axis=0), 1)
    recall_by_class = true_positives / np.maximum(support, 1)
    f1_by_class = (2 * precision_by_class * recall_by_class /
                   np.maximum(precision_by_class + recall_by_class, 1e-12))
    class_accuracy = (cm_counts.sum() - cm_counts.sum(axis=1) - cm_counts.sum(axis=0) +
                      2 * true_positives) / max(int(cm_counts.sum()), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision_by_class, width, label="Precision")
    ax.bar(x, recall_by_class, width, label="Recall")
    ax.bar(x + width, f1_by_class, width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Per-class evaluation metrics")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "evaluation_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    metric_rows = []
    for class_id, class_name in enumerate(classes):
        metric_rows.append({"scope": class_name, "precision": precision_by_class[class_id],
                            "accuracy": class_accuracy[class_id], "recall": recall_by_class[class_id],
                            "f1_score": f1_by_class[class_id], "roc_auc_ovr": class_auc[class_id],
                            "support": int(support[class_id])})
    total = max(int(cm_counts.sum()), 1)
    metric_rows.extend([
        {"scope": "macro_avg", "precision": precision_macro, "accuracy": acc,
         "recall": recall_macro, "f1_score": f1_macro, "roc_auc_ovr": auc_macro,
         "support": total},
        {"scope": "weighted_avg", "precision": precision_weighted, "accuracy": acc,
         "recall": recall_weighted, "f1_score": f1_weighted,
         "roc_auc_ovr": auc_weighted, "support": total},
    ])
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(args.output_dir, "evaluation_metrics.csv"), index=False)

    # --- Optional ablation: fused features versus raw-only features ---
    ablation = ""
    if args.run_raw_ablation:
        clf_raw = XGBClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, max_delta_step=1,
            objective="multi:softprob", num_class=len(classes),
            eval_metric="mlogloss", tree_method="hist", device=args.xgb_device,
            early_stopping_rounds=30, random_state=args.seed, n_jobs=-1,
        )
        raw_sample_w = compute_sample_weight("balanced", ytr_orig)
        Xtr_raw = np.asarray(scaled[train_indices], dtype=np.float32)
        Xva_raw = np.asarray(scaled[val_indices], dtype=np.float32)
        clf_raw.fit(Xtr_raw, ytr_orig, sample_weight=raw_sample_w,
                    eval_set=[(Xva_raw, yva)], verbose=False)
        pred_raw = np.empty(test_count, dtype=np.int32)
        for start in range(0, test_count, args.chunk_size):
            end = min(start + args.chunk_size, test_count)
            pred_raw[start:end] = clf_raw.predict(scaled[test_indices[start:end]])
        ablation = (
            f"Raw-features-only   F1 (macro): "
            f"{f1_score(yte, pred_raw, average='macro', zero_division=0):.4f}\n"
            f"DCAE-fused-features F1 (macro): {f1_macro:.4f}\n"
        )
        print(ablation)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report + "\n" + summary + "\n" + ablation)

    # --- Save artifacts ---
    dcae.save(os.path.join(args.output_dir, "dcae.keras"))
    encoder.save(os.path.join(args.output_dir, "dcae_encoder.keras"))
    clf.save_model(os.path.join(args.output_dir, "xgb_classifier.json"))
    joblib.dump({"scaler": scaler, "classes": classes, "feat_cols": feat_cols,
                 "use_log1p": USE_LOG1P, "granularity": args.granularity,
                 "class_thresholds": class_thresholds},
                os.path.join(args.output_dir, "preproc.joblib"))
    print(f"\nDone. Artifacts and reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
