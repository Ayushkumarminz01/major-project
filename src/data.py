"""Loading and preprocessing for local CIC-IoT-2023 CSV shards."""
import glob
import math
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import GROUP_MAP, MIN_CLASS_N, USE_LOG1P

# These groups dominate CIC-IoT-2023 (DDoS + DoS alone are >85% of all rows).
# When subsampling for RAM, only these are cut — every row of the already-rare
# groups (Benign, Mirai, Recon, Spoofing, Web, BruteForce) is always kept.
MAJORITY_GROUPS_TO_SUBSAMPLE = {"DDoS", "DoS"}


def find_csvs(data_dir):
    return sorted(glob.glob(f"{data_dir}/**/*.csv", recursive=True))


def load_frames(files, sample_frac=1.0, seed=42, preserve_rare=True):
    """Load CSVs as float32 (half the memory of pandas' float64 default) and,
    optionally, keep only a random fraction of each file's rows so the whole
    dataset doesn't have to fit in RAM at once.

    If preserve_rare=True (default), sample_frac is only applied to the huge
    DDoS/DoS rows in each file — every row belonging to an already-rare class
    (BruteForce, Web, Recon, Spoofing, Mirai, Benign) is always kept in full,
    so subsampling for RAM doesn't compound the class-imbalance problem.
    """
    parts = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        label_col = next(c for c in df.columns if c.lower() == "label")
        df = df.rename(columns={label_col: "label"})
        num_cols = df.columns.drop("label")
        df[num_cols] = df[num_cols].astype("float32")
        if sample_frac < 1.0:
            if preserve_rare:
                grp = df["label"].map(GROUP_MAP).fillna("Other")
                is_majority = grp.isin(MAJORITY_GROUPS_TO_SUBSAMPLE)
                majority_rows = df[is_majority].sample(frac=sample_frac, random_state=seed)
                rare_rows = df[~is_majority]
                df = pd.concat([majority_rows, rare_rows], ignore_index=True)
            else:
                df = df.sample(frac=sample_frac, random_state=seed)
        parts.append(df)
    data = pd.concat(parts, ignore_index=True)
    return data


def _deduplicate_chunk(frame, connection):
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    hashes = hashes.view(np.int64)
    connection.executemany(
        "INSERT OR IGNORE INTO batch_hashes (row_hash, row_index) VALUES (?, ?)",
        ((int(row_hash), index) for index, row_hash in enumerate(hashes)),
    )
    keep = np.zeros(len(frame), dtype=bool)
    new_rows = connection.execute(
        "SELECT row_index FROM batch_hashes "
        "WHERE row_hash NOT IN (SELECT row_hash FROM seen_hashes)"
    )
    keep[[index for (index,) in new_rows]] = True
    connection.execute(
        "INSERT OR IGNORE INTO seen_hashes SELECT row_hash FROM batch_hashes"
    )
    connection.execute("DELETE FROM batch_hashes")
    connection.commit()
    return frame.loc[keep].reset_index(drop=True), int((~keep).sum())


def _iter_clean_chunks(files, chunk_size, sample_frac, seed, preserve_rare,
                       dedupe_db_path, dedupe_stats=None):
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(dedupe_db_path + suffix)
        except FileNotFoundError:
            pass

    connection = sqlite3.connect(dedupe_db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE seen_hashes (row_hash INTEGER PRIMARY KEY) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TEMP TABLE batch_hashes "
        "(row_hash INTEGER PRIMARY KEY, row_index INTEGER NOT NULL) WITHOUT ROWID"
    )
    chunk_number = 0
    expected_features = None
    try:
        for path in files:
            for frame in pd.read_csv(path, chunksize=chunk_size):
                frame.columns = frame.columns.str.strip()
                label_col = next(c for c in frame.columns if c.lower() == "label")
                frame = frame.rename(columns={label_col: "label"})
                feat_cols = [c for c in frame.columns if c != "label"]
                if expected_features is None:
                    expected_features = feat_cols
                elif feat_cols != expected_features:
                    raise ValueError(f"CSV feature columns do not match in {path!r}")

                if sample_frac < 1.0:
                    if preserve_rare:
                        groups = frame["label"].map(GROUP_MAP).fillna("Other")
                        majority = groups.isin(MAJORITY_GROUPS_TO_SUBSAMPLE)
                        selected = frame.loc[majority].sample(
                            frac=sample_frac, random_state=seed + chunk_number)
                        frame = pd.concat([selected, frame.loc[~majority]], ignore_index=True)
                    else:
                        frame = frame.sample(frac=sample_frac, random_state=seed + chunk_number)
                chunk_number += 1

                frame[feat_cols] = frame[feat_cols].apply(pd.to_numeric, errors="coerce")
                frame = frame.replace([np.inf, -np.inf], np.nan)
                frame = frame.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)
                if not frame.empty:
                    frame[feat_cols] = frame[feat_cols].astype(np.float32)
                    frame, duplicate_count = _deduplicate_chunk(frame, connection)
                    if dedupe_stats is not None:
                        dedupe_stats["duplicates"] += duplicate_count
                    if not frame.empty:
                        yield frame, feat_cols
    finally:
        connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(dedupe_db_path + suffix)
            except FileNotFoundError:
                pass


def prepare_memmaps(files, cache_dir, chunk_size=100_000, sample_frac=1.0,
                    seed=42, preserve_rare=True, granularity="8class"):
    """Clean, deduplicate, and store CSV features/labels in disk-backed arrays."""
    os.makedirs(cache_dir, exist_ok=True)
    label_counts = {}
    feat_cols = None
    row_count = 0
    dedupe_stats = {"duplicates": 0}
    dedupe_db_path = os.path.join(cache_dir, "dedupe.sqlite")
    for frame, columns in _iter_clean_chunks(
            files, chunk_size, sample_frac, seed, preserve_rare,
            dedupe_db_path, dedupe_stats):
        feat_cols = columns
        counts = frame["label"].value_counts()
        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + int(count)
        row_count += len(frame)
    if not row_count or feat_cols is None:
        raise ValueError("No usable rows remain after cleaning and sampling.")
    if granularity == "8class":
        target_by_label = {label: GROUP_MAP.get(label, "Other") for label in label_counts}
    else:
        target_by_label = {
            label: label if count >= MIN_CLASS_N else "Other"
            for label, count in label_counts.items()
        }
    classes = sorted(set(target_by_label.values()))
    class_to_id = {name: index for index, name in enumerate(classes)}

    features = np.lib.format.open_memmap(
        os.path.join(cache_dir, "features.npy"), mode="w+", dtype=np.float32,
        shape=(row_count, len(feat_cols)))
    labels = np.lib.format.open_memmap(
        os.path.join(cache_dir, "labels.npy"), mode="w+", dtype=np.int32,
        shape=(row_count,))
    offset = 0
    for frame, columns in _iter_clean_chunks(
            files, chunk_size, sample_frac, seed, preserve_rare, dedupe_db_path):
        if columns != feat_cols:
            raise ValueError("CSV feature columns changed between preprocessing passes.")
        end = offset + len(frame)
        values = frame[feat_cols].to_numpy(dtype=np.float32, copy=False)
        if USE_LOG1P:
            values = np.log1p(np.clip(values, 0, None))
        features[offset:end] = values
        targets = frame["label"].map(target_by_label).to_numpy()
        labels[offset:end] = np.fromiter(
            (class_to_id[target] for target in targets), dtype=np.int32, count=len(targets))
        offset = end
    if offset != row_count:
        raise RuntimeError(f"Preprocessing row count changed: expected {row_count}, wrote {offset}.")
    features.flush()
    labels.flush()
    return features, labels, feat_cols, classes


def add_target(data, granularity):
    if granularity == "8class":
        data["target"] = data["label"].map(GROUP_MAP).fillna("Other")
    else:
        vc = data["label"].value_counts()
        rare = vc[vc < MIN_CLASS_N].index
        data["target"] = data["label"].where(~data["label"].isin(rare), "Other")
    return data


def clean(data):
    feat_cols = [c for c in data.columns if c not in ("label", "target")]
    data[feat_cols] = data[feat_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data, feat_cols


def to_arrays(data, feat_cols, use_log1p):
    X_raw = data[feat_cols].values.astype(np.float32)
    y_bin = (data["target"] != "Benign").astype(int).values
    classes = sorted(data["target"].unique())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_multi = data["target"].map(class_to_id).values.astype(np.int64)
    if use_log1p:
        X_raw = np.log1p(np.clip(X_raw, 0, None))
    return X_raw, y_bin, y_multi, classes


def fit_scaler(X_raw, benign_train_idx):
    return MinMaxScaler().fit(X_raw[benign_train_idx])


def scale(scaler, X_raw):
    return np.clip(scaler.transform(X_raw), 0, 1).astype(np.float32)


def to_img(X_scaled):
    n_feat = X_scaled.shape[1]
    side = math.ceil(math.sqrt(n_feat))
    pad = side * side - n_feat
    img = np.pad(X_scaled, ((0, 0), (0, pad))).reshape(-1, side, side, 1)
    return img, side
