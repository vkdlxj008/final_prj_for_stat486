"""Shared folds and per-fold metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold


def folds_path(results_dir: Path, cv_cfg: dict) -> Path:
    name = f"folds_{cv_cfg['n_splits']}x{cv_cfg['n_repeats']}_seed{cv_cfg['seed']}.csv"
    return results_dir / name


def load_or_make_folds(df: pd.DataFrame, y: pd.Series, id_col: str, cv_cfg: dict, results_dir: Path) -> pd.DataFrame:
    """One row per (repeat, fold, passenger) in the validation set.

    Saved once and reused, so every method sees identical folds. An existing
    file is never overwritten; it is checked against the current data instead.
    """
    path = folds_path(results_dir, cv_cfg)
    if path.exists():
        folds = pd.read_csv(path)
        if set(folds[id_col]) != set(df[id_col]):
            raise ValueError(f"{path} does not match the training IDs; refusing to reuse it.")
        return folds

    rskf = RepeatedStratifiedKFold(
        n_splits=cv_cfg["n_splits"], n_repeats=cv_cfg["n_repeats"], random_state=cv_cfg["seed"]
    )
    rows = []
    for i, (_, val_idx) in enumerate(rskf.split(df, y)):
        rep, fold = divmod(i, cv_cfg["n_splits"])
        rows.append(pd.DataFrame({"repeat": rep, "fold": fold, id_col: df[id_col].to_numpy()[val_idx]}))
    folds = pd.concat(rows, ignore_index=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return folds


def cross_validate(estimator, X: pd.DataFrame, y: pd.Series, ids: pd.Series, folds: pd.DataFrame, id_col: str, eps: float) -> pd.DataFrame:
    """Fit a fresh clone per fold; return accuracy and log loss per fold."""
    out = []
    for (rep, fold), grp in folds.groupby(["repeat", "fold"]):
        val_mask = ids.isin(grp[id_col]).to_numpy()
        est = clone(estimator).fit(X[~val_mask], y[~val_mask])
        p_val = np.clip(est.predict_proba(X[val_mask])[:, 1], eps, 1 - eps)
        p_tr = est.predict_proba(X[~val_mask])[:, 1]
        out.append({
            "repeat": rep,
            "fold": fold,
            "n_val": int(val_mask.sum()),
            "accuracy": accuracy_score(y[val_mask], (p_val >= 0.5).astype(int)),
            "log_loss": log_loss(y[val_mask], p_val, labels=[0, 1]),
            "train_accuracy": accuracy_score(y[~val_mask], (p_tr >= 0.5).astype(int)),
        })
    return pd.DataFrame(out)
