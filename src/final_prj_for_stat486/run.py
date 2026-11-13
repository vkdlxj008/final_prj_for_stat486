"""Run one pipeline end to end from a config.

    uv run python -m final_prj_for_stat486.run --config configs/pipeline1.toml

Writes per-fold scores, appends one row to results/runs.csv, and writes a
Kaggle submission fit on the full training set.
"""

import argparse
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd

from .evaluate import cross_validate, folds_path, load_or_make_folds
from .features import add_features
from .models import build_estimator


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_row(path: Path, row: dict) -> None:
    """Append only; never rewrite earlier runs."""
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())

    paths = cfg["paths"]
    results_dir = Path(paths["results_dir"])
    id_col, target = cfg["data"]["id_col"], cfg["data"]["target"]

    train = add_features(pd.read_csv(paths["train"]), cfg.get("features", {}))
    test = add_features(pd.read_csv(paths["test"]), cfg.get("features", {}))
    y = train[target]
    X = train.drop(columns=[target])

    folds = load_or_make_folds(train, y, id_col, cfg["cv"], results_dir)
    estimator = build_estimator(cfg["model"], cfg.get("preprocess"))

    scores = cross_validate(estimator, X, y, train[id_col], folds, id_col, cfg["cv"]["clip_eps"])
    name = cfg["name"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fold_dir = results_dir / "fold_scores"
    fold_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(fold_dir / f"{name}_{stamp}.csv", index=False)

    final = estimator.fit(X, y)
    sub_dir = Path(paths["submissions_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / f"{name}.csv"
    pd.DataFrame({id_col: test[id_col], target: final.predict(test)}).to_csv(sub_path, index=False)

    row = {
        "timestamp": stamp,
        "name": name,
        "config": args.config,
        "commit": git_commit(),
        "folds": folds_path(results_dir, cfg["cv"]).name,
        "n_folds": len(scores),
        "acc_mean": scores["accuracy"].mean(),
        "acc_sd": scores["accuracy"].std(),
        "logloss_mean": scores["log_loss"].mean(),
        "logloss_sd": scores["log_loss"].std(),
        "train_acc_mean": scores["train_accuracy"].mean(),
        "fold_scores": f"fold_scores/{name}_{stamp}.csv",
        "submission": str(sub_path),
        "kaggle_score": "",
    }
    append_row(results_dir / "runs.csv", row)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
