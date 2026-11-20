"""Deterministic feature engineering (no fitting, so no leakage across folds)."""

import pandas as pd


def extract_title(names: pd.Series, title_map: dict[str, str], keep: list[str], other: str) -> pd.Series:
    """Pull the title from 'Surname, Title. Given names' and group rare titles.

    title_map renames variants (e.g. Mlle -> Miss); titles not in `keep` after
    mapping become `other`.
    """
    raw = names.str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    mapped = raw.replace(title_map)
    return mapped.where(mapped.isin(keep), other)


def add_group_size(train: pd.DataFrame, test: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Number of passengers sharing `key` (e.g. Ticket), counted over train + test.

    Uses no labels, so it does not leak the target. It is transductive: the
    count includes test passengers, which is allowed because test features
    are given to every competitor.
    """
    counts = pd.concat([train[key], test[key]]).value_counts()
    return (
        train.assign(group_size=train[key].map(counts).astype(int)),
        test.assign(group_size=test[key].map(counts).astype(int)),
    )


def add_features(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """Return a copy of df with derived columns requested in the config."""
    out = df.copy()
    title_cfg = feat_cfg.get("title")
    if title_cfg:
        out["Title"] = extract_title(
            out["Name"], title_cfg.get("map", {}), title_cfg["keep"], title_cfg["other"]
        )
    return out
