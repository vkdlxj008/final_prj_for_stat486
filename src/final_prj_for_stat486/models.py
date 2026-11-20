"""Estimators built from config. Every preprocessing step lives inside the
returned estimator so it is fit on the training fold only."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class GroupRateClassifier(ClassifierMixin, BaseEstimator):
    """p_hat = training-fold survival rate of the passenger's group (e.g. Sex).

    Hard predictions match the Kaggle sex-only rule; probabilities make log
    loss finite. Unseen groups fall back to the overall training rate.
    """

    def __init__(self, column: str = "Sex"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.rates_ = y.groupby(X[self.column]).mean().to_dict()
        self.prior_ = float(y.mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        p1 = X[self.column].map(self.rates_).fillna(self.prior_).to_numpy(dtype=float)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CompanionFeatures(TransformerMixin, BaseEstimator):
    """Survival of a passenger's known travel companions, from training labels only.

    Companions are other passengers with the same `group_col` (Ticket). In fit,
    the labels of the training rows are stored. In transform:
      - a training row uses its companions' labels with its own label removed
        (leave-one-out), so no row ever sees its own label;
      - any other row (validation fold, Kaggle test) uses all training
        companions' labels.
    Rows are identified by `id_col`, so this works inside a Pipeline, where
    transform is called on both training and validation rows.

    Adds: companion_surv_rate (filled with `fill` when there are no known
    companions), n_known_companions, has_known_companion.
    """

    def __init__(self, group_col: str = "Ticket", id_col: str = "PassengerId", fill: float = 0.5):
        self.group_col = group_col
        self.id_col = id_col
        self.fill = fill

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        df = pd.DataFrame({"g": X[self.group_col].to_numpy(), "y": y})
        agg = df.groupby("g")["y"].agg(["sum", "count"])
        self.sum_ = agg["sum"].to_dict()
        self.count_ = agg["count"].to_dict()
        self.own_label_ = dict(zip(X[self.id_col].to_numpy(), y))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        g = X[self.group_col]
        s = g.map(self.sum_).fillna(0.0).to_numpy(dtype=float)
        c = g.map(self.count_).fillna(0.0).to_numpy(dtype=float)
        own = X[self.id_col].map(self.own_label_)
        in_train = own.notna().to_numpy()
        s = s - own.fillna(0.0).to_numpy(dtype=float)
        c = c - in_train.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(c > 0, s / np.where(c > 0, c, 1.0), self.fill)
        out = X.copy()
        out["companion_surv_rate"] = rate
        out["n_known_companions"] = c
        out["has_known_companion"] = (c > 0).astype(int)
        return out


def build_preprocessor(prep_cfg: dict) -> ColumnTransformer:
    transformers = []
    for name, spec in prep_cfg.items():
        steps = [("impute", SimpleImputer(strategy=spec["impute"]))]
        if spec.get("onehot", False):
            steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
        transformers.append((name, Pipeline(steps), spec["columns"]))
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(model_cfg: dict, prep_cfg: dict | None, companion_cfg: dict | None = None):
    kind = model_cfg["type"]
    params = model_cfg.get("params", {})
    if kind == "group_rate":
        return GroupRateClassifier(**params)
    if kind == "random_forest":
        clf = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {kind}")
    if prep_cfg is None:
        return clf
    steps = [("prep", build_preprocessor(prep_cfg)), ("model", clf)]
    if companion_cfg:
        steps.insert(0, ("companions", CompanionFeatures(**companion_cfg)))
    return Pipeline(steps)
