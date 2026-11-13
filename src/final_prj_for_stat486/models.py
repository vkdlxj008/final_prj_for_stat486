"""Estimators built from config. Every preprocessing step lives inside the
returned estimator so it is fit on the training fold only."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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


def build_preprocessor(prep_cfg: dict) -> ColumnTransformer:
    transformers = []
    for name, spec in prep_cfg.items():
        steps = [("impute", SimpleImputer(strategy=spec["impute"]))]
        if spec.get("onehot", False):
            steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
        transformers.append((name, Pipeline(steps), spec["columns"]))
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(model_cfg: dict, prep_cfg: dict | None):
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
    return Pipeline([("prep", build_preprocessor(prep_cfg)), ("model", clf)])
