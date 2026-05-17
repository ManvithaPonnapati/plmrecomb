from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class CorrelationResult:
    feature: str
    n: int
    pointbiserial_r: float
    pointbiserial_p: float
    roc_auc: float
    class0_mean: float
    class1_mean: float


def correlate_feature(
    y: np.ndarray,
    feature: np.ndarray,
    feature_name: str,
    higher_is_functional: bool = True,
) -> CorrelationResult:
    mask = ~np.isnan(feature)
    yv, fv = y[mask], feature[mask]
    r, pval = pointbiserialr(yv, fv)
    score = fv if higher_is_functional else -fv
    try:
        auc = roc_auc_score(yv, score)
    except ValueError:
        auc = float("nan")
    return CorrelationResult(
        feature=feature_name,
        n=int(mask.sum()),
        pointbiserial_r=float(r),
        pointbiserial_p=float(pval),
        roc_auc=float(auc),
        class0_mean=float(fv[yv == 0].mean()) if (yv == 0).any() else float("nan"),
        class1_mean=float(fv[yv == 1].mean()) if (yv == 1).any() else float("nan"),
    )


def plot_feature_by_class(
    y: np.ndarray,
    feature: np.ndarray,
    feature_name: str,
    outpath: Path,
    title: Optional[str] = None,
) -> None:
    mask = ~np.isnan(feature)
    yv, fv = y[mask], feature[mask]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([fv[yv == 0], fv[yv == 1]], labels=["non-functional (0)", "functional (1)"])
    ax.set_ylabel(feature_name)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# =====================================================================
# Classifier comparison
# =====================================================================
def make_classifiers() -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            min_samples_leaf=2, random_state=0, n_jobs=-1,
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, random_state=0
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 64), max_iter=2000,
            early_stopping=True, random_state=0,
        ),
    }


@dataclass
class CVResults:
    summary: pd.DataFrame
    confusion: Dict[str, np.ndarray]
    fold_scores: Dict[str, Dict[str, List[float]]]

    @property
    def best_model_name(self) -> str:
        return str(self.summary.iloc[0]["model"])


def stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> CVResults:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = make_classifiers()
    fold_scores: Dict[str, Dict[str, List[float]]] = {
        n: {"roc_auc": [], "pr_auc": [], "f1": [], "acc": []} for n in models
    }
    cms: Dict[str, np.ndarray] = {n: np.zeros((2, 2), dtype=int) for n in models}

    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        for name, model in make_classifiers().items():
            model.fit(Xtr, ytr)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xte)[:, 1]
            else:
                proba = model.decision_function(Xte)
            pred = (proba >= 0.5).astype(int)
            try:
                fold_scores[name]["roc_auc"].append(roc_auc_score(yte, proba))
            except ValueError:
                fold_scores[name]["roc_auc"].append(float("nan"))
            fold_scores[name]["pr_auc"].append(average_precision_score(yte, proba))
            fold_scores[name]["f1"].append(f1_score(yte, pred, zero_division=0))
            fold_scores[name]["acc"].append(float((pred == yte).mean()))
            cms[name] += confusion_matrix(yte, pred, labels=[0, 1])

    rows = []
    for name, m in fold_scores.items():
        rows.append({
            "model": name,
            "roc_auc_mean": float(np.nanmean(m["roc_auc"])),
            "roc_auc_std":  float(np.nanstd(m["roc_auc"])),
            "pr_auc_mean":  float(np.nanmean(m["pr_auc"])),
            "f1_mean":      float(np.nanmean(m["f1"])),
            "acc_mean":     float(np.nanmean(m["acc"])),
        })
    summary = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return CVResults(summary=summary, confusion=cms, fold_scores=fold_scores)


@dataclass
class FittedClassifier:
    name: str
    scaler: StandardScaler
    model: object

    def proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)[:, 1]
        return self.model.decision_function(Xs)


def fit_final(name: str, X: np.ndarray, y: np.ndarray) -> FittedClassifier:
    scaler = StandardScaler().fit(X)
    model = make_classifiers()[name]
    model.fit(scaler.transform(X), y)
    return FittedClassifier(name=name, scaler=scaler, model=model)
