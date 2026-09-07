"""Kaggle runner for the ccfraud tuned + stacked-ensemble experiment.

Self-contained (no local package import) so it runs as a Kaggle kernel with the
`mlg-ulb/creditcardfraud` dataset attached. Mirrors src/ccfraud/: stratified
split, leakage-free scale->resample->estimator pipelines, Optuna search over
LightGBM + XGBoost, out-of-fold threshold, isotonic calibration, and a
StackingClassifier. Writes /kaggle/working/metrics.json.

The ULB dataset's honest PR-AUC ceiling is ~0.88; this measures how much a
proper search + stack actually adds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

SEED = 42
CV = 5
N_TRIALS = 60
MIN_PRECISION = 0.90
SELECT_SAMPLE = 120_000
STRATEGIES = ("none", "smote", "class_weight")
OUT = Path("/kaggle/working")

CSV = next(Path("/kaggle/input").rglob("creditcard.csv"))
df = pd.read_csv(CSV)
X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
y = df["Class"].to_numpy().astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"train {X_tr.shape} pos={y_tr.sum()}  test {X_te.shape} pos={y_te.sum()}")

cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)


def build_pipeline(estimator, strategy):
    est = clone(estimator)
    steps = [("scaler", RobustScaler())]
    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=SEED)))
    elif strategy == "smotetomek":
        steps.append(("resample", SMOTETomek(random_state=SEED)))
    elif strategy == "class_weight":
        params = est.get_params()
        if "class_weight" in params:
            est.set_params(class_weight="balanced")
        elif "scale_pos_weight" in params:
            est.set_params(scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    steps.append(("clf", est))
    return Pipeline(steps)


def make_estimator(name, params):
    if name == "lightgbm":
        return LGBMClassifier(n_jobs=-1, random_state=SEED, verbosity=-1, **params)
    return XGBClassifier(
        n_jobs=-1, random_state=SEED, tree_method="hist", eval_metric="aucpr", **params
    )


def report(model, Xe, ye, threshold):
    proba = model.predict_proba(Xe)[:, 1]
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        ye, pred, labels=[1], average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(ye, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(ye, proba)),
        "roc_auc": float(roc_auc_score(ye, proba)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def pick_threshold(ye, proba, min_precision):
    prec, rec, thr = precision_recall_curve(ye, proba)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(prec >= min_precision)[0]
    if ok.size:
        return float(thr[ok[0]])
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(thr[int(np.argmax(f1))])


# --- selection subsample (winner refit on full train) ---
rng = np.random.default_rng(SEED)
neg = np.where(y_tr == 0)[0]
pos = np.where(y_tr == 1)[0]
keep = np.sort(np.concatenate([pos, rng.choice(neg, min(SELECT_SAMPLE - pos.size, neg.size), replace=False)]))
Xs, ys = X_tr[keep], y_tr[keep]

SPACES = {
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 1000, step=100),
        "num_leaves": t.suggest_int("num_leaves", 16, 256, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 100),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-3, 5.0, log=True),
    },
}
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(name):
    def objective(trial):
        params = SPACES[name](trial)
        strat = trial.suggest_categorical("imbalance_strategy", list(STRATEGIES))
        sc = []
        for tri, vai in cv.split(Xs, ys):
            pipe = build_pipeline(make_estimator(name, params), strat)
            pipe.fit(Xs[tri], ys[tri])
            ap = average_precision_score(ys[vai], pipe.predict_proba(Xs[vai])[:, 1])
            sc.append(0.0 if np.isnan(ap) else ap)
        return float(np.mean(sc))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    bp = dict(study.best_params)
    strat = bp.pop("imbalance_strategy")
    print(f"  {name}: CV PR-AUC {study.best_value:.4f}  strat={strat}")
    return {"params": bp, "strategy": strat, "cv_pr_auc": float(study.best_value)}


print("tuning...")
tuned = {n: tune(n) for n in ("lightgbm", "xgboost")}
best_name = max(tuned, key=lambda n: tuned[n]["cv_pr_auc"])
best = tuned[best_name]


def base():
    return build_pipeline(make_estimator(best_name, best["params"]), best["strategy"])


print("threshold + calibration...")
oof = cross_val_predict(base(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
threshold = pick_threshold(y_tr, oof, MIN_PRECISION)

results = {}
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
results["baseline_most_frequent"] = report(dummy, X_te, y_te, 0.5)
lr = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"), "none").fit(X_tr, y_tr)
results["baseline_logreg"] = report(lr, X_te, y_te, 0.5)

single = base().fit(X_tr, y_tr)
results["tuned_single"] = report(single, X_te, y_te, threshold)

calib = CalibratedClassifierCV(base(), method="isotonic", cv=cv).fit(X_tr, y_tr)
results["tuned_calibrated"] = report(calib, X_te, y_te, threshold)
brier = float(brier_score_loss(y_te, calib.predict_proba(X_te)[:, 1]))

print("stacking...")
members = [
    (n, build_pipeline(make_estimator(n, tuned[n]["params"]), tuned[n]["strategy"]))
    for n in ("lightgbm", "xgboost")
]
stack = StackingClassifier(
    estimators=members,
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    stack_method="predict_proba",
    cv=StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED),
    n_jobs=1,
)
stack.fit(X_tr, y_tr)
stack_oof = cross_val_predict(
    StackingClassifier(
        estimators=members,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
    ),
    X_tr, y_tr, cv=3, method="predict_proba",
)[:, 1]
stack_threshold = pick_threshold(y_tr, stack_oof, MIN_PRECISION)
results["ensemble_stack"] = report(stack, X_te, y_te, stack_threshold)

payload = {
    "config": {"seed": SEED, "cv_folds": CV, "n_trials": N_TRIALS, "min_precision": MIN_PRECISION},
    "prevalence": {"test_positive_count": int(y_te.sum()), "test_size": int(len(y_te))},
    "tuned": tuned,
    "selected": best_name,
    "threshold": threshold,
    "stack_threshold": stack_threshold,
    "calibrated_brier_score": brier,
    "results": results,
}
(OUT / "metrics.json").write_text(json.dumps(payload, indent=2))

print("\n=== TEST-SPLIT RESULTS ===")
for k, v in results.items():
    print(f"{k:<22} PR-AUC {v['pr_auc']:.4f}  ROC {v['roc_auc']:.4f}  "
          f"P {v['precision']:.4f}  R {v['recall']:.4f}  F1 {v['f1']:.4f}  cm {v['confusion']}")
print("\nwrote", OUT / "metrics.json")
