# Credit-card fraud detection

Detect fraudulent card transactions in the **ULB dataset** (284,807 transactions,
492 frauds, 0.172 % positive). The dataset is extremely imbalanced, so the work
here is about measuring the model honestly, not chasing accuracy.

What this repo does differently from a typical notebook version:

- **PR-AUC (average precision) is the headline metric.** At 0.17 % positives,
  accuracy and ROC-AUC are close to meaningless.
- **ROC-AUC and PR-AUC are computed from predicted probabilities**, never from
  hard 0/1 labels.
- **Resampling (SMOTE) runs inside each cross-validation fold**, not once on the
  whole training set. The validation fold keeps the real class ratio.
- **The decision threshold is chosen from the precision-recall curve** to meet a
  target precision, then the model's probabilities are **isotonic-calibrated**.
- Every result is reported against a **most-frequent baseline** and a
  **logistic-regression baseline**.

## Dataset

The dataset is **not** stored in Git (144 MB). Download it once from Kaggle:

```bash
# Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip        # -> creditcard.csv in the repo root
```

`creditcard.csv` has `Time`, `V1..V28` (PCA components), `Amount`, and the
`Class` target (1 = fraud).

## Layout

```
src/ccfraud/
  config.py     frozen Config + CostMatrix - every tunable in one place
  data.py       load_creditcard(), stratified_split()
  pipeline.py   build_pipeline(): RobustScaler -> resampler -> estimator (imblearn)
  select.py     select_model(): StratifiedKFold sweep ranked by mean CV PR-AUC
  evaluate.py   PR-AUC, ROC-AUC (from probabilities), pick_threshold, expected_cost
  train.py      end-to-end run -> output/metrics.json + PR/ROC curves
scripts/train.py  thin CLI shim
tests/            offline tests on synthetic data
```

## Use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                       # ~20 s, no dataset needed (synthetic data)
ruff check . && mypy src

ccfraud-train                    # full run: selection -> calibration -> test eval
ccfraud-train --fast             # small grid + subsample, for a quick check
ccfraud-train --tune --ensemble  # Optuna search + stacked ensemble (see below)
```

`ccfraud-train` writes `output/metrics.json`, `output/pr_curve.png`, and
`output/roc_curve.png`.

## Results and honest limitations

Held-out **20 % test split: 56,962 transactions, 98 frauds**. Seed 42.
Numbers are from `output/metrics.json` (run: `ccfraud-train`).

**Selected model: LightGBM + SMOTE.** Cross-validation PR-AUC (mean over 5
stratified folds, on the 80k selection subsample): **0.877 ± 0.037**. The next
six pairs (LightGBM/none, XGBoost variants, LightGBM/class-weight) all sit
within one standard deviation, so "LightGBM won" is real but the margin over
XGBoost is noise. Random forest and logistic regression trail
(CV PR-AUC ~0.86 and ~0.84).

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Frauds caught | False alarms |
|---|---|---|---|---|---|---|---|
| Baseline — predict "genuine" always | 0.002 | 0.500 | 0.000 | 0.000 | 0.000 | 0 / 98 | 0 |
| Baseline — logistic regression (balanced, thr 0.5) | 0.716 | 0.972 | 0.061 | 0.918 | 0.115 | 90 / 98 | 1,382 |
| **Selected — LightGBM + SMOTE (thr 0.88)** | **0.880** | 0.981 | **0.932** | **0.837** | **0.882** | 82 / 98 | 6 |
| Selected — isotonic-calibrated (same thr) | 0.881 | 0.987 | 0.971 | 0.684 | 0.802 | 67 / 98 | 2 |

- **PR-AUC 0.88 vs a 0.002 prevalence floor.** ROC-AUC (0.98) looks great but is
  a weak signal at this imbalance; PR-AUC is the number to watch.
- **The 0.90-precision target is met.** The operating threshold (0.88) is picked
  from out-of-fold probabilities on the training set to reach 90 % precision;
  on the held-out set it lands at **93.2 % precision / 83.7 % recall, F1 0.88** —
  6 false alarms and 16 missed frauds out of 98.
- This beats the original notebook's best (random forest, F1 0.83), and that
  notebook's "ROC-AUC" was computed from hard labels, not probabilities, so it
  was not a real ROC-AUC.
- **Calibration is kept for probability quality, not for this threshold.**
  Isotonic calibration gives an excellent Brier score (0.0004) and higher
  precision, but at the same threshold it cuts recall to 68 %. Use the
  uncalibrated model for the yes/no decision and the calibrated probabilities
  when ranking or costing alerts.

### Does a hyperparameter search or an ensemble help? No.

`ccfraud-train --tune --ensemble` runs an Optuna TPE search (60 trials each over
LightGBM and XGBoost hyperparameters + imbalance strategy) and a
`StackingClassifier` of the two. Run on Kaggle;
`results/tune_ensemble_metrics.json`:

| Model | Test PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Default LightGBM + SMOTE (above) | 0.880 | 0.981 | 0.932 | 0.837 | 0.882 |
| **Tuned XGBoost** (60 trials) | **0.881** | 0.969 | 0.865 | 0.847 | 0.856 |
| Tuned + isotonic-calibrated | 0.880 | 0.977 | 0.899 | 0.816 | 0.856 |
| Stacked ensemble (tuned LGBM + XGB) | 0.824 | 0.974 | 0.865 | 0.847 | 0.856 |

- **Tuning bought ~0.001 PR-AUC** — inside the noise band. The fixed
  hyperparameters were already near-optimal; the CV PR-AUC even dipped slightly.
- **The stack made it worse** (0.824). The two members' probability outputs are
  highly correlated, and a linear meta-model over them produced a degenerate,
  poorly-calibrated score.
- This is the point: on the ULB dataset **~0.88 PR-AUC is a property of the 30
  anonymised features, not of model effort**. Published "0.95+" results on this
  dataset come from leakage — most often SMOTE applied *before* the train/test
  split. Higher genuinely needs richer data (per-card velocity, merchant,
  device, time-of-day) and a sequence or graph model.

### Limitations

- **One dataset, one place, one period.** ULB is Sept 2013, European
  cardholders, two days. Nothing here says the model generalises to other
  issuers, regions, or years.
- **PCA features have no meaning.** `V1..V28` are anonymised components, so
  there is no feature-level explanation of a decision and no domain sanity
  check.
- **492 positives.** Every fraud-class metric has a wide confidence interval;
  treat third-decimal differences as noise. The reported CV standard deviation
  is the honest spread.
- **No temporal validation.** The split is random, not time-ordered, so it does
  not measure concept drift or a realistic "train on the past, score the
  future" setup.
- **The cost matrix is illustrative.** `CostMatrix` uses placeholder amounts;
  the chosen threshold is only as good as those numbers, which should come from
  the fraud-operations team.
- **Not a production scorer.** No streaming, no feature store, no monitoring,
  no model registry. This is an offline analysis.
