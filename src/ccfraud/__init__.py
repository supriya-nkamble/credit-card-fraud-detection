"""Credit-card fraud detection on the ULB dataset.

Public modules:
    config    - frozen run configuration (paths, seed, CV, candidates, cost matrix)
    data      - load the CSV and make a stratified train/test split
    pipeline  - build a leakage-free RobustScaler -> resampler -> estimator pipeline
    evaluate  - PR-AUC, ROC-AUC on probabilities, threshold selection, expected cost
    select    - stratified K-fold model selection scored on PR-AUC
"""

__version__ = "0.1.0"
