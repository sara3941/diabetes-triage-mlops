# Changelog

All notable changes to the Diabetes Triage Microservice will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.2] - 2026-10-13

### Added
- Upgraded the regression pipeline from baseline OLS to `RidgeCV` with cross-validated $L_2$ regularization.
- Dynamic model selection via the `MODEL_VERSION` environment variable.
- Automated code formatting and linting compliance checks in CI/CD (`black`, `flake8`).

### Changed
- Refactored model artifact export in `src/train.py` to persist `selected_best_alpha`.

### Model Performance & Metric Deltas

Evaluation performed on the held-out test split (20%, `random_state=42`) using the standard diabetes progression target:

| Metric | v0.1 (LinearRegression) | v0.2 (RidgeCV) | Delta ($\Delta$) | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Test RMSE** | 54.55 | 53.45 | -1.10 | Improvement (Lower error) |
| **Train RMSE** | 53.48 | 53.98 | +0.50 | Regularized (Reduced overfitting) |
| **Test $R^2$** | 0.4500 | 0.4608 | +0.0108 | Improvement (Higher variance explained) |
| **Selected $\alpha$** | N/A | 44.98 | — | Optimal $L_2$ shrinkage parameter |

> **Summary**: The `v0.2` RidgeCV pipeline reduces test RMSE from 54.55 to 53.45 and increases test $R^2$ by +0.0108. The $L_2$ penalty mitigates collinearity among blood serum markers (`s1`–`s6`), yielding improved generalization on unseen patient data.

---

## [v0.1] - 2026-10-12

### Added
- Baseline Scikit-learn ordinary least squares (`LinearRegression`) progression model.
- FastAPI service with `/health` and `/predict` endpoints.
- Docker containerization exposing port `80` (mapped to `9696` locally).
- Automated test suite validating prediction bounds, health status, and input feature schemas.
- Baseline CI/CD pipeline deploying images to GitHub Container Registry (GHCR).