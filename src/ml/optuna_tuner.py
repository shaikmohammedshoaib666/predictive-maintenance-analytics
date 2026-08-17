"""Optuna hyperparameter tuning for PdM models (ported from Forge AutoML patterns)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

import config


def tune_rul_model(
    df: pd.DataFrame,
    target: str = "failure_within_days",
    n_trials: int = 20,
) -> dict[str, Any]:
    """Optuna bake-off: RandomForest vs GradientBoosting for RUL regression."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError("Optuna not installed. pip install optuna") from exc

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")

    work = df.copy()
    y = pd.to_numeric(work[target], errors="coerce")
    feature_cols = [
        c
        for c in work.columns
        if c != target
        and pd.api.types.is_numeric_dtype(work[c])
        and not str(c).lower().endswith("_bin")
    ]
    if not feature_cols:
        feature_cols = [c for c in config.SENSOR_COLUMNS if c in work.columns]
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if len(X) < 30:
        raise ValueError("Need at least 30 labeled rows for Optuna RUL tuning")

    def objective(trial: Any) -> float:
        model_name = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting"])
        if model_name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 250),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                random_state=config.RUL_RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                random_state=config.RUL_RANDOM_STATE,
            )
        scores = cross_val_score(
            model, X, y, cv=min(5, max(2, len(X) // 20)), scoring="neg_root_mean_squared_error"
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_params
    model_name = str(best.get("model", "RandomForest"))
    params = {k: v for k, v in best.items() if k != "model"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RUL_RANDOM_STATE
    )
    if model_name == "RandomForest":
        model = RandomForestRegressor(random_state=config.RUL_RANDOM_STATE, n_jobs=-1, **params)
    else:
        model = GradientBoostingRegressor(random_state=config.RUL_RANDOM_STATE, **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "best_model": model_name,
        "best_params": best,
        "optuna_best_value": round(float(study.best_value), 4),
        "n_trials": int(n_trials),
        "features": feature_cols,
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "target": target,
    }


def tune_anomaly_contamination(
    df: pd.DataFrame,
    n_trials: int = 12,
) -> dict[str, Any]:
    """Search IsolationForest contamination to maximize separation proxy (score spread)."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError("Optuna not installed. pip install optuna") from exc

    cols = [c for c in config.SENSOR_COLUMNS if c in df.columns]
    if len(cols) < 2:
        cols = list(df.select_dtypes(include=[np.number]).columns)[:6]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if len(X) < 40:
        raise ValueError("Need more rows for anomaly tuning")

    def objective(trial: Any) -> float:
        cont = trial.suggest_float("contamination", 0.02, 0.15)
        iso = IsolationForest(contamination=cont, random_state=42, n_estimators=100)
        scores = -iso.fit(X).score_samples(X)
        # Prefer contamination that yields clear high-score tail without collapsing
        thr = np.quantile(scores, 1 - cont)
        sep = float(scores[scores >= thr].mean() - scores[scores < thr].mean())
        return sep

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return {
        "best_contamination": round(float(study.best_params["contamination"]), 4),
        "optuna_best_value": round(float(study.best_value), 4),
        "n_trials": int(n_trials),
        "features": cols,
    }
