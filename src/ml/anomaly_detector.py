"""Isolation Forest anomaly detection for sensor readings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from src.industry_packs import OPTIONAL_IF_SENSORS

_SKIP_SUFFIXES = ("_bin", "_smooth")
SCORE_COL = "anomaly_score"
FLAG_COL = "is_anomaly"
_EXCLUDE_COLS = frozenset(
    {"failure_within_days", "predicted_rul_days", SCORE_COL, FLAG_COL, "is_anomaly"}
)


class AnomalyDetector:
    """Detect anomalous sensor readings using Isolation Forest."""

    def __init__(self, contamination: float | None = None, random_state: int = 42):
        self.contamination = contamination or config.ANOMALY_CONTAMINATION
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.feature_columns: list[str] = []
        self.is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in config.SENSOR_COLUMNS if c in df.columns]
        for extra in OPTIONAL_IF_SENSORS:
            if extra in df.columns and extra not in cols:
                cols.append(extra)
        if not cols:
            cols = [
                c
                for c in df.select_dtypes(include="number").columns.tolist()
                if not str(c).endswith(_SKIP_SUFFIXES) and c not in _EXCLUDE_COLS
            ]
        self.feature_columns = cols
        if not cols:
            return pd.DataFrame(index=df.index)
        return df[cols].fillna(df[cols].median())

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = self._get_features(df)
        if X.empty:
            raise ValueError("No numeric sensor columns to fit Isolation Forest.")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            self.fit(df)
        X = self._get_features(df)
        return self.model.predict(X)

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            self.fit(df)
        X = self._get_features(df)
        return -self.model.score_samples(X)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach Isolation Forest score (higher = more anomalous) and flag columns."""
        out = df.copy()
        out[SCORE_COL] = self.anomaly_scores(out)
        out[FLAG_COL] = self.predict(out) == -1
        return out

    def summary(self, df: pd.DataFrame) -> dict:
        labels = self.predict(df)
        n_anomalies = int((labels == -1).sum())
        scores = self.anomaly_scores(df)
        return {
            "total_records": len(df),
            "anomaly_count": n_anomalies,
            "anomaly_rate_pct": round(n_anomalies / max(len(df), 1) * 100, 2),
            "features_used": self.feature_columns,
            "score_mean": round(float(np.mean(scores)), 4) if len(scores) else 0.0,
            "score_max": round(float(np.max(scores)), 4) if len(scores) else 0.0,
        }
