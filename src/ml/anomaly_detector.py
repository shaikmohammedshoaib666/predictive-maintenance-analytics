"""Isolation Forest anomaly detection for sensor readings."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config


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
        if not cols:
            cols = df.select_dtypes(include="number").columns.tolist()
        self.feature_columns = cols
        return df[cols].fillna(df[cols].median())

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = self._get_features(df)
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

    def summary(self, df: pd.DataFrame) -> dict:
        labels = self.predict(df)
        n_anomalies = int((labels == -1).sum())
        return {
            "total_records": len(df),
            "anomaly_count": n_anomalies,
            "anomaly_rate_pct": round(n_anomalies / max(len(df), 1) * 100, 2),
            "features_used": self.feature_columns,
        }
