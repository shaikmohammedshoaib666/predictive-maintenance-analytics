"""Random Forest RUL (Remaining Useful Life) predictor."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import config


class RULPredictor:
    """Predict remaining useful life (days until failure) using Random Forest."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state or config.RUL_RANDOM_STATE
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_columns: list[str] = []
        self.is_fitted = False
        self.metrics: dict[str, float] = {}
        self.feature_importance: pd.DataFrame | None = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling features and degradation indicators per machine."""
        result = df.copy()
        sensor_cols = [c for c in config.SENSOR_COLUMNS if c in result.columns]
        if not sensor_cols:
            sensor_cols = result.select_dtypes(include="number").columns.tolist()
            sensor_cols = [c for c in sensor_cols if c != "failure_within_days"]

        group_col = "machine_id" if "machine_id" in result.columns else None
        window = config.ROLLING_WINDOW

        for col in sensor_cols:
            if group_col:
                result[f"{col}_roll_mean"] = result.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=min(window, len(x)), min_periods=1).mean()
                )
                result[f"{col}_roll_std"] = result.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=min(window, len(x)), min_periods=1).std().fillna(0)
                )
                result[f"{col}_trend"] = result.groupby(group_col)[col].transform(
                    lambda x: x.diff().fillna(0)
                )
            else:
                result[f"{col}_roll_mean"] = result[col].rolling(window, min_periods=1).mean()
                result[f"{col}_roll_std"] = result[col].rolling(window, min_periods=1).std().fillna(0)
                result[f"{col}_trend"] = result[col].diff().fillna(0)

        return result

    def _get_feature_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        engineered = self._engineer_features(df)
        exclude = {"failure_within_days", "timestamp", "machine_id"}
        feature_cols = [
            c for c in engineered.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(engineered[c])
        ]
        self.feature_columns = feature_cols
        X = engineered[feature_cols].fillna(0)
        y = engineered["failure_within_days"] if "failure_within_days" in engineered.columns else None
        return X, y

    def _synthetic_rul(self, df: pd.DataFrame) -> pd.Series:
        """Generate synthetic RUL labels from sensor degradation when target is missing."""
        sensor_cols = [c for c in config.SENSOR_COLUMNS if c in df.columns]
        if not sensor_cols:
            return pd.Series(np.random.uniform(1, 30, len(df)), index=df.index)

        degradation = df[sensor_cols].apply(
            lambda row: sum(row[c] / (df[c].max() + 1e-6) for c in sensor_cols), axis=1
        )
        rul = (1 - degradation / (degradation.max() + 1e-6)) * 30 + 1
        return rul.clip(1, 30)

    def fit(self, df: pd.DataFrame) -> "RULPredictor":
        X, y = self._get_feature_matrix(df)
        if y is None or y.isna().all():
            y = self._synthetic_rul(df)

        mask = y.notna()
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=0.2, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            "r2": round(float(r2_score(y_test, y_pred)), 3),
        }
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return self

    def predict_latest_per_machine(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Predict RUL for the latest reading of each machine."""
        if not self.is_fitted:
            self.fit(df)

        engineered = self._engineer_features(df)
        predictions = []

        if "machine_id" in df.columns:
            for machine in df["machine_id"].unique():
                mdf = engineered[engineered["machine_id"] == machine]
                latest = mdf.iloc[[-1]]
                X = latest[self.feature_columns].fillna(0)
                rul = float(self.model.predict(X)[0])
                rul = max(1, round(rul))
                predictions.append({
                    "machine_id": str(machine),
                    "predicted_rul_days": rul,
                    "message": f"{machine} will fail in {rul} days",
                    "risk_level": "High" if rul <= 7 else ("Medium" if rul <= 14 else "Low"),
                })
        else:
            latest = engineered.iloc[[-1]]
            X = latest[self.feature_columns].fillna(0)
            rul = max(1, round(float(self.model.predict(X)[0])))
            predictions.append({
                "machine_id": "All",
                "predicted_rul_days": rul,
                "message": f"Equipment will fail in approximately {rul} days",
                "risk_level": "High" if rul <= 7 else ("Medium" if rul <= 14 else "Low"),
            })

        return sorted(predictions, key=lambda x: x["predicted_rul_days"])
