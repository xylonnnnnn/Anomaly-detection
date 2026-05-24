from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED_COLUMNS = {"value"}


def validate_series(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Файл пустой: загрузите CSV с числовым рядом.")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("Колонка value должна быть числовой.")


def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    validate_series(df)
    prepared = df.copy()
    if "timestamp" not in prepared.columns:
        prepared["timestamp"] = pd.RangeIndex(start=0, stop=len(prepared), step=1)
    else:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="ignore")
    prepared = prepared.dropna(subset=["value"]).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("После очистки не осталось числовых значений.")
    return prepared


def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    prepared = prepare_series(df)
    values = prepared["value"].astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0:
        prepared["score"] = 0.0
        prepared["is_anomaly"] = False
        return prepared
    z_scores = (values - mean) / std
    prepared["score"] = z_scores.abs().round(4)
    prepared["is_anomaly"] = prepared["score"] > threshold
    return prepared


def detect_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    prepared = prepare_series(df)
    values = prepared["value"].astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    distance = np.maximum(lower - values, values - upper)
    prepared["score"] = np.maximum(distance, 0).round(4)
    prepared["is_anomaly"] = (values < lower) | (values > upper)
    return prepared


def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    prepared = prepare_series(df)
    values = prepared[["value"]].astype(float)
    contamination = min(max(contamination, 0.001), 0.49)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(values)
    prepared["score"] = (-model.decision_function(values)).round(4)
    prepared["is_anomaly"] = preds == -1
    return prepared


def detect_anomalies(df: pd.DataFrame, method: str = "Z-score", threshold: float = 3.0, contamination: float = 0.05) -> pd.DataFrame:
    if method == "Z-score":
        return detect_zscore(df, threshold=threshold)
    if method == "IQR":
        return detect_iqr(df, multiplier=threshold)
    if method == "Isolation Forest":
        return detect_isolation_forest(df, contamination=contamination)
    raise ValueError(f"Неизвестный метод: {method}")


def calculate_metrics(result: pd.DataFrame) -> dict[str, float | int]:
    anomalies = int(result["is_anomaly"].sum())
    total = int(len(result))
    return {
        "total_points": total,
        "anomaly_count": anomalies,
        "anomaly_share_pct": round(100 * anomalies / total, 2) if total else 0,
        "mean_value": round(float(result["value"].mean()), 3) if total else 0,
        "std_value": round(float(result["value"].std(ddof=0)), 3) if total else 0,
    }
