from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_series(
    n_points: int = 500,
    anomaly_count: int = 25,
    seed: int = 42,
    trend_strength: float = 0.03,
    seasonality_strength: float = 10.0,
    noise_std: float = 2.0,
) -> pd.DataFrame:
    """Generate synthetic time series with trend, seasonality, noise and injected anomalies."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_points)
    trend = trend_strength * x
    seasonality = seasonality_strength * np.sin(2 * np.pi * x / 50)
    noise = rng.normal(0, noise_std, n_points)
    baseline = 50 + trend + seasonality + noise

    labels = np.zeros(n_points, dtype=int)
    anomaly_count = min(max(anomaly_count, 0), max(n_points // 4, 1))
    anomaly_idx = rng.choice(n_points, size=anomaly_count, replace=False)
    anomaly_shift = rng.choice([-1, 1], size=anomaly_count) * rng.uniform(18, 35, size=anomaly_count)
    values = baseline.copy()
    values[anomaly_idx] += anomaly_shift
    labels[anomaly_idx] = 1

    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="D")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": np.round(values, 3),
            "is_anomaly_true": labels,
        }
    )


def save_sample(path: str | Path = "data/sample_timeseries.csv", n_points: int = 500, anomaly_count: int = 25, seed: int = 42) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_series(n_points=n_points, anomaly_count=anomaly_count, seed=seed)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved = save_sample()
    print(f"Sample dataset saved to {saved}")
