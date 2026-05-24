import pandas as pd

from src.detector import calculate_metrics, detect_zscore


def test_zscore_detects_clear_outlier():
    df = pd.DataFrame({"value": [10, 11, 10, 12, 11, 200]})
    result = detect_zscore(df, threshold=1.5)
    assert result["is_anomaly"].sum() == 1
    assert bool(result.iloc[-1]["is_anomaly"]) is True


def test_metrics_are_calculated():
    result = pd.DataFrame({"value": [1, 2, 3], "is_anomaly": [False, True, False]})
    metrics = calculate_metrics(result)
    assert metrics["total_points"] == 3
    assert metrics["anomaly_count"] == 1
    assert metrics["anomaly_share_pct"] == 33.33
