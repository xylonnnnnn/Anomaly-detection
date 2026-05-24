from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.detector import calculate_metrics, detect_anomalies
from src.generate_data import generate_series, save_sample

st.set_page_config(page_title="Детекция аномалий", page_icon="📈", layout="wide")

DATA_PATH = Path("data/sample_timeseries.csv")


@st.cache_data
def load_sample() -> pd.DataFrame:
    if not DATA_PATH.exists():
        save_sample(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def build_chart(result: pd.DataFrame) -> go.Figure:
    normal = result[~result["is_anomaly"]]
    anomalies = result[result["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result["timestamp"],
            y=result["value"],
            mode="lines",
            name="Ряд",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=normal["timestamp"],
            y=normal["value"],
            mode="markers",
            name="Нормальные точки",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["value"],
            mode="markers",
            marker={"size": 11, "symbol": "x"},
            name="Аномалии",
        )
    )
    fig.update_layout(title="Исходный ряд и найденные аномалии", xaxis_title="Время", yaxis_title="Значение")
    return fig


st.title("📈 Модель обнаружения аномалий в числовых последовательностях")
st.caption("MVP для генерации временного ряда, поиска выбросов и визуализации результата.")

with st.sidebar:
    st.header("Данные")
    mode = st.radio("Источник", ["Демо-данные", "Загрузить CSV", "Сгенерировать новые"], index=0)
    method = st.selectbox("Метод", ["Z-score", "IQR", "Isolation Forest"], index=0)

    if method == "Z-score":
        threshold = st.slider("Порог Z-score", 1.0, 6.0, 3.0, 0.1)
        contamination = 0.05
    elif method == "IQR":
        threshold = st.slider("Множитель IQR", 0.5, 4.0, 1.5, 0.1)
        contamination = 0.05
    else:
        threshold = 3.0
        contamination = st.slider("Ожидаемая доля аномалий", 0.01, 0.30, 0.05, 0.01)

    st.markdown("**Формат CSV:** обязательная колонка `value`, опционально `timestamp`.")

try:
    if mode == "Демо-данные":
        df = load_sample()
    elif mode == "Загрузить CSV":
        uploaded = st.file_uploader("Загрузите CSV с рядом", type=["csv"])
        df = pd.read_csv(uploaded) if uploaded is not None else load_sample()
    else:
        n_points = st.sidebar.number_input("Количество точек", min_value=50, max_value=5000, value=500, step=50)
        anomaly_count = st.sidebar.number_input("Количество искусственных аномалий", min_value=0, max_value=int(n_points) // 4, value=25, step=1)
        seed = st.sidebar.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)
        df = generate_series(n_points=int(n_points), anomaly_count=int(anomaly_count), seed=int(seed))

    result = detect_anomalies(df, method=method, threshold=float(threshold), contamination=float(contamination))
    metrics = calculate_metrics(result)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего точек", metrics["total_points"])
    col2.metric("Найдено аномалий", metrics["anomaly_count"])
    col3.metric("Доля аномалий", f"{metrics['anomaly_share_pct']}%")
    col4.metric("Стандартное отклонение", metrics["std_value"])

    st.plotly_chart(build_chart(result), use_container_width=True)

    st.subheader("Результаты детекции")
    st.dataframe(result, use_container_width=True)

    csv = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать результат CSV", csv, "anomaly_detection_result.csv", "text/csv")

except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
