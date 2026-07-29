"""
Plotly Dash 대시보드용 데이터 준비와 차트 생성 유틸리티.

Dash 콜백은 얇게 유지하고, 테스트 가능한 데이터 처리 로직은 이 모듈에 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import (
    AI_SENTIMENT_ORDER,
    DEFAULT_DATA_PATH,
    MODEL_DIR,
    MULTIVARIATE_MODEL_NAME,
    TARGET_COLUMN,
)
from .preprocess import preprocess

EXPERIENCE_LABELS = ["0~5년", "5~10년", "10~15년", "15~25년", "25년 이상"]
FAVORABLE_SENTIMENTS = {"Favorable", "Very favorable"}


class _SilentReporter:
    """대시보드 로딩 중 파이프라인 로그가 콘솔을 채우지 않게 하는 no-op reporter."""

    def log(self, message: str) -> None:
        pass

    def section(self, title: str) -> None:
        pass


def load_dashboard_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """원본 설문 CSV를 읽고 기존 전처리 규칙을 적용한 대시보드용 DataFrame을 반환한다."""
    raw = pd.read_csv(data_path, na_values=["NA"])
    cleaned, _ = preprocess(raw, _SilentReporter())  # type: ignore[arg-type]
    return add_experience_groups(cleaned)


def add_experience_groups(df: pd.DataFrame) -> pd.DataFrame:
    """전문 경력(`YearsCodePro`)을 대시보드 필터용 구간 라벨로 변환한다."""
    enriched = df.copy()
    enriched["ExperienceGroup"] = pd.cut(
        enriched["YearsCodePro"],
        bins=[0, 5, 10, 15, 25, float("inf")],
        labels=EXPERIENCE_LABELS,
        right=False,
    ).astype(str)
    return enriched


def filter_dashboard_data(
    df: pd.DataFrame,
    *,
    countries: list[str] | None = None,
    experience_groups: list[str] | None = None,
    dev_types: list[str] | None = None,
    sentiments: list[str] | None = None,
) -> pd.DataFrame:
    """선택된 필터를 모두 만족하는 행만 반환한다."""
    filtered = df
    if countries:
        filtered = filtered[filtered["Country"].isin(countries)]
    if experience_groups:
        filtered = filtered[filtered["ExperienceGroup"].isin(experience_groups)]
    if dev_types:
        filtered = filtered[filtered["DevType"].isin(dev_types)]
    if sentiments:
        filtered = filtered[filtered["AISent"].isin(sentiments)]
    return filtered


def build_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """현재 필터 결과에서 카드에 표시할 핵심 지표를 계산한다."""
    if df.empty:
        return {
            "rows": 0,
            "median_salary": None,
            "mean_salary": None,
            "favorable_mean": None,
            "others_mean": None,
            "mean_gap": None,
        }

    is_favorable = df["AISent"].isin(FAVORABLE_SENTIMENTS)
    favorable_mean = _mean_or_none(df.loc[is_favorable, TARGET_COLUMN])
    others_mean = _mean_or_none(df.loc[~is_favorable, TARGET_COLUMN])
    mean_gap = (
        favorable_mean - others_mean if favorable_mean is not None and others_mean is not None else None
    )
    return {
        "rows": int(len(df)),
        "median_salary": float(df[TARGET_COLUMN].median()),
        "mean_salary": float(df[TARGET_COLUMN].mean()),
        "favorable_mean": favorable_mean,
        "others_mean": others_mean,
        "mean_gap": mean_gap,
    }


def make_salary_distribution_figure(df: pd.DataFrame) -> go.Figure:
    """AI 태도별 연봉 분포 박스플롯을 만든다."""
    if df.empty:
        return _empty_figure("선택한 조건에 해당하는 데이터가 없습니다")
    figure = px.box(
        df,
        x="AISent",
        y=TARGET_COLUMN,
        color="AISent",
        category_orders={"AISent": AI_SENTIMENT_ORDER},
        points=False,
        title="AI 태도별 연봉 분포",
        labels={"AISent": "AI 태도", TARGET_COLUMN: "연봉 (USD)"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    figure.update_layout(showlegend=False, margin={"l": 32, "r": 16, "t": 56, "b": 32})
    return figure


def make_sentiment_median_figure(df: pd.DataFrame) -> go.Figure:
    """AI 태도별 중위 연봉 막대그래프를 만든다."""
    if df.empty:
        return _empty_figure("선택한 조건에 해당하는 데이터가 없습니다")
    grouped = (
        df.groupby("AISent", observed=True)[TARGET_COLUMN]
        .median()
        .reindex(AI_SENTIMENT_ORDER)
        .dropna()
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x="AISent",
        y=TARGET_COLUMN,
        color="AISent",
        title="AI 태도별 중위 연봉",
        labels={"AISent": "AI 태도", TARGET_COLUMN: "중위 연봉 (USD)"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    figure.update_layout(showlegend=False, margin={"l": 32, "r": 16, "t": 56, "b": 32})
    return figure


def make_country_income_figure(df: pd.DataFrame, limit: int = 10) -> go.Figure:
    """표본 수 상위 국가의 AI 태도별 평균 연봉 비교 차트를 만든다."""
    if df.empty:
        return _empty_figure("선택한 조건에 해당하는 데이터가 없습니다")
    top_countries = df["Country"].value_counts().nlargest(limit).index
    grouped = (
        df[df["Country"].isin(top_countries)]
        .groupby(["Country", "AISent"], observed=True)[TARGET_COLUMN]
        .mean()
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x="Country",
        y=TARGET_COLUMN,
        color="AISent",
        barmode="group",
        category_orders={"AISent": AI_SENTIMENT_ORDER},
        title="국가별 AI 태도와 평균 연봉",
        labels={"Country": "국가", "AISent": "AI 태도", TARGET_COLUMN: "평균 연봉 (USD)"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    figure.update_layout(xaxis_tickangle=-25, margin={"l": 32, "r": 16, "t": 56, "b": 96})
    return figure


def make_coefficient_figure(
    model_path: Path = MODEL_DIR / MULTIVARIATE_MODEL_NAME,
    *,
    limit: int = 12,
) -> go.Figure:
    """저장된 다변량 Ridge 모델에서 영향도가 큰 회귀 계수 차트를 만든다."""
    if not model_path.exists():
        return _empty_figure("저장된 다변량 모델이 없습니다")

    pipeline = joblib.load(model_path)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_
    rows = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coef=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(limit)
        .sort_values("coefficient")
    )
    rows["feature"] = rows["feature"].map(_clean_feature_name)

    figure = px.bar(
        rows,
        x="coefficient",
        y="feature",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        title="회귀 계수 Top N",
        labels={"coefficient": "계수 (USD 근사)", "feature": "요인"},
    )
    figure.update_layout(margin={"l": 32, "r": 16, "t": 56, "b": 32})
    return figure


def dropdown_options(values: pd.Series, *, limit: int | None = None) -> list[dict[str, str]]:
    """Dash Dropdown에 전달할 label/value 목록을 만든다."""
    counts = values.dropna().value_counts()
    if limit is not None:
        counts = counts.head(limit)
    return [{"label": str(value), "value": str(value)} for value in counts.index]


def _mean_or_none(series: pd.Series) -> float | None:
    return None if series.empty else float(series.mean())


def _clean_feature_name(name: Any) -> str:
    text = str(name)
    for prefix in ("cat__", "num__"):
        text = text.removeprefix(prefix)
    return text.replace("_", " ")


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(margin={"l": 24, "r": 24, "t": 48, "b": 24})
    return figure
