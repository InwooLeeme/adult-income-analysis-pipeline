"""산출물을 만들어내는 파이프라인 단계와 보고서 렌더링에 대한 통합 테스트.

경로 상수를 tmp_path로 갈아끼워, 실제 프로젝트의 outputs/·report.md를 건드리지 않는다.
각 단계 모듈이 config의 경로 상수를 자신의 네임스페이스로 import해두므로, config가 아니라
파일을 실제로 쓰는 모듈(viz·deep_dive·benchmark·model·report) 쪽을 개별적으로 패치한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from src import benchmark, config, deep_dive, model, report, viz
from src.benchmark import run_benchmark
from src.deep_dive import run_deep_dive
from src.eda import run_eda
from src.loading import load_and_compare
from src.model import run_multivariate, train_model
from src.report import render_report, write_report
from src.reporting import Reporter
from src.results import BenchmarkResult, LoadResult, PreprocessResult
from src.stats import run_statistics
from src.viz import create_charts


def _as_embedded_json(html: str, text: str) -> bool:
    """Plotly가 HTML에 삽입한 JSON 안에 해당 문자열이 있는지 확인한다."""
    return json.dumps(text)[1:-1] in html


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """산출물을 실제로 쓰는 모듈들의 경로 상수를 임시 디렉터리로 교체한다."""
    charts = tmp_path / "outputs" / "charts"
    models = tmp_path / "outputs" / "models"
    charts.mkdir(parents=True)
    models.mkdir(parents=True)

    for target in (viz, deep_dive, benchmark, model, report):
        monkeypatch.setattr(target, "BASE_DIR", tmp_path)
    monkeypatch.setattr(viz, "CHART_DIR", charts)
    monkeypatch.setattr(deep_dive, "CHART_DIR", charts)
    monkeypatch.setattr(benchmark, "CHART_DIR", charts)
    monkeypatch.setattr(model, "MODEL_DIR", models)
    monkeypatch.setattr(report, "REPORT_FILE", tmp_path / "report.md")
    return tmp_path


@pytest.fixture
def dataset():
    """전처리를 통과한 상태의 합성 데이터셋.

    t-test와 train_test_split이 동작할 만큼의 표본과, 모든 AI 태도·다변량 통제
    변수(학력·조직규모·근무형태·연령)를 포함한다.
    """
    rows = 240
    rng = np.random.default_rng(42)
    countries = ["United States of America", "Germany", "India"]
    dev_types = ["Developer, full-stack", "Developer, back-end", "Data scientist"]
    ed_levels = ["Bachelor's degree", "Master's degree"]
    org_sizes = ["20 to 99 employees", "10,000 or more employees"]
    remote_options = ["Remote", "Hybrid", "In-person"]
    ages = ["25-34 years old", "35-44 years old"]

    return pd.DataFrame(
        {
            "ConvertedCompYearly": rng.integers(30_000, 200_000, rows).astype(float),
            "YearsCodePro": rng.integers(1, 30, rows).astype(float),
            "YearsCode": rng.integers(1, 35, rows).astype(float),
            "AISent": [config.AI_SENTIMENT_ORDER[i % len(config.AI_SENTIMENT_ORDER)] for i in range(rows)],
            "Country": [countries[i % len(countries)] for i in range(rows)],
            "DevType": [dev_types[i % len(dev_types)] for i in range(rows)],
            "EdLevel": [ed_levels[i % len(ed_levels)] for i in range(rows)],
            "OrgSize": [org_sizes[i % len(org_sizes)] for i in range(rows)],
            "RemoteWork": [remote_options[i % len(remote_options)] for i in range(rows)],
            "Age": [ages[i % len(ages)] for i in range(rows)],
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# 0~1. 로딩 (Pandas vs Polars)
# ──────────────────────────────────────────────────────────────────────────────
def _write_survey_csv(path: Path) -> Path:
    """결측치가 문자열 "NA"로 적힌 소형 설문 CSV를 만든다. (NA 2개)"""
    path.write_text(
        "ConvertedCompYearly,YearsCodePro,YearsCode,AISent,Country,DevType,EdLevel,"
        "OrgSize,RemoteWork,Age\n"
        "100000,5,8,Favorable,United States of America,Backend developer,"
        "Bachelor's degree,20 to 99 employees,Remote,25-34 years old\n"
        "NA,3,5,Unfavorable,Germany,Frontend developer,Master's degree,"
        '"10,000 or more employees",Hybrid,35-44 years old\n'
        "80000,NA,10,Very favorable,India,Backend developer,Bachelor's degree,"
        "20 to 99 employees,In-person,25-34 years old\n",
        encoding="utf-8",
    )
    return path


def test_run_benchmark_reads_NA_string_as_missing_in_both_engines(tmp_path):
    """회귀 테스트: 이 설문 CSV는 결측치를 문자열 "NA"로 표기한다.

    null_values를 지정하지 않으면 Polars가 "NA"를 문자열로 읽어 결측치가 0으로 집계되고
    숫자 컬럼까지 String이 된다. 0단계(벤치마크)가 읽어낸 두 엔진의 결과가 일치해야 한다.
    """
    csv_path = _write_survey_csv(tmp_path / "survey.csv")

    result, df_pd, df_pl = run_benchmark(csv_path, Reporter())

    pandas_nulls = int(df_pd[config.COLUMNS_OF_INTEREST].isna().sum().sum())
    polars_nulls = int(sum(df_pl.select(config.COLUMNS_OF_INTEREST).null_count().row(0)))
    assert pandas_nulls == polars_nulls == 2
    assert df_pd.shape == (3, 10)
    assert (df_pl.height, df_pl.width) == (3, 10)
    assert result.load[0] >= 0.0 and result.load[1] >= 0.0


def test_load_and_compare_compares_provided_frames_without_reading_files():
    """1단계는 0단계가 이미 읽어 둔 결과를 비교만 하고, 파일을 다시 읽지 않는다."""
    base = {column: ["x", "y"] for column in config.COLUMNS_OF_INTEREST}
    base["AISent"] = ["Favorable", None]
    pandas_df = pd.DataFrame(base)
    polars_df = pl.DataFrame(base)

    loaded, result = load_and_compare(pandas_df, polars_df, (0.9, 0.05), Reporter())

    assert loaded is pandas_df
    assert result.pandas_seconds == 0.9
    assert result.polars_seconds == 0.05
    assert result.pandas_nulls == result.polars_nulls == 1
    assert result.same_nulls
    assert result.same_shape


def test_load_result_speed_ratio_is_orientation_independent():
    """어느 쪽이 빠르든 비율은 항상 1 이상이어야 한다."""
    polars_faster = LoadResult(1.0, 0.25, (1, 1), (1, 1), 0, 0)
    pandas_faster = LoadResult(0.25, 1.0, (1, 1), (1, 1), 0, 0)

    assert polars_faster.faster_engine == "Polars"
    assert polars_faster.speed_ratio == pytest.approx(4.0)
    assert pandas_faster.faster_engine == "Pandas"
    assert pandas_faster.speed_ratio == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EDA / 5. 통계 분석
# ──────────────────────────────────────────────────────────────────────────────
def test_run_eda_returns_descriptive_statistics(dataset):
    result = run_eda(dataset, Reporter())

    # 채점 기준이 요구하는 평균·표준편차·분위수가 모두 있어야 한다.
    for label in ("mean", "std", "25%", "50%", "75%"):
        assert label in result.describe.index
    assert result.total_rows == len(dataset)
    assert result.sentiment_counts.sum() == len(dataset)
    assert result.country_count == 3


def test_run_statistics_returns_correlation_matrix_and_ttest(dataset):
    result = run_statistics(dataset, Reporter())

    # 상관계수 행렬에 수치형 변수(연봉·경력)가 모두 포함되어야 한다.
    assert list(result.correlation_matrix.columns) == [
        "ConvertedCompYearly",
        "YearsCodePro",
        "YearsCode",
    ]
    assert result.favorable_size + result.others_size == len(dataset)
    assert np.isfinite(result.t_statistic)
    assert 0.0 <= result.p_value <= 1.0
    assert result.is_significant == (result.p_value < config.ALPHA)


def test_stats_result_mean_gap_matches_group_means(dataset):
    result = run_statistics(dataset, Reporter())

    assert result.mean_gap == pytest.approx(result.favorable_mean - result.others_mean)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 시각화
# ──────────────────────────────────────────────────────────────────────────────
def test_create_charts_writes_static_and_interactive_files(dataset, sandbox):
    create_charts(dataset, Reporter())

    static_chart = sandbox / "outputs" / "charts" / config.BOXPLOT_NAME
    interactive_chart = sandbox / "outputs" / "charts" / config.INTERACTIVE_NAME
    assert static_chart.stat().st_size > 0
    assert interactive_chart.stat().st_size > 0

    # 인터랙티브 차트에는 제목과 축 레이블이 들어가야 한다.
    # Plotly는 HTML 안의 JSON에 한글을 \uXXXX로 이스케이프해 넣으므로 같은 형태로 비교한다.
    html = interactive_chart.read_text(encoding="utf-8")
    assert _as_embedded_json(html, "주요 국가별 AI 도구 인식에 따른 평균 연봉 비교")
    assert _as_embedded_json(html, "평균 연봉 (USD)")


# ──────────────────────────────────────────────────────────────────────────────
# 6. ML Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def test_train_model_returns_fitted_pipeline_and_saves_joblib(dataset, sandbox):
    pipeline, result = train_model(dataset, Reporter())

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "regressor"]
    assert (sandbox / "outputs" / "models" / config.MODEL_NAME).stat().st_size > 0
    assert result.train_rows + result.test_rows == len(dataset)
    assert all(np.isfinite([result.r2, result.rmse, result.mae, result.selected_alpha]))


def test_train_model_pipeline_predicts_from_raw_unencoded_input(dataset, sandbox):
    """전처리가 Pipeline 안에 들어있어야 원본 형태의 입력을 그대로 예측할 수 있다."""
    pipeline, _ = train_model(dataset, Reporter())

    prediction = pipeline.predict(
        pd.DataFrame(
            [
                {
                    "YearsCodePro": 10.0,
                    "YearsCode": 14.0,
                    "AISent": "Favorable",
                    "Country": "Germany",
                    "DevType": "Data scientist",
                }
            ]
        )
    )

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_train_model_handles_unseen_category_without_crashing(dataset, sandbox):
    """OneHotEncoder(handle_unknown="ignore") 설정이 살아있는지 확인한다."""
    pipeline, _ = train_model(dataset, Reporter())

    prediction = pipeline.predict(
        pd.DataFrame(
            [
                {
                    "YearsCodePro": 5.0,
                    "YearsCode": 7.0,
                    "AISent": "Favorable",
                    "Country": "존재하지 않는 국가",  # 학습에 없던 범주
                    "DevType": "존재하지 않는 직군",
                }
            ]
        )
    )

    assert np.isfinite(prediction[0])


def test_run_multivariate_adds_more_features_and_saves_second_model(dataset, sandbox):
    """확장 모델은 기본 모델보다 변수가 많고, 별도 joblib 파일로 저장되어야 한다."""
    _, model_result = train_model(dataset, Reporter())

    result = run_multivariate(dataset, model_result.r2, Reporter())

    assert result.n_enriched_features > result.n_baseline_features
    assert (sandbox / "outputs" / "models" / config.MULTIVARIATE_MODEL_NAME).stat().st_size > 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. 심층 분석
# ──────────────────────────────────────────────────────────────────────────────
def test_run_deep_dive_returns_markdown_for_every_control(dataset, sandbox):
    markdown = run_deep_dive(dataset, Reporter())

    for heading in ("1차 관찰", "통제 ① 국가", "통제 ② 경력 구간", "통제 ③ 직군 구성", "최종 결론"):
        assert f"### {heading}" in markdown or heading in markdown
    assert (sandbox / "outputs" / "charts" / config.DEEP_DIVE_CHART_NAME).stat().st_size > 0


def test_run_deep_dive_renders_tables_not_bullet_dumps(dataset, sandbox):
    """심층 분석 결과도 보고서 형식(표)으로 나와야 한다."""
    markdown = run_deep_dive(dataset, Reporter())

    assert "| 경력 구간 | Very favorable (USD) | Very unfavorable (USD) | 우위 |" in markdown
    assert "| AI 태도 | 1위 직군 |" in markdown


def test_run_deep_dive_does_not_mutate_caller_dataframe(dataset, sandbox):
    """경력 구간 컬럼은 내부 계산용이므로 원본에 남으면 안 된다."""
    original_columns = list(dataset.columns)

    run_deep_dive(dataset, Reporter())

    assert list(dataset.columns) == original_columns


# ──────────────────────────────────────────────────────────────────────────────
# 8. 보고서 렌더링
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def report_markdown(dataset, sandbox):
    """실제 분석 단계 결과로 보고서 전문을 렌더링한다."""
    reporter = Reporter()
    _, model_result = train_model(dataset, reporter)
    multivariate_result = run_multivariate(dataset, model_result.r2, reporter)
    benchmark_result = BenchmarkResult(
        runs=3,
        load=(0.9, 0.05),
        filter_op=(0.01, 0.01),
        group_op=(0.01, 0.01),
        chart_name=config.BENCHMARK_CHART_NAME,
    )

    return render_report(
        data_path=Path("data/results.csv"),
        load=LoadResult(0.9, 0.05, (65437, 114), (65437, 114), 129239, 129239),
        prep=PreprocessResult(
            raw_rows=65437,
            clean_rows=12194,
            missing_by_column=pd.Series({"ConvertedCompYearly": 42002, "AISent": 19564}),
            duplicate_rows=9731,
            outlier_threshold=373755.0,
        ),
        eda=run_eda(dataset, reporter),
        stats_result=run_statistics(dataset, reporter),
        model=model_result,
        multivariate=multivariate_result,
        deep_dive_markdown=run_deep_dive(dataset, reporter),
        benchmark=benchmark_result,
    )


def test_report_has_numbered_sections_in_order(report_markdown):
    headings = [line for line in report_markdown.splitlines() if line.startswith("## ")]

    assert headings == [
        "## 1. 분석 개요",
        "## 2. 데이터 준비",
        "## 3. 탐색적 데이터 분석 (EDA)",
        "## 4. 시각화",
        "## 5. 통계 분석",
        "## 6. 머신러닝 파이프라인",
        "## 7. 핵심 인사이트",
        "## 8. 부록 — Pandas vs Polars 성능 벤치마크",
    ]


def test_report_states_hypothesis_and_verdict(report_markdown):
    assert "귀무가설 H₀" in report_markdown
    assert "대립가설 H₁" in report_markdown
    assert "H₀ 기각" in report_markdown or "H₀ 채택" in report_markdown


def test_report_renders_results_as_tables_not_console_dump(report_markdown):
    """보고서는 표로 구성되어야 하고, 콘솔 로그가 섞여 들어가면 안 된다."""
    # 콘솔 로그 특유의 구분선·이모지 헤더가 남아 있으면 안 된다.
    assert "=====" not in report_markdown
    assert "📊" not in report_markdown

    # 각 장의 핵심 표가 있어야 한다.
    assert "| 항목 | Pandas | Polars |" in report_markdown
    assert "| 컬럼 | 결측치 | 비율 |" in report_markdown
    assert "| AI 태도 | 응답자 | 비율 |" in report_markdown
    assert "| 검정 통계량 | 값 |" in report_markdown
    assert "| 지표 | 값 | 의미 |" in report_markdown


def test_report_appendix_is_structured_table_not_raw_log(report_markdown):
    """부록(8장)은 실행 로그를 그대로 붙인 게 아니라 표로 구조화되어 있어야 한다."""
    assert "<details>" not in report_markdown
    assert "```text" not in report_markdown
    assert "## 8. 부록" in report_markdown
    assert "| 작업 | Pandas(초) | Polars(초) | Polars 배속 |" in report_markdown


def test_report_links_every_generated_artifact(report_markdown):
    for name in (
        config.BOXPLOT_NAME,
        config.INTERACTIVE_NAME,
        config.DEEP_DIVE_CHART_NAME,
        config.BENCHMARK_CHART_NAME,
    ):
        assert f"./outputs/charts/{name}" in report_markdown
    assert f"outputs/models/{config.MODEL_NAME}" in report_markdown


def test_report_documents_the_polars_null_values_pitfall(report_markdown):
    """로딩 비교에서 얻은 교훈이 보고서에 근거로 남아야 한다."""
    assert "null_values" in report_markdown


def test_write_report_saves_markdown_to_report_file(sandbox):
    write_report("# 제목\n\n본문")

    assert (sandbox / "report.md").read_text(encoding="utf-8") == "# 제목\n\n본문"
