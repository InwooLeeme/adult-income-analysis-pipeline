"""전처리·포매팅 등 순수 함수에 대한 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.formatting import (
    format_p_expression,
    format_p_value,
    frame_to_markdown,
    interpret_p_value,
    markdown_table,
    series_to_markdown,
)
from src.preprocess import _parse_years, preprocess
from src.reporting import Reporter


# ──────────────────────────────────────────────────────────────────────────────
# p-value 포매팅 및 해석
# ──────────────────────────────────────────────────────────────────────────────
def test_format_p_value_marks_underflow_instead_of_printing_zero():
    """p-value가 부동소수점 하한 아래로 내려가면 0이 아니라 부등호로 표기해야 한다."""
    assert format_p_value(0.0) == "< 1e-308"


def test_format_p_value_keeps_significant_digits():
    assert format_p_value(1.3344e-09) == "1.334e-09"
    assert format_p_value(0.05) == "0.05"


def test_format_p_expression_avoids_equals_before_inequality():
    """`p = < 1e-308` 같은 어색한 표기가 나오면 안 된다."""
    assert format_p_expression(0.0) == "p < 1e-308"
    assert format_p_expression(0.001) == "p = 0.001"


def test_interpret_p_value_rejects_null_hypothesis_when_significant():
    result = interpret_p_value(0.001)

    assert "기각" in result
    assert "유의미함" in result


def test_interpret_p_value_accepts_null_hypothesis_when_not_significant():
    result = interpret_p_value(0.2)

    assert "채택" in result
    assert "유의미하지 않음" in result


def test_interpret_p_value_respects_custom_alpha():
    """같은 p-value라도 유의수준을 좁히면 결론이 뒤집혀야 한다."""
    assert "기각" in interpret_p_value(0.03, alpha=0.05)
    assert "채택" in interpret_p_value(0.03, alpha=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# 경력 문자열 파싱
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Less than 1 year", 0.5),
        ("More than 50 years", 51.0),
        ("12", 12.0),
        (7, 7.0),
    ],
)
def test_parse_years_converts_survey_notation_to_numbers(raw, expected):
    assert _parse_years(raw) == expected


@pytest.mark.parametrize("raw", ["", "모름", "N/A", None])
def test_parse_years_returns_none_for_unparsable_values(raw):
    assert _parse_years(raw) is None


def test_parse_years_passes_nan_through():
    """NaN은 float()로 변환되므로 None이 아닌 NaN이 나온다.

    preprocess()가 파싱보다 먼저 dropna()를 수행하므로 실제 파이프라인에서는
    NaN이 여기까지 오지 않고, 남더라도 뒤이은 dropna()가 걸러낸다.
    """
    assert np.isnan(_parse_years(np.nan))


# ──────────────────────────────────────────────────────────────────────────────
# Reporter
# ──────────────────────────────────────────────────────────────────────────────
def test_reporter_prints_messages_and_section_headers(capsys):
    reporter = Reporter()
    reporter.log("첫 줄")
    reporter.section("제목")

    printed = capsys.readouterr().out
    assert "첫 줄" in printed
    assert "===== 제목 =====" in printed


# ──────────────────────────────────────────────────────────────────────────────
# 마크다운 표 렌더링 (tabulate 없이 직접 조립)
# ──────────────────────────────────────────────────────────────────────────────
def test_markdown_table_emits_github_flavored_table():
    table = markdown_table(["이름", "값"], [["A", "1"], ["B", "2"]])

    assert table.splitlines() == [
        "| 이름 | 값 |",
        "| --- | --- |",
        "| A | 1 |",
        "| B | 2 |",
    ]


def test_frame_to_markdown_includes_index_column_and_formats_numbers():
    frame = pd.DataFrame({"연봉": [75417.456], "경력": [9.5]}, index=["mean"])

    table = frame_to_markdown(frame, "통계량")

    assert table.splitlines() == [
        "| 통계량 | 연봉 | 경력 |",
        "| --- | --- | --- |",
        "| mean | 75,417.46 | 9.50 |",
    ]


def test_frame_to_markdown_respects_custom_number_format():
    frame = pd.DataFrame({"YearsCodePro": [0.4213]}, index=["ConvertedCompYearly"])

    table = frame_to_markdown(frame, "변수", number_format=".3f")

    assert "| ConvertedCompYearly | 0.421 |" in table


def test_series_to_markdown_emits_two_column_table():
    series = pd.Series({"Favorable": 5966.0, "Unsure": 283.0})

    table = series_to_markdown(series, "AI 태도", "응답자")

    assert "| AI 태도 | 응답자 |" in table
    assert "| Favorable | 5,966 |" in table


def test_markdown_helpers_leave_non_numeric_cells_untouched():
    frame = pd.DataFrame({"직군": ["Developer, full-stack"]}, index=["Very favorable"])

    assert "Developer, full-stack" in frame_to_markdown(frame, "AI 태도")


# ──────────────────────────────────────────────────────────────────────────────
# 전처리
# ──────────────────────────────────────────────────────────────────────────────
def _row(salary, years_pro, sentiment="Favorable"):
    """전처리 테스트용 단일 응답 행을 만든다. (config.COLUMNS_OF_INTEREST 전체를 채운다)"""
    return {
        "ConvertedCompYearly": salary,
        "YearsCodePro": years_pro,
        "YearsCode": "10",
        "AISent": sentiment,
        "Country": "United States of America",
        "DevType": "Developer, back-end",
        "EdLevel": "Bachelor's degree",
        "OrgSize": "20 to 99 employees",
        "RemoteWork": "Remote",
        "Age": "25-34 years old",
    }


def test_preprocess_removes_missing_duplicate_unparsable_and_outlier_rows():
    raw = pd.DataFrame(
        [
            _row(50_000, "5"),
            _row(50_000, "5"),  # 완전 중복 → 1건만 남아야 함
            {**_row(70_000, "3"), "AISent": np.nan},  # 결측치 → 제거
            _row(60_000, "Less than 1 year"),  # 문자열 경력 → 0.5로 변환
            _row(90_000, "모름"),  # 파싱 불가 경력 → 제거
            _row(5_000_000, "20"),  # 상위 1% 이상치 → 절단
        ]
    )

    cleaned, _ = preprocess(raw, Reporter())

    assert sorted(cleaned["ConvertedCompYearly"]) == [50_000, 60_000]
    assert cleaned.loc[cleaned["ConvertedCompYearly"] == 60_000, "YearsCodePro"].item() == 0.5
    # 인덱스가 0부터 재설정되어야 이후 단계에서 위치 기반 접근이 안전하다.
    assert list(cleaned.index) == [0, 1]


def test_preprocess_result_records_what_was_removed():
    """'무엇을 얼마나 지웠는지'가 결과 구조체에 남아야 보고서에 근거를 실을 수 있다."""
    raw = pd.DataFrame([_row(50_000, "5"), _row(50_000, "5"), {**_row(70_000, "3"), "AISent": np.nan}])

    _, result = preprocess(raw, Reporter())

    assert result.raw_rows == 3
    assert result.clean_rows == 1
    assert result.duplicate_rows == 1
    assert result.missing_by_column["AISent"] == 1
    assert result.retention_rate == pytest.approx(1 / 3)


def test_preprocess_keeps_only_columns_of_interest():
    """원본은 114개 컬럼이므로 관심 컬럼만 남는지 확인한다."""
    raw = pd.DataFrame([{**_row(50_000, "5"), "무관한컬럼": "버려야 함"}])

    cleaned, _ = preprocess(raw, Reporter())

    assert "무관한컬럼" not in cleaned.columns
