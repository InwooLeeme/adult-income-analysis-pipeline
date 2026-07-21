"""
파일명: loading.py
설명: 파이프라인 1단계 — Pandas와 Polars로 동일 CSV를 각각 읽어 정합성을 비교한다.
기능:
    - load_and_compare(): 0단계(벤치마크)가 이미 읽어 둔 DataFrame과 로딩 시간을 받아,
      두 라이브러리의 결과(행/열 수, 결측치 수)가 일치하는지 비교한다.
      같은 파일을 다시 읽지 않도록, 벤치마크 단계의 로딩 결과를 그대로 재사용한다.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from .config import COLUMNS_OF_INTEREST
from .reporting import Reporter
from .results import LoadResult


def load_and_compare(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
    load_seconds: tuple[float, float],
    reporter: Reporter,
) -> tuple[pd.DataFrame, LoadResult]:
    """0단계에서 이미 읽어 둔 DataFrame으로 두 라이브러리의 로딩 결과를 비교한다.

    두 라이브러리가 동일한 데이터를 읽어내는지(행 수·결측치 수) 확인한 뒤,
    이후 단계에서 사용할 Pandas DataFrame과 비교 결과를 반환한다.
    """
    reporter.section("1. 데이터 로딩 (Pandas vs Polars)")

    pandas_seconds, polars_seconds = load_seconds

    # 속도뿐 아니라 "읽어낸 결과"가 같은지 확인한다.
    result = LoadResult(
        pandas_seconds=pandas_seconds,
        polars_seconds=polars_seconds,
        pandas_shape=pandas_df.shape,
        polars_shape=(polars_df.height, polars_df.width),
        pandas_nulls=int(pandas_df[COLUMNS_OF_INTEREST].isna().sum().sum()),
        polars_nulls=int(sum(polars_df.select(COLUMNS_OF_INTEREST).null_count().row(0))),
    )

    reporter.log("📊 [로딩 속도 비교 (3회 평균)]")
    reporter.log(f"- Pandas: {result.pandas_seconds:.4f}초")
    reporter.log(f"- Polars: {result.polars_seconds:.4f}초")
    reporter.log(f"-> {result.faster_engine}가 약 {result.speed_ratio:.1f}배 빠릅니다.")

    reporter.log("\n🔍 [로딩 결과 비교]")
    reporter.log(f"- Pandas shape: {result.pandas_shape[0]:,} rows x {result.pandas_shape[1]} cols")
    reporter.log(f"- Polars shape: {result.polars_shape[0]:,} rows x {result.polars_shape[1]} cols")
    reporter.log(f"- 동일 shape 여부: {result.same_shape}")
    reporter.log(f"- 관심 컬럼 결측치 — Pandas: {result.pandas_nulls:,} / Polars: {result.polars_nulls:,}")
    reporter.log(f"- 결측치 집계 일치 여부: {result.same_nulls}")

    return pandas_df, result
