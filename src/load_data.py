from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
import polars as pl


UCI_ADULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def is_url(source: str | Path) -> bool:
    parsed = urlparse(str(source))
    return parsed.scheme in {"http", "https"}


def ensure_data_source(source: str | Path) -> str | Path:
    if is_url(source):
        return str(source)

    data_path = Path(source)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Add the Adult Income CSV at this path."
        )
    return data_path


def ensure_data_file(path: str | Path) -> str | Path:
    return ensure_data_source(path)


def _read_polars_source(source: str | Path, has_header: bool) -> pl.DataFrame:
    if is_url(source):
        with urlopen(str(source)) as response:
            return pl.read_csv(
                io.BytesIO(response.read()),
                has_header=has_header,
                new_columns=COLUMN_NAMES if not has_header else None,
                null_values=["?"],
                try_parse_dates=False,
            )

    return pl.read_csv(
        source,
        has_header=has_header,
        new_columns=COLUMN_NAMES if not has_header else None,
        null_values=["?"],
        try_parse_dates=False,
    )


def load_with_pandas(path: str | Path) -> pd.DataFrame:
    data_path = ensure_data_source(path)
    df = pd.read_csv(data_path, na_values=["?"], skipinitialspace=True)

    if not set(COLUMN_NAMES).issubset(df.columns) and len(df.columns) == len(COLUMN_NAMES):
        df = pd.read_csv(
            data_path,
            header=None,
            names=COLUMN_NAMES,
            na_values=["?"],
            skipinitialspace=True,
        )

    df.columns = [str(column).strip() for column in df.columns]
    return df


def load_with_polars(path: str | Path) -> pl.DataFrame:
    data_path = ensure_data_source(path)
    df = _read_polars_source(data_path, has_header=True)

    if not set(COLUMN_NAMES).issubset(set(df.columns)) and len(df.columns) == len(COLUMN_NAMES):
        df = _read_polars_source(data_path, has_header=False)

    df = df.rename({column: column.strip() for column in df.columns})
    for column, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.String:
            stripped = pl.col(column).str.strip_chars()
            df = df.with_columns(
                pl.when(stripped == "?").then(None).otherwise(stripped).alias(column)
            )

    return df


def compare_loaders(pandas_df: pd.DataFrame, polars_df: pl.DataFrame) -> dict[str, Any]:
    pandas_missing = int(pandas_df.isna().sum().sum())
    polars_missing = int(sum(polars_df.null_count().row(0)))

    return {
        "pandas": {
            "rows": int(len(pandas_df)),
            "columns": int(len(pandas_df.columns)),
            "missing_values": pandas_missing,
            "duplicate_rows": int(pandas_df.duplicated().sum()),
            "dtypes": {column: str(dtype) for column, dtype in pandas_df.dtypes.items()},
        },
        "polars": {
            "rows": int(polars_df.height),
            "columns": int(polars_df.width),
            "missing_values": polars_missing,
            "duplicate_rows": int(polars_df.is_duplicated().sum()),
            "dtypes": {column: str(dtype) for column, dtype in zip(polars_df.columns, polars_df.dtypes)},
        },
        "same_shape": pandas_df.shape == polars_df.shape,
    }
