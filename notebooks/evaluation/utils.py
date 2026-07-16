"""Shared text normalization, table caching, and output-path helpers."""

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

INTERACTION_COLUMNS = ["pmid", "drug_name", "gene_name"]
MISSING_VALUES = {"", "NULL", "NA", "N/A", "NONE", "NAN"}


def clean_value(value: object) -> str:
    """Normalize missing-like scalar values to an empty string."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in MISSING_VALUES else text


def collapse_ws(value: object) -> str:
    """Collapse all whitespace runs in a value."""
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def norm_text(value: object) -> str:
    """Normalize text for case-insensitive exact matching."""
    return collapse_ws(re.sub(r"[^a-z0-9]+", " ", clean_value(value).lower()))


def norm_gene(value: object) -> str:
    """Normalize gene symbols for punctuation-insensitive matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", clean_value(value)).upper()


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a column by matching normalized candidate names."""
    normalized = {_column_key(column): column for column in df.columns}
    for name in candidates:
        if column := normalized.get(_column_key(name)):
            return column
    raise ValueError(f"Missing one of {candidates}; found {list(df.columns)}")


def read_table(source: str | Path, sep: str = "\t") -> pd.DataFrame:
    """Read a local or remote delimited table as strings."""
    source = str(source)
    if source.startswith(("http://", "https://")):
        response = requests.get(
            source, headers={"User-Agent": "dgilit-pmid-eval"}, timeout=120
        )
        response.raise_for_status()
        return pd.read_csv(
            StringIO(response.text), sep=sep, dtype=str, keep_default_na=False
        )
    return pd.read_csv(source, sep=sep, dtype=str, keep_default_na=False)


def read_cached_table(
    source: str | Path,
    cache_path: str | Path,
    sep: str = "\t",
    refresh: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Read a table through a local cache path when available."""
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, sep=sep, dtype=str, keep_default_na=False)

    df = read_table(source, sep=sep)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, sep=sep, index=False)
    return df


def out_path(
    out_dir: str | Path, name: str, suffix: str = "csv", n_pmids: int | None = None
) -> Path:
    """Build a workflow output path with an optional PMID-count suffix."""
    stem = f"{name}_{n_pmids}" if n_pmids is not None else name
    return Path(out_dir) / f"{stem}.{suffix}"
