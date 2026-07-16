"""Build deterministic answer sheets and task payloads for evaluation."""

import random

import pandas as pd
from notebooks.evaluation.utils import INTERACTION_COLUMNS, clean_value


def simple_random_sample_pmids(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Sample unique PMIDs from an eligible interaction frame."""
    pmids = sorted(df["pmid"].astype(str).unique())
    if len(pmids) < n:
        raise ValueError(
            f"Only {len(pmids)} eligible PMIDs are available; cannot sample {n}."
        )
    return sorted(random.Random(seed).sample(pmids, n))  # noqa: S311


def clean_interaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return complete, unique interaction rows in stable order."""
    df = df.reindex(columns=INTERACTION_COLUMNS, fill_value="").fillna("")
    df = df.assign(
        **{column: df[column].map(clean_value) for column in INTERACTION_COLUMNS}
    )
    return (
        df[(df[INTERACTION_COLUMNS] != "").all(axis=1)]
        .drop_duplicates()
        .sort_values(INTERACTION_COLUMNS)
        .reset_index(drop=True)
    )


def build_answer_sheet(
    eligible_df: pd.DataFrame,
    n_pmids: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create answer-sheet interactions for a deterministic PMID sample."""
    selected_pmids = simple_random_sample_pmids(eligible_df, n_pmids, seed)
    df = eligible_df[eligible_df["pmid"].astype(str).isin(selected_pmids)].copy()
    df = clean_interaction_frame(df)

    if df["pmid"].nunique() != n_pmids:
        raise ValueError(
            "Answer sheet does not contain exactly the requested number of PMIDs."
        )
    return df, df.copy(), selected_pmids


def task_pmids_from_abstracts(
    selected_pmids: list[str], abstracts_df: pd.DataFrame
) -> pd.DataFrame:
    """Return selected PubMed records in task order."""
    order = {str(pmid).strip(): index for index, pmid in enumerate(selected_pmids)}
    df = abstracts_df[abstracts_df["pmid"].astype(str).isin(order)].copy()
    missing = set(order) - set(df["pmid"].astype(str))
    if missing:
        raise ValueError(
            f"Missing PubMed abstracts for selected PMID(s): {sorted(missing)}"
        )
    df["_order"] = df["pmid"].astype(str).map(order)
    return df.sort_values("_order")[["pmid", "article_title", "abstract"]].reset_index(
        drop=True
    )


def task_payloads_from_pmids(task_pmids_df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert PubMed task rows into WAGS prompt payloads."""
    return [
        {
            "pmid": str(row.pmid),
            "article_title": str(row.article_title or ""),
            "abstract": str(row.abstract or ""),
        }
        for row in task_pmids_df.itertuples(index=False)
    ]
