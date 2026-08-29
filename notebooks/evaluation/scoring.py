"""Scoring helpers for extracted drug-gene interaction predictions."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from notebooks.evaluation.utils import clean_value, norm_gene, norm_text

logger = logging.getLogger(__name__)

INTERACTION_COLUMNS = ["pmid", "drug_name", "gene_name"]
MATCH_COLUMNS = ["pmid_key", "drug_key", "gene_key"]
METRIC_COLUMNS = [
    "expected",
    "predicted",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "scored_pmids",
    "sampled_answer_pmids",
    "prediction_records",
    "successful_prediction_records",
    "error_records",
]


def predictions_to_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    """Convert saved prediction records into an interaction DataFrame."""
    rows = []
    for pmid, payload in predictions.items():
        if not isinstance(payload, dict):
            continue
        rows.extend(
            {
                "pmid": str(item.get("pmid") or pmid).strip(),
                "drug_name": clean_value(item.get("drug_name")),
                "gene_name": clean_value(item.get("gene_name")),
            }
            for item in payload.get("interactions", [])
        )
    return pd.DataFrame(rows, columns=INTERACTION_COLUMNS).drop_duplicates()


def standardize_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized matching keys for interaction rows."""
    df = df.copy().fillna("")
    for column in INTERACTION_COLUMNS:
        if column not in df:
            df[column] = ""
    df["pmid_key"] = df["pmid"].astype(str).str.strip()
    df["drug_key"] = df["drug_name"].map(norm_text)
    df["gene_key"] = df["gene_name"].map(norm_gene)
    return df[(df[MATCH_COLUMNS] != "").all(axis=1)]


def _tuples(df: pd.DataFrame, columns: list[str]) -> set[tuple[Any, ...]]:
    if df.empty:
        return set()
    return set(map(tuple, df[columns].drop_duplicates().to_numpy()))


def _metrics(
    expected_df: pd.DataFrame, predicted_df: pd.DataFrame, columns: list[str]
) -> dict[str, Any]:
    expected = _tuples(expected_df, columns)
    predicted = _tuples(predicted_df, columns)
    true_positives = expected & predicted
    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(expected) if expected else 0.0
    return {
        "expected": len(expected),
        "predicted": len(predicted),
        "true_positives": len(true_positives),
        "false_positives": len(predicted - expected),
        "false_negatives": len(expected - predicted),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def score_interaction_frames(
    expected_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score expected and predicted interaction frames at the drug-gene pair level."""
    expected = standardize_interactions(expected_df)
    predicted = standardize_interactions(predicted_df)
    return pd.DataFrame([_metrics(expected, predicted, MATCH_COLUMNS)])


def is_successful_prediction_record(payload: Any) -> bool:
    """Return whether a saved prediction record contains parsed interactions."""
    return (
        isinstance(payload, dict)
        and not payload.get("error")
        and "interactions" in payload
    )


def score_prediction_dict(
    answer_sheet: pd.DataFrame,
    predictions: dict[str, Any],
) -> pd.DataFrame:
    """Score an in-memory prediction dictionary against the answer sheet."""
    scored_pmids = {
        str(pmid)
        for pmid, payload in predictions.items()
        if is_successful_prediction_record(payload)
    }
    if not scored_pmids:
        logger.warning(
            "No successful prediction records are available; "
            "extraction metrics were not computed."
        )
        return pd.DataFrame(columns=METRIC_COLUMNS)

    expected = answer_sheet[answer_sheet["pmid"].astype(str).isin(scored_pmids)]
    predicted = {
        pmid: payload
        for pmid, payload in predictions.items()
        if str(pmid) in scored_pmids
    }
    metrics = score_interaction_frames(expected, predictions_to_frame(predicted))
    metrics["scored_pmids"] = len(scored_pmids)
    metrics["sampled_answer_pmids"] = answer_sheet["pmid"].astype(str).nunique()
    metrics["prediction_records"] = len(predictions)
    metrics["successful_prediction_records"] = len(scored_pmids)
    metrics["error_records"] = sum(
        1
        for payload in predictions.values()
        if isinstance(payload, dict) and payload.get("error")
    )
    return metrics


def score_predictions(
    answer_sheet: pd.DataFrame,
    predictions_path: str | Path,
    metrics_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load predictions from disk, score them, and optionally write metrics."""
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        logger.warning("Prediction file does not exist yet: %s", predictions_path)
        return score_prediction_dict(answer_sheet, {})

    metrics = score_prediction_dict(
        answer_sheet, json.loads(predictions_path.read_text())
    )
    if metrics_path is not None and not metrics.empty:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
    return metrics
