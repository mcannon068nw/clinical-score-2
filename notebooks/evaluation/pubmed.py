"""Fetch, parse, cache, and filter PubMed abstracts for evaluation."""

import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from defusedxml import ElementTree
from notebooks.evaluation.utils import INTERACTION_COLUMNS, clean_value, collapse_ws, norm_text

logger = logging.getLogger(__name__)

DRUG_TOKEN_STOPWORDS = {
    "acid",
    "chloride",
    "disodium",
    "esylate",
    "fumarate",
    "hydrochloride",
    "maleate",
    "mesylate",
    "phosphate",
    "potassium",
    "sodium",
    "sulfate",
    "tartrate",
}

GENERIC_DRUG_TOKEN_STOPWORDS = {
    "agent",
    "agents",
    "antagonist",
    "antagonists",
    "antibody",
    "antibodies",
    "biologic",
    "biological",
    "biosimilar",
    "cell",
    "cells",
    "compound",
    "compounds",
    "dna",
    "dose",
    "drug",
    "drugs",
    "factor",
    "factors",
    "fibroblast",
    "generic",
    "growth",
    "human",
    "inhibitor",
    "inhibitors",
    "injection",
    "origin",
    "peptide",
    "peptides",
    "protein",
    "proteins",
    "recombinant",
    "salt",
    "solution",
    "therapy",
    "treatment",
}

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.Timeout,
)
ABSTRACT_COLUMNS = ["pmid", "article_title", "abstract"]


def element_text(element: object | None) -> str:
    """Return collapsed text content for an XML element."""
    return "" if element is None else collapse_ws("".join(element.itertext()))


def parse_pubmed_abstracts(xml_text: str) -> pd.DataFrame:
    """Parse PubMed efetch XML into PMID, title, and abstract rows."""
    rows = []
    for article in ElementTree.fromstring(xml_text).findall(".//PubmedArticle"):
        parts = []
        for item in article.findall(".//Abstract/AbstractText"):
            text = element_text(item)
            label = clean_value(
                item.attrib.get("Label") or item.attrib.get("NlmCategory") or ""
            )
            if text:
                parts.append(f"{label}: {text}" if label else text)
        rows.append(
            {
                "pmid": clean_value(
                    element_text(article.find(".//MedlineCitation/PMID"))
                ),
                "article_title": element_text(article.find(".//ArticleTitle")),
                "abstract": "\n".join(parts).strip(),
            }
        )
    return pd.DataFrame(rows, columns=ABSTRACT_COLUMNS).drop_duplicates("pmid")


def _sleep_before_retry(attempt: int, backoff_seconds: float) -> None:
    """Sleep with exponential backoff and small jitter before a retry."""
    time.sleep(backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.25))


def fetch_pubmed_abstract_batch(
    pmids: list[str],
    efetch_url: str,
    ncbi_tool: str = "dgilit-pmid-eval",
    ncbi_email: str = "",
    max_retries: int = 5,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch one PubMed abstract batch from NCBI efetch with retries."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": ncbi_tool,
    }
    if ncbi_email:
        params["email"] = ncbi_email

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                efetch_url,
                params=params,
                headers={"User-Agent": ncbi_tool},
                timeout=(10, 120),
            )
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"Retryable HTTP status {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            return parse_pubmed_abstracts(response.text)
        except RETRYABLE_REQUEST_EXCEPTIONS as exc:
            last_error = exc
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS_CODES:
                raise

        if attempt < max_retries:
            logger.warning(
                "PubMed efetch failed for %s PMID(s) on attempt %s/%s: %s",
                len(pmids),
                attempt,
                max_retries,
                last_error,
            )
            _sleep_before_retry(attempt, backoff_seconds)

    raise RuntimeError(
        f"PubMed efetch failed after {max_retries} attempts for {len(pmids)} PMID(s)."
    ) from last_error


def _merge_abstracts(cached: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat([cached, fetched], ignore_index=True)[ABSTRACT_COLUMNS]
        .fillna("")
        .drop_duplicates("pmid", keep="last")
    )


def fetch_pubmed_abstracts(
    pmids: list[str],
    cache_path: str | Path,
    efetch_url: str,
    batch_size: int = 100,
    sleep_seconds: float = 0.34,
    refresh: bool = False,
    use_cache: bool = True,
    ncbi_tool: str = "dgilit-pmid-eval",
    ncbi_email: str = "",
    progress: bool = False,
    max_retries: int = 5,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Load cached abstracts and fetch missing PMID records."""
    pmids = [str(pmid).strip() for pmid in dict.fromkeys(pmids) if str(pmid).strip()]
    cache_path = Path(cache_path)

    cached = pd.DataFrame(columns=ABSTRACT_COLUMNS)
    if use_cache and cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, dtype=str, keep_default_na=False)
        cached = cached.reindex(columns=ABSTRACT_COLUMNS, fill_value="")
        cached = cached.drop_duplicates("pmid")

    cached_pmids = set(cached["pmid"].astype(str))
    missing = [pmid for pmid in pmids if pmid not in cached_pmids]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if progress:
            logger.info(
                "Fetching PubMed abstracts %s-%s of %s missing PMIDs",
                start + 1,
                min(start + batch_size, len(missing)),
                len(missing),
            )

        try:
            frame = fetch_pubmed_abstract_batch(
                batch,
                efetch_url,
                ncbi_tool,
                ncbi_email,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
        except RuntimeError:
            if len(batch) == 1:
                raise
            logger.warning(
                "Retrying failed PubMed batch as single-PMID requests: %s-%s",
                start + 1,
                min(start + batch_size, len(missing)),
            )
            frame = pd.concat(
                [
                    fetch_pubmed_abstract_batch(
                        [pmid],
                        efetch_url,
                        ncbi_tool,
                        ncbi_email,
                        max_retries=max_retries,
                        backoff_seconds=backoff_seconds,
                    )
                    for pmid in batch
                ],
                ignore_index=True,
            )

        found = set(frame["pmid"].astype(str)) if not frame.empty else set()
        blanks = pd.DataFrame(
            [
                {"pmid": pmid, "article_title": "", "abstract": ""}
                for pmid in batch
                if pmid not in found
            ]
        )
        fetched_batch = pd.concat([frame, blanks], ignore_index=True)[ABSTRACT_COLUMNS]
        cached = _merge_abstracts(cached, fetched_batch)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(cache_path, index=False)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return cached[cached["pmid"].astype(str).isin(pmids)].reset_index(drop=True)


def normalized_text_contains(text: object, name: object) -> bool:
    """Return whether normalized text contains a normalized name phrase."""
    name_norm = norm_text(name)
    return bool(name_norm and f" {name_norm} " in f" {norm_text(text)} ")


def compact_alnum_contains(text: object, name: object) -> bool:
    """Return whether compacted text contains a compact alphanumeric name."""
    compact_name = re.sub(r"[^a-z0-9]+", "", norm_text(name))
    if not (re.search(r"[a-z]", compact_name) and re.search(r"[0-9]", compact_name)):
        return False
    compact_text = re.sub(r"[^a-z0-9]+", "", norm_text(text))
    return compact_name in compact_text


def drug_token_recoverable(
    text: object, token: str, *, compact_alnum_drug_code_match: bool = True
) -> bool:
    """Return whether a fallback drug token is recoverable from text."""
    return normalized_text_contains(text, token) or (
        compact_alnum_drug_code_match and compact_alnum_contains(text, token)
    )


def drug_name_tokens(
    drug_name: object,
    *,
    strict_drug_token_filter: bool = True,
    extra_generic_drug_tokens: set[str] | None = None,
) -> list[str]:
    """Return drug-name tokens eligible for fallback recoverability matching."""
    stopwords = set(DRUG_TOKEN_STOPWORDS)
    if strict_drug_token_filter:
        stopwords.update(GENERIC_DRUG_TOKEN_STOPWORDS)
    if extra_generic_drug_tokens:
        stopwords.update(norm_text(token) for token in extra_generic_drug_tokens)
    return [
        token
        for token in norm_text(drug_name).split()
        if len(token) >= 4 and token not in stopwords
    ]


def drug_name_recoverable(
    text: object,
    drug_name: object,
    *,
    strict_drug_token_filter: bool = True,
    extra_generic_drug_tokens: set[str] | None = None,
    compact_alnum_drug_code_match: bool | None = None,
) -> bool:
    """Return whether a drug name is recoverable from title or abstract text."""
    if compact_alnum_drug_code_match is None:
        compact_alnum_drug_code_match = strict_drug_token_filter
    if normalized_text_contains(text, drug_name):
        return True
    return any(
        drug_token_recoverable(
            text,
            token,
            compact_alnum_drug_code_match=compact_alnum_drug_code_match,
        )
        for token in drug_name_tokens(
            drug_name,
            strict_drug_token_filter=strict_drug_token_filter,
            extra_generic_drug_tokens=extra_generic_drug_tokens,
        )
    )


def gene_name_recoverable(text: object, gene_name: object) -> bool:
    """Return whether a gene symbol is recoverable from title or abstract text."""
    gene = re.sub(r"[^A-Za-z0-9]+", "", clean_value(gene_name))
    if not gene:
        return False
    pattern = (
        r"(?<![A-Za-z0-9])"
        + r"[^A-Za-z0-9]*".join(map(re.escape, gene))
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, str(text), flags=re.IGNORECASE) is not None


def pair_recoverable(
    article_title: object,
    abstract: object,
    drug_name: object,
    gene_name: object,
    *,
    strict_drug_token_filter: bool = True,
    extra_generic_drug_tokens: set[str] | None = None,
    compact_alnum_drug_code_match: bool | None = None,
) -> bool:
    """Return whether a drug-gene pair is recoverable from title and abstract text."""
    text = f"{article_title or ''}\n{abstract or ''}"
    return drug_name_recoverable(
        text,
        drug_name,
        strict_drug_token_filter=strict_drug_token_filter,
        extra_generic_drug_tokens=extra_generic_drug_tokens,
        compact_alnum_drug_code_match=compact_alnum_drug_code_match,
    ) and gene_name_recoverable(text, gene_name)


def filter_interactions_with_abstracts(
    interactions_df: pd.DataFrame,
    abstracts_df: pd.DataFrame,
    min_pmids: int | None = None,
    *,
    strict_drug_token_filter: bool = True,
    extra_generic_drug_tokens: set[str] | None = None,
    compact_alnum_drug_code_match: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep interactions whose PMID has a non-empty abstract containing both names."""
    before = interactions_df["pmid"].astype(str).nunique()
    abstracts = abstracts_df[
        abstracts_df["abstract"].map(lambda value: collapse_ws(value) != "")
    ][["pmid", "article_title", "abstract"]].copy()
    abstract_pmids = set(abstracts["pmid"].astype(str))
    interactions = interactions_df[
        interactions_df["pmid"].astype(str).isin(abstract_pmids)
    ].copy()
    after_abstract = interactions["pmid"].astype(str).nunique()

    merged = interactions.merge(abstracts, on="pmid", how="left")
    if merged.empty:
        interactions = interactions.reindex(columns=INTERACTION_COLUMNS)
    else:
        pair_mask = merged.apply(
            lambda row: pair_recoverable(
                row["article_title"],
                row["abstract"],
                row["drug_name"],
                row["gene_name"],
                strict_drug_token_filter=strict_drug_token_filter,
                extra_generic_drug_tokens=extra_generic_drug_tokens,
                compact_alnum_drug_code_match=compact_alnum_drug_code_match,
            ),
            axis=1,
        )
        interactions = merged.loc[pair_mask, INTERACTION_COLUMNS].drop_duplicates()

    interactions = interactions.sort_values(INTERACTION_COLUMNS).reset_index(drop=True)
    after_pair = interactions["pmid"].astype(str).nunique()
    interaction_pmids = set(interactions["pmid"].astype(str))
    abstracts = abstracts[abstracts["pmid"].astype(str).isin(interaction_pmids)]

    if interactions.empty:
        raise ValueError(
            "No eligible interactions with PubMed abstracts are available."
        )
    if min_pmids is not None and after_pair < min_pmids:
        raise ValueError(
            f"Only {after_pair} eligible PMID(s) have recoverable PubMed pairs; "
            f"cannot sample {min_pmids}."
        )

    return (
        interactions,
        abstracts.reset_index(drop=True),
        {
            "pmids_before_abstract_filter": before,
            "pmids_removed_without_abstract": before - after_abstract,
            "pmids_after_abstract_filter": after_abstract,
            "pmids_removed_without_recoverable_pair": after_abstract - after_pair,
            "pmids_after_pair_filter": after_pair,
            "eligible_pair_rows_after_filter": len(interactions),
        },
    )
