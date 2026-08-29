"""Fetch, cache, and filter DGIdb publication-backed drug-gene pairs."""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from notebooks.evaluation.utils import INTERACTION_COLUMNS, clean_value

logger = logging.getLogger(__name__)

GRAPHQL_INTERACTION_SELECTION = """
    nodes {
      id
      interactionScore
      evidenceScore
      drug { name conceptId approved immunotherapy antiNeoplastic }
      gene { name conceptId longName }
      publications { pmid citation }
      sources {
        sourceDbName
        fullName
        sourceTrustLevel { level }
      }
      interactionClaims {
        id
        drugClaim { name }
        geneClaim { name }
        source {
          sourceDbName
          fullName
          sourceTrustLevel { level }
        }
        publications { pmid citation }
      }
    }
    pageInfo { hasNextPage endCursor }
"""

GRAPHQL_QUERY = f"""
query getPublicationInteractions($first: Int!, $after: String) {{
  interactions(first: $first, after: $after) {{
{GRAPHQL_INTERACTION_SELECTION}
  }}
}}
"""

GRAPHQL_GENE_QUERY = f"""
query getPublicationInteractionsByGene(
  $geneNames: [String!],
  $first: Int!,
  $after: String
) {{
  interactions(geneNames: $geneNames, first: $first, after: $after) {{
{GRAPHQL_INTERACTION_SELECTION}
  }}
}}
"""

SOURCE_COLUMNS = [
    "pmid",
    "drug_name",
    "gene_name",
    "pmid_source",
    "interaction_id",
    "interaction_score",
    "evidence_score",
    "drug_concept_id",
    "drug_approved",
    "drug_immunotherapy",
    "drug_anti_neoplastic",
    "gene_concept_id",
    "gene_long_name",
    "source_db_name",
    "source_name",
    "source_trust_level",
    "claim_id",
    "claim_drug_name",
    "claim_gene_name",
    "claim_source_db_name",
    "claim_source_name",
    "claim_source_trust_level",
    "publication_citation",
]

CLAIM_COLUMNS = [
    "claim_id",
    "claim_drug_name",
    "claim_gene_name",
    "claim_source_db_name",
    "claim_source_name",
    "claim_source_trust_level",
]


def graphql_post(
    session: requests.Session, graphql_url: str, query: str, variables: dict[str, Any]
) -> dict[str, Any]:
    """POST a DGIdb GraphQL request and return the decoded payload."""
    response = session.post(
        graphql_url,
        json={"query": query, "variables": variables},
        headers={
            "User-Agent": "dgilit-pmid-eval",
            "dgidb-client-name": "dgilit-pmid-eval",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2)[:4000])
    return payload


def publication_records(
    publications: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Extract non-empty PMID records from DGIdb publication records."""
    records = []
    seen = set()
    for publication in publications or []:
        pmid = clean_value(publication.get("pmid"))
        if pmid and pmid not in seen:
            records.append(
                {
                    "pmid": pmid,
                    "publication_citation": clean_value(publication.get("citation")),
                }
            )
            seen.add(pmid)
    return records


def source_fields(source: dict[str, Any] | None, prefix: str = "") -> dict[str, str]:
    """Flatten a DGIdb source record."""
    source = source or {}
    trust = source.get("sourceTrustLevel") or {}
    return {
        f"{prefix}source_db_name": clean_value(source.get("sourceDbName")),
        f"{prefix}source_name": clean_value(source.get("fullName")),
        f"{prefix}source_trust_level": clean_value(trust.get("level")),
    }


def joined(values: Iterable[str]) -> str:
    """Return unique, sorted, non-empty values as a pipe-delimited string."""
    return "|".join(
        sorted({cleaned for value in values if (cleaned := clean_value(value))})
    )


def interaction_source_fields(interaction: dict[str, Any]) -> dict[str, str]:
    """Flatten aggregate source records for one DGIdb interaction."""
    sources = interaction.get("sources") or []
    return {
        "source_db_name": joined(source.get("sourceDbName") for source in sources),
        "source_name": joined(source.get("fullName") for source in sources),
        "source_trust_level": joined(
            (source.get("sourceTrustLevel") or {}).get("level") for source in sources
        ),
    }


def base_interaction_fields(interaction: dict[str, Any]) -> dict[str, str]:
    """Flatten fields shared by rows from one DGIdb interaction."""
    drug = interaction.get("drug") or {}
    gene = interaction.get("gene") or {}
    return {
        "interaction_id": clean_value(interaction.get("id")),
        "interaction_score": clean_value(interaction.get("interactionScore")),
        "evidence_score": clean_value(interaction.get("evidenceScore")),
        "drug_name": clean_value(drug.get("name")),
        "gene_name": clean_value(gene.get("name")),
        "drug_concept_id": clean_value(drug.get("conceptId")),
        "drug_approved": clean_value(drug.get("approved")),
        "drug_immunotherapy": clean_value(drug.get("immunotherapy")),
        "drug_anti_neoplastic": clean_value(drug.get("antiNeoplastic")),
        "gene_concept_id": clean_value(gene.get("conceptId")),
        "gene_long_name": clean_value(gene.get("longName")),
        **interaction_source_fields(interaction),
    }


def claim_fields(claim: dict[str, Any] | None = None) -> dict[str, str]:
    """Flatten DGIdb claim fields, or return empty claim columns."""
    if claim is None:
        return dict.fromkeys(CLAIM_COLUMNS, "")
    return {
        "claim_id": clean_value(claim.get("id")),
        "claim_drug_name": clean_value((claim.get("drugClaim") or {}).get("name")),
        "claim_gene_name": clean_value((claim.get("geneClaim") or {}).get("name")),
        **source_fields(claim.get("source"), "claim_"),
    }


def row_from_parts(
    base: dict[str, str],
    publication: dict[str, str],
    claim: dict[str, Any] | None = None,
    pmid_source: str = "interaction",
) -> dict[str, str]:
    """Build one publication-level drug-gene pair row."""
    return {**base, **publication, "pmid_source": pmid_source, **claim_fields(claim)}


def rows_from_claim(
    base: dict[str, str], claim: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand one DGIdb interaction claim into publication-level rows."""
    pmids = publication_records(claim.get("publications"))
    if not pmids or not base["drug_name"] or not base["gene_name"]:
        return []
    return [
        row_from_parts(base, publication, claim=claim, pmid_source="claim")
        for publication in pmids
    ]


def rows_from_interaction(interaction: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one DGIdb interaction into publication-level rows."""
    base = base_interaction_fields(interaction)
    if not base["gene_name"] or not base["drug_name"]:
        return []

    rows = [
        row
        for claim in interaction.get("interactionClaims") or []
        for row in rows_from_claim(base, claim)
    ]
    if rows:
        return rows
    return [
        row_from_parts(base, publication)
        for publication in publication_records(interaction.get("publications"))
    ]


def fetch_publication_interactions(
    graphql_url: str,
    graphql_query: str | None = None,
    gene_names: Iterable[str] | None = None,
    batch_size: int = 100,
    sleep_seconds: float = 0.05,
    max_pages: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch publication-level drug-gene pair rows from DGIdb GraphQL."""
    genes = None
    if gene_names is not None:
        genes = [gene for gene in dict.fromkeys(map(clean_value, gene_names)) if gene]

    rows: list[dict[str, Any]] = []
    after = None
    page = 0
    query = graphql_query or (
        GRAPHQL_GENE_QUERY if genes is not None else GRAPHQL_QUERY
    )

    with requests.Session() as session:
        while True:
            page += 1
            if progress:
                logger.info("Fetching DGIdb interaction page %s", page)
            variables = {"first": batch_size, "after": after}
            if genes is not None:
                variables["geneNames"] = genes

            payload = graphql_post(session, graphql_url, query, variables)
            interactions = payload.get("data", {}).get("interactions", {}) or {}
            rows.extend(
                row
                for interaction in interactions.get("nodes") or []
                for row in rows_from_interaction(interaction)
            )

            page_info = interactions.get("pageInfo") or {}
            if not page_info.get("hasNextPage") or (max_pages and page >= max_pages):
                break
            after = page_info.get("endCursor")
            if sleep_seconds:
                time.sleep(sleep_seconds)

    return (
        pd.DataFrame(rows, columns=SOURCE_COLUMNS)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def candidate_pair_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    """Return unique PMID-drug-gene candidate rows from a DGIdb source frame."""
    df = source_df.reindex(columns=INTERACTION_COLUMNS, fill_value="").fillna("")
    df = df.assign(
        **{column: df[column].map(clean_value) for column in INTERACTION_COLUMNS}
    )
    return (
        df[(df[INTERACTION_COLUMNS] != "").all(axis=1)][INTERACTION_COLUMNS]
        .drop_duplicates()
        .sort_values(INTERACTION_COLUMNS)
        .reset_index(drop=True)
    )


def load_cached_dgidb_source(
    source_cache_path: str | Path,
    graphql_url: str,
    refresh: bool = False,
    use_cache: bool = True,
    graphql_batch_size: int = 100,
    graphql_sleep_seconds: float = 0.05,
    max_graphql_pages: int | None = None,
    gene_names: Iterable[str] | None = None,
    progress: bool = False,
    return_source: bool = False,
) -> pd.DataFrame:
    """Load cached DGIdb source rows or rebuild them from DGIdb GraphQL."""
    source_cache_path = Path(source_cache_path)
    if use_cache and source_cache_path.exists() and not refresh:
        source = pd.read_csv(source_cache_path, dtype=str, keep_default_na=False)
        return source if return_source else candidate_pair_frame(source)

    source = fetch_publication_interactions(
        graphql_url=graphql_url,
        batch_size=graphql_batch_size,
        sleep_seconds=graphql_sleep_seconds,
        max_pages=max_graphql_pages,
        gene_names=gene_names,
        progress=progress,
    )
    if use_cache:
        source_cache_path.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(source_cache_path, index=False)
    return source if return_source else candidate_pair_frame(source)
