"""Curate high-confidence PubMed drug-gene pair answer sheets."""

from collections import Counter

import pandas as pd
from notebooks.evaluation.pubmed import drug_name_recoverable, gene_name_recoverable, normalized_text_contains
from notebooks.evaluation.utils import INTERACTION_COLUMNS, clean_value, collapse_ws

CURATION_SOURCE_COLUMNS = [
    *INTERACTION_COLUMNS,
    "article_title",
    "abstract",
    "evidence_score",
    "interaction_score",
    "source_db_name",
    "claim_source_db_name",
    "source_trust_level",
    "claim_source_trust_level",
    "drug_approved",
    "drug_anti_neoplastic",
    "publication_citation",
]

CURATED_LITERATURE_COLUMNS = [
    "selection_rank",
    *INTERACTION_COLUMNS,
    "article_title",
    "abstract",
    "evidence_score_max",
    "interaction_score_max",
    "source_db_names",
    "claim_source_db_names",
    "source_trust_levels",
    "claim_source_trust_levels",
    "drug_approved",
    "drug_anti_neoplastic",
    "publication_citation",
    "selection_score",
    "interaction_count",
    "min_evidence_score",
    "mean_evidence_score",
]


def pipe_unique(values: pd.Series) -> str:
    parts: set[str] = set()
    for value in values:
        parts.update(part for part in str(value).split("|") if part)
    return "|".join(sorted(parts))


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = clean_value(value)
        if text:
            return text
    return ""


def true_if_any(values: pd.Series) -> str:
    return "True" if any(clean_value(value) == "True" for value in values) else "False"


def build_curated_pair_frame(
    source_df: pd.DataFrame,
    abstracts_df: pd.DataFrame,
    *,
    strict_drug_token_filter: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build scored, recoverable PMID-drug-gene candidate rows."""
    source = source_df.reindex(columns=CURATION_SOURCE_COLUMNS, fill_value="").fillna("")
    abstracts = abstracts_df.reindex(columns=["pmid", "article_title", "abstract"], fill_value="")
    abstracts = abstracts[abstracts["abstract"].map(collapse_ws).ne("")]

    merged = source.drop(columns=["article_title", "abstract"]).merge(
        abstracts,
        on="pmid",
        how="inner",
    )
    text = merged["article_title"].fillna("") + "\n" + merged["abstract"].fillna("")
    merged["drug_full_match"] = [
        normalized_text_contains(item, drug)
        for item, drug in zip(text, merged["drug_name"], strict=False)
    ]
    merged["drug_token_match"] = [
        (not full)
        and drug_name_recoverable(
            item,
            drug,
            strict_drug_token_filter=strict_drug_token_filter,
        )
        for full, item, drug in zip(
            merged["drug_full_match"], text, merged["drug_name"], strict=False
        )
    ]
    merged["gene_recoverable"] = [
        gene_name_recoverable(item, gene)
        for item, gene in zip(text, merged["gene_name"], strict=False)
    ]
    merged["pair_recoverable"] = (
        merged["drug_full_match"] | merged["drug_token_match"]
    ) & merged["gene_recoverable"]
    merged["evidence_score_num"] = pd.to_numeric(
        merged["evidence_score"], errors="coerce"
    ).fillna(0)
    merged["interaction_score_num"] = pd.to_numeric(
        merged["interaction_score"], errors="coerce"
    ).fillna(0)
    merged["expert_source"] = merged["source_trust_level"].str.contains(
        "Expert curated", regex=False
    ) | merged["claim_source_trust_level"].str.contains("Expert curated", regex=False)

    pair_frame = (
        merged[merged["pair_recoverable"]]
        .groupby(INTERACTION_COLUMNS, as_index=False)
        .agg(
            article_title=("article_title", "first"),
            abstract=("abstract", "first"),
            evidence_score_max=("evidence_score_num", "max"),
            interaction_score_max=("interaction_score_num", "max"),
            source_db_names=("source_db_name", pipe_unique),
            claim_source_db_names=("claim_source_db_name", pipe_unique),
            source_trust_levels=("source_trust_level", pipe_unique),
            claim_source_trust_levels=("claim_source_trust_level", pipe_unique),
            drug_approved=("drug_approved", true_if_any),
            drug_anti_neoplastic=("drug_anti_neoplastic", true_if_any),
            publication_citation=("publication_citation", first_nonempty),
            drug_full_match=("drug_full_match", "max"),
            drug_token_match=("drug_token_match", "max"),
            expert_source=("expert_source", "max"),
            source_row_count=("pmid", "size"),
        )
        .reset_index(drop=True)
    )

    return pair_frame, {
        "source_rows": int(len(source_df)),
        "source_unique_pmids": int(source_df["pmid"].astype(str).nunique()),
        "source_unique_pairs": int(
            source_df.reindex(columns=INTERACTION_COLUMNS, fill_value="")
            .drop_duplicates()
            .shape[0]
        ),
        "pubmed_cache_pmids": int(abstracts["pmid"].astype(str).nunique()),
        "recoverable_pair_pmids": int(pair_frame["pmid"].astype(str).nunique()),
        "recoverable_pairs": int(len(pair_frame)),
    }


def select_curated_literature_set(
    pair_frame: pd.DataFrame,
    n_pmids: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Select a diverse high-confidence literature set."""
    qualified = pair_frame[
        pair_frame["drug_full_match"]
        & pair_frame["expert_source"]
        & (pair_frame["evidence_score_max"] >= 10)
    ].copy()

    recoverable_counts = pair_frame.groupby("pmid").size()
    qualified_counts = qualified.groupby("pmid").size()
    clean_pmids = [
        pmid
        for pmid, count in recoverable_counts.items()
        if qualified_counts.get(pmid, 0) == count
    ]
    qualified = qualified[qualified["pmid"].isin(clean_pmids)].copy()

    pmid_frame = (
        qualified.groupby("pmid", as_index=False)
        .agg(
            interaction_count=("gene_name", "size"),
            min_evidence_score=("evidence_score_max", "min"),
            mean_evidence_score=("evidence_score_max", "mean"),
            max_evidence_score=("evidence_score_max", "max"),
            mean_interaction_score=("interaction_score_max", "mean"),
            article_title=("article_title", "first"),
            abstract=("abstract", "first"),
            drugs=("drug_name", lambda values: tuple(sorted(set(values)))),
            genes=("gene_name", lambda values: tuple(sorted(set(values)))),
        )
        .reset_index(drop=True)
    )
    pmid_frame["abstract_length"] = pmid_frame["abstract"].str.len()
    pmid_frame["selection_score"] = (
        pmid_frame["min_evidence_score"] * 10
        + pmid_frame["mean_evidence_score"]
        + pmid_frame["mean_interaction_score"] * 100
        + pmid_frame["interaction_count"] * 2
        + pmid_frame["abstract_length"].clip(upper=2500) / 500
    )
    pmid_frame = pmid_frame.sort_values(
        ["selection_score", "min_evidence_score", "mean_evidence_score", "pmid"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    selected: list[str] = []
    gene_counter: Counter[str] = Counter()
    drug_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()
    selection_rank: dict[str, int] = {}
    caps = {"gene": 5, "drug": 5, "pair": 3}

    for row in pmid_frame.itertuples(index=False):
        rows = qualified[qualified["pmid"] == row.pmid]
        genes = set(rows["gene_name"])
        drugs = set(rows["drug_name"])
        pairs = set(rows[["drug_name", "gene_name"]].itertuples(index=False, name=None))
        if any(gene_counter[gene] >= caps["gene"] for gene in genes):
            continue
        if any(drug_counter[drug] >= caps["drug"] for drug in drugs):
            continue
        if any(pair_counter[pair] >= caps["pair"] for pair in pairs):
            continue

        selected.append(row.pmid)
        selection_rank[row.pmid] = len(selected)
        for gene in genes:
            gene_counter[gene] += 1
        for drug in drugs:
            drug_counter[drug] += 1
        for pair in pairs:
            pair_counter[pair] += 1
        if len(selected) == n_pmids:
            break

    if len(selected) != n_pmids:
        raise RuntimeError(f"Selected {len(selected)} PMIDs; expected {n_pmids}.")

    final = qualified[qualified["pmid"].isin(selected)].copy()
    pmid_selected = pmid_frame[pmid_frame["pmid"].isin(selected)].copy()
    final["selection_rank"] = final["pmid"].map(selection_rank)
    final = final.merge(
        pmid_selected[
            [
                "pmid",
                "selection_score",
                "interaction_count",
                "min_evidence_score",
                "mean_evidence_score",
            ]
        ],
        on="pmid",
        how="left",
        suffixes=("", "_pmid"),
    )
    final = final.sort_values(
        ["selection_rank", "pmid", "drug_name", "gene_name"]
    ).reset_index(drop=True)
    pmid_selected["selection_rank"] = pmid_selected["pmid"].map(selection_rank)
    pmid_selected = pmid_selected.sort_values("selection_rank").reset_index(drop=True)

    return final[CURATED_LITERATURE_COLUMNS], pmid_selected, {
        "qualified_pmids_after_quality_filters": int(qualified["pmid"].nunique()),
        "qualified_pairs_after_quality_filters": int(len(qualified)),
        "selected_pmids": int(final["pmid"].nunique()),
        "selected_interaction_rows": int(len(final)),
        "selected_unique_drugs": int(final["drug_name"].nunique()),
        "selected_unique_genes": int(final["gene_name"].nunique()),
        "max_pmids_per_gene_cap": caps["gene"],
        "max_pmids_per_drug_cap": caps["drug"],
        "max_pmids_per_pair_cap": caps["pair"],
        "min_selected_evidence_score": float(final["evidence_score_max"].min()),
    }


def build_curated_answer_sheet(
    source_df: pd.DataFrame,
    abstracts_df: pd.DataFrame,
    n_pmids: int,
    *,
    strict_drug_token_filter: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Build answer sheet, full curated rows, PMID audit rows, and summary stats."""
    pair_frame, source_stats = build_curated_pair_frame(
        source_df,
        abstracts_df,
        strict_drug_token_filter=strict_drug_token_filter,
    )
    final, pmid_audit, selection_stats = select_curated_literature_set(
        pair_frame,
        n_pmids,
    )
    answer_sheet = final[INTERACTION_COLUMNS].copy()
    pmid_audit = pmid_audit.drop(columns=["drugs", "genes"], errors="ignore")
    return answer_sheet, final, pmid_audit, {**source_stats, **selection_stats}
