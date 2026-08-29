"""Prompt optimization helpers for the evaluation workflow."""

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from notebooks.evaluation.scoring import score_prediction_dict
from notebooks.evaluation.utils import clean_value, collapse_ws
from notebooks.evaluation.wags import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_ABSTRACT_TOKEN,
    PROMPT_PMID_TOKEN,
    PROMPT_TITLE_TOKEN,
    default_user_prompt_template,
    initialize_wags_client,
    run_wags_predictions,
)

logger = logging.getLogger(__name__)

OPTIMIZER_SYSTEM_PROMPT = (
    "You improve prompt templates for biomedical extraction tasks. "
    "Return only valid JSON matching the requested schema."
)


class PromptCandidateResponse(BaseModel):
    """Candidate prompt templates returned by the optimizer model."""

    model_config = ConfigDict(extra="ignore")

    prompts: list[str]


def valid_prompt_template(prompt_template: str) -> bool:
    """Return whether a candidate keeps the required task placeholders."""
    return all(
        token in prompt_template
        for token in [PROMPT_PMID_TOKEN, PROMPT_TITLE_TOKEN, PROMPT_ABSTRACT_TOKEN]
    )


def unique_valid_templates(
    prompt_templates: Iterable[str],
    current_prompt_template: str,
    limit: int,
) -> list[str]:
    """Return unique valid prompt templates not equal to the current best."""
    selected = []
    seen = {current_prompt_template.strip()}
    for prompt_template in prompt_templates:
        prompt_template = clean_value(prompt_template)
        if prompt_template not in seen and valid_prompt_template(prompt_template):
            selected.append(prompt_template)
            seen.add(prompt_template)
    return selected[:limit]


def fallback_candidate_templates() -> list[str]:
    """Return deterministic optimizer candidates."""
    return [
        f"""
Extract only explicitly supported drug-gene interaction pairs from the PubMed title and abstract.

Return only JSON matching this shape:
{{"interactions": [{{"drug_name": "drug name", "gene_name": "gene symbol/name"}}]}}

Rules:
- Base the answer only on the supplied title and abstract.
- Include a pair only when a named drug, compound, inhibitor, agonist, antagonist, antibody, or therapy is directly linked to a named gene, protein, receptor, enzyme, or kinase.
- Create one object per distinct drug-gene pair.
- Use names as written in the title or abstract when possible.
- Exclude pathway context, disease response, toxicity, pharmacokinetics, biomarkers, and tumor response unless a specific drug-gene relationship is stated.
- Return {{"interactions": []}} if no supported pair is present.

PMID: {PROMPT_PMID_TOKEN}
Article title: {PROMPT_TITLE_TOKEN}

Abstract:
{PROMPT_ABSTRACT_TOKEN}
""".strip(),
        f"""
You are extracting curated drug-gene pairs from PubMed abstracts.

Output valid JSON only:
{{"interactions": [{{"drug_name": "...", "gene_name": "..."}}]}}

Extraction policy:
- Extract every explicit relationship between a named drug or compound and a named gene/protein target.
- Keep separate pairs for multiple targets and multiple drugs.
- Prefer the most specific compound name and the most specific gene/protein name stated in the abstract.
- Do not infer interactions from class membership, downstream signaling, clinical benefit, cell-line response, or disease association alone.
- Do not include interactions that require outside knowledge.
- Return {{"interactions": []}} when no drug-gene pair is supported.

PMID: {PROMPT_PMID_TOKEN}
Title: {PROMPT_TITLE_TOKEN}
Abstract:
{PROMPT_ABSTRACT_TOKEN}
""".strip(),
        f"""
From the PMID, title, and abstract below, identify drug-gene interaction pairs supported by direct text evidence.

Return exactly one JSON object and no prose:
{{"interactions": [{{"drug_name": "drug name", "gene_name": "gene name"}}]}}

Guidelines:
- A valid interaction names both a drug/compound/therapeutic agent and a gene/protein/receptor/enzyme target.
- Direct inhibition, activation, binding, antagonism, agonism, targeting, modulation, sensitization tied to a named target, or resistance mediated by a named gene may support a pair.
- Exclude broad pathway effects and unsupported inferences.
- Use one row per PMID-drug-gene combination and remove duplicates.
- Return {{"interactions": []}} for no supported interactions.

PMID: {PROMPT_PMID_TOKEN}
Article title: {PROMPT_TITLE_TOKEN}

Abstract:
{PROMPT_ABSTRACT_TOKEN}
""".strip(),
    ]


def prompt_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    """Return the first metrics row as a plain dictionary."""
    return {} if metrics.empty else metrics.iloc[0].to_dict()


def prompt_score(metrics: pd.DataFrame, score_metric: str) -> float:
    """Return the selected optimization score."""
    values = prompt_metrics(metrics)
    return float(values.get(score_metric, 0.0) or 0.0)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def prompt_predictions_path(out_dir: str | Path, prompt_id: str) -> Path:
    """Return the prediction path for one optimizer prompt."""
    return Path(out_dir) / "prompt_optimizer" / f"predictions_{_slug(prompt_id)}.json"


def test_prompt_template(
    prompt_template: str,
    prompt_id: str,
    iteration: int,
    prompt_kind: str,
    task_payloads: list[dict[str, str]],
    answer_sheet: pd.DataFrame,
    out_dir: str | Path,
    enable_llm_calls: bool,
    model_key: str,
    model_id: str,
    prompt_name: str,
    prompt_version: str,
    aws_profile_name: str,
    bedrock_region: str,
    max_tokens: int,
    temperature: float,
    cache_max_entries: int = 500,
    request_sleep_seconds: float = 0.25,
    allowed_pmids: Iterable[str] | None = None,
    progress: bool = False,
    system_prompt: str | None = None,
    score_metric: str = "f1",
) -> dict[str, Any]:
    """Run one prompt template through prediction and scoring."""
    predictions = run_wags_predictions(
        task_payloads,
        predictions_path=prompt_predictions_path(out_dir, prompt_id),
        enable_llm_calls=enable_llm_calls,
        model_key=model_key,
        model_id=model_id,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        aws_profile_name=aws_profile_name,
        bedrock_region=bedrock_region,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_max_entries=cache_max_entries,
        overwrite=True,
        request_sleep_seconds=request_sleep_seconds,
        allowed_pmids=allowed_pmids,
        progress=progress,
        system_prompt=system_prompt,
        user_prompt_template=prompt_template,
    )
    metrics = score_prediction_dict(answer_sheet, predictions)
    return {
        "prompt_id": prompt_id,
        "iteration": iteration,
        "prompt_kind": prompt_kind,
        "prompt_template": prompt_template,
        "score": prompt_score(metrics, score_metric),
        "metrics": prompt_metrics(metrics),
        "prediction_records": len(predictions),
    }


def summary_row(
    result: dict[str, Any],
    score_metric: str,
    best_score_before_test: float | None,
    selected_after_iteration: bool = False,
) -> dict[str, Any]:
    """Return a compact optimizer summary row."""
    row = {
        "prompt_id": result["prompt_id"],
        "iteration": result["iteration"],
        "prompt_kind": result["prompt_kind"],
        "score_metric": score_metric,
        "score": result["score"],
        "best_score_before_test": best_score_before_test,
        "score_delta_from_best_before_test": None
        if best_score_before_test is None
        else result["score"] - best_score_before_test,
        "selected_after_iteration": selected_after_iteration,
        "prompt_preview": collapse_ws(result["prompt_template"])[:300],
    }
    row.update(result["metrics"])
    return row


def candidate_generation_prompt(
    current_best_prompt: str,
    best_score: float,
    score_metric: str,
    candidates_per_iteration: int,
    summary_rows: list[dict[str, Any]],
) -> str:
    """Build the optimizer model prompt."""
    recent = [
        {
            "prompt_id": row["prompt_id"],
            "iteration": row["iteration"],
            "score": row["score"],
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "f1": row.get("f1"),
        }
        for row in summary_rows[-8:]
    ]
    return f"""
You are improving a compact prompt template for extracting drug-gene interaction pairs from PubMed abstracts.

The prompt template must include these exact placeholder tokens:
{PROMPT_PMID_TOKEN}
{PROMPT_TITLE_TOKEN}
{PROMPT_ABSTRACT_TOKEN}

Current best {score_metric}: {best_score:.4f}

Recent tested prompt scores:
{json.dumps(recent, indent=2)}

Current best prompt template:
{current_best_prompt}

Generate {candidates_per_iteration} improved full prompt templates.
Return only JSON matching this schema:
{{"prompts": ["full prompt template", "full prompt template"]}}
Do not replace placeholders with real PMID, title, or abstract values.
Preserve the output JSON fields: interactions, drug_name, gene_name.
""".strip()


def generate_candidate_templates(
    current_best_prompt: str,
    best_score: float,
    score_metric: str,
    candidates_per_iteration: int,
    summary_rows: list[dict[str, Any]],
    enable_llm_calls: bool,
    model_id: str,
    aws_profile_name: str,
    bedrock_region: str,
    optimizer_max_tokens: int,
    optimizer_temperature: float,
    optimizer_system_prompt: str | None = None,
) -> dict[str, Any]:
    """Generate candidate prompt templates."""
    if not enable_llm_calls:
        return {"raw_response": "", "error": "ENABLE_LLM_CALLS is False", "prompts": []}

    try:
        client = initialize_wags_client(
            model_id,
            bedrock_region,
            aws_profile_name,
            optimizer_max_tokens,
            optimizer_temperature,
        )
        response = client.invoke_json(
            system_prompt=optimizer_system_prompt or OPTIMIZER_SYSTEM_PROMPT,
            user_prompt=candidate_generation_prompt(
                current_best_prompt,
                best_score,
                score_metric,
                candidates_per_iteration,
                summary_rows,
            ),
            json_schema=PromptCandidateResponse.model_json_schema(),
        )
        payload = PromptCandidateResponse.model_validate(response.parsed_json)
        return {
            "raw_response": getattr(response, "raw_text", ""),
            "error": "",
            "prompts": payload.prompts,
        }
    except Exception as exc:
        logger.warning("Prompt candidate generation failed: %s", exc)
        return {"raw_response": "", "error": str(exc), "prompts": []}


def save_optimizer_outputs(
    log: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    log_path: str | Path,
    summary_path: str | Path,
) -> pd.DataFrame:
    """Persist optimizer log and summary outputs."""
    log_path = Path(log_path)
    summary_path = Path(summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str) + "\n")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    return summary


def optimize_prompt(
    task_payloads: list[dict[str, str]],
    answer_sheet: pd.DataFrame,
    out_dir: str | Path,
    enable_llm_calls: bool,
    model_key: str,
    model_id: str,
    prompt_name: str,
    prompt_version: str,
    aws_profile_name: str,
    bedrock_region: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    iterations: int = 3,
    candidates_per_iteration: int = 3,
    minimum_improvement_threshold: float = 0.001,
    optimizer_max_tokens: int = 4096,
    optimizer_temperature: float = 0.7,
    score_metric: str = "f1",
    base_prompt_template: str | None = None,
    system_prompt: str | None = None,
    cache_max_entries: int = 500,
    request_sleep_seconds: float = 0.25,
    allowed_pmids: Iterable[str] | None = None,
    progress: bool = False,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Optimize the extraction prompt and return the best prompt, summary, and log."""
    out_dir = Path(out_dir)
    run_slug = f"{_slug(model_key)}_{_slug(prompt_version)}"
    log_path = out_dir / f"prompt_optimization_log_{run_slug}.json"
    summary_path = out_dir / f"prompt_optimization_summary_{run_slug}.csv"
    base_prompt_template = base_prompt_template or default_user_prompt_template()
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    log = {
        "model_key": model_key,
        "model_id": model_id,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "score_metric": score_metric,
        "iterations": iterations,
        "candidates_per_iteration": candidates_per_iteration,
        "minimum_improvement_threshold": minimum_improvement_threshold,
        "starting_prompt": base_prompt_template,
        "candidate_generation": [],
        "tested_prompts": [],
        "selection_history": [],
        "log_path": str(log_path),
        "summary_path": str(summary_path),
    }
    summary_rows = []

    best_result = test_prompt_template(
        base_prompt_template,
        "starting_prompt",
        0,
        "starting",
        task_payloads,
        answer_sheet,
        out_dir,
        enable_llm_calls,
        model_key,
        model_id,
        prompt_name,
        prompt_version,
        aws_profile_name,
        bedrock_region,
        max_tokens,
        temperature,
        cache_max_entries,
        request_sleep_seconds,
        allowed_pmids,
        progress,
        system_prompt,
        score_metric,
    )
    best_prompt_template = base_prompt_template
    best_score = best_result["score"]
    log["tested_prompts"].append(best_result)
    summary_rows.append(summary_row(best_result, score_metric, None, True))
    summary = save_optimizer_outputs(log, summary_rows, log_path, summary_path)

    for iteration in range(1, iterations + 1):
        best_score_before_iteration = best_score
        generation = generate_candidate_templates(
            best_prompt_template,
            best_score,
            score_metric,
            candidates_per_iteration,
            summary_rows,
            enable_llm_calls,
            model_id,
            aws_profile_name,
            bedrock_region,
            optimizer_max_tokens,
            optimizer_temperature,
        )
        candidate_templates = unique_valid_templates(
            generation["prompts"], best_prompt_template, candidates_per_iteration
        )
        used_fallback = False
        if not candidate_templates:
            candidate_templates = unique_valid_templates(
                fallback_candidate_templates(),
                best_prompt_template,
                candidates_per_iteration,
            )
            used_fallback = bool(candidate_templates)

        log["candidate_generation"].append(
            {
                "iteration": iteration,
                "raw_response": generation["raw_response"],
                "generation_error": generation["error"],
                "generated_candidate_prompts": generation["prompts"],
                "tested_candidate_prompts": candidate_templates,
                "used_fallback_candidate_prompts": used_fallback,
            }
        )

        iteration_results = []
        for candidate_number, candidate_template in enumerate(candidate_templates, start=1):
            result = test_prompt_template(
                candidate_template,
                f"iteration_{iteration}_candidate_{candidate_number}",
                iteration,
                "candidate",
                task_payloads,
                answer_sheet,
                out_dir,
                enable_llm_calls,
                model_key,
                model_id,
                prompt_name,
                prompt_version,
                aws_profile_name,
                bedrock_region,
                max_tokens,
                temperature,
                cache_max_entries,
                request_sleep_seconds,
                allowed_pmids,
                progress,
                system_prompt,
                score_metric,
            )
            iteration_results.append(result)
            log["tested_prompts"].append(result)
            summary_rows.append(summary_row(result, score_metric, best_score_before_iteration))
            save_optimizer_outputs(log, summary_rows, log_path, summary_path)

        best_candidate = max(iteration_results, key=lambda result: result["score"], default=None)
        if best_candidate and best_candidate["score"] > best_score + minimum_improvement_threshold:
            best_result = best_candidate
            best_prompt_template = best_candidate["prompt_template"]
            best_score = best_candidate["score"]

        log["selection_history"].append(
            {
                "iteration": iteration,
                "best_score_before_iteration": best_score_before_iteration,
                "selected_prompt_id": best_result["prompt_id"],
                "selected_score": best_score,
                "candidate_improved": best_score > best_score_before_iteration,
                "selected_prompt": best_prompt_template,
            }
        )
        for row in summary_rows:
            row["selected_after_iteration"] = row["prompt_id"] == best_result["prompt_id"]
        summary = save_optimizer_outputs(log, summary_rows, log_path, summary_path)

    return best_prompt_template, summary, log
