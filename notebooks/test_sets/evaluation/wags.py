"""WAGS-backed interaction extraction helpers for the evaluation notebook."""

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from notebooks.test_sets.evaluation.utils import clean_value, collapse_ws

logger = logging.getLogger(__name__)

PROMPT_PMID_TOKEN = "{{PMID}}"
PROMPT_TITLE_TOKEN = "{{ARTICLE_TITLE}}"
PROMPT_ABSTRACT_TOKEN = "{{ABSTRACT}}"
DEFAULT_SYSTEM_PROMPT = (
    "Extract drug-gene interaction pairs from the supplied PubMed title "
    "and abstract. Return only JSON matching the schema."
)


class ExtractedInteraction(BaseModel):
    """One extracted drug-gene pair."""

    model_config = ConfigDict(extra="forbid")

    drug_name: str
    gene_name: str


class InteractionExtractionResponse(BaseModel):
    """Structured response containing extracted interactions."""

    model_config = ConfigDict(extra="forbid")

    interactions: list[ExtractedInteraction]


def default_user_prompt_template() -> str:
    """Return the default extraction prompt with payload placeholders."""
    lines = [
        "Return {\"interactions\": []} if no drug-gene interaction pair is "
        "mentioned or supported by the abstract.",
        "Each interaction object must include only drug_name and gene_name.",
        "Use the drug and gene names as written in the title or abstract "
        "when possible.",
        "Do not infer interactions from pathway context alone.",
        "",
        f"PMID: {PROMPT_PMID_TOKEN}",
        f"Article title: {PROMPT_TITLE_TOKEN}",
        "",
        "Abstract:",
        PROMPT_ABSTRACT_TOKEN,
    ]
    return "\n".join(lines).strip()


def render_prompt_template(
    prompt_template: str,
    payload: Mapping[str, Any],
) -> str:
    """Fill a prompt template with a PubMed task payload."""
    return (
        prompt_template.replace(PROMPT_PMID_TOKEN, clean_value(payload.get("pmid")))
        .replace(PROMPT_TITLE_TOKEN, collapse_ws(payload.get("article_title", "")))
        .replace(PROMPT_ABSTRACT_TOKEN, str(payload.get("abstract") or "").strip())
        .strip()
    )


def make_prompt_template(
    prompt_name: str,
    prompt_version: str,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> Any:
    """Build a WAGS prompt template for PubMed extraction tasks."""
    from wags_llm.prompts import BasePromptTemplate

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or default_user_prompt_template()

    class Prompt(BasePromptTemplate):
        name = prompt_name
        version = prompt_version

        def build_system_prompt(self) -> str:
            return system_prompt

        def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
            return render_prompt_template(user_prompt_template, payload)

    return Prompt()


def make_prompt_registry(
    prompt_name: str,
    prompt_version: str,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> Any:
    """Register and return the evaluation prompt template."""
    from wags_llm.prompts import PromptRegistry

    registry = PromptRegistry()
    registry.register(
        make_prompt_template(
            prompt_name,
            prompt_version,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
    )
    return registry


def initialize_wags_client(
    model_id: str,
    bedrock_region: str,
    aws_profile_name: str,
    max_tokens: int,
    temperature: float,
) -> Any:
    """Create the Bedrock-backed WAGS JSON client."""
    from wags_llm.client import BedrockClaudeJsonClient

    return BedrockClaudeJsonClient(
        model_id=model_id,
        region_name=bedrock_region,
        profile_name=aws_profile_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def make_wags_runner(
    client: Any,
    prompt_registry: Any,
    cache_max_entries: int,
) -> Any:
    """Create a structured WAGS task runner with an in-memory cache."""
    from wags_llm.cache import InMemoryCache
    from wags_llm.services import StructuredTaskRunner

    cache = InMemoryCache(max_entries=cache_max_entries)
    return StructuredTaskRunner(
        client=client, prompt_registry=prompt_registry, cache=cache
    )


def execute_wags_task(
    runner: Any,
    payload: Mapping[str, Any],
    prompt_name: str,
    prompt_version: str,
) -> InteractionExtractionResponse:
    """Execute one WAGS extraction task and validate the response model."""
    result = runner.execute(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        payload=payload,
        response_model=InteractionExtractionResponse,
    )
    return InteractionExtractionResponse.model_validate(result)


def interaction_rows(
    response: InteractionExtractionResponse, pmid: str
) -> list[dict[str, str]]:
    """Convert a structured WAGS response into filtered interaction rows."""
    rows = []
    seen = set()
    pmid = clean_value(pmid)
    for item in response.interactions:
        row = {
            "pmid": pmid,
            "drug_name": clean_value(item.drug_name),
            "gene_name": clean_value(item.gene_name),
        }
        key = tuple(row.values())
        if all(row.values()) and key not in seen:
            rows.append(row)
            seen.add(key)
    return rows


def load_predictions(path: str | Path) -> dict[str, Any]:
    """Load saved prediction records from JSON if present."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save_predictions(predictions: dict[str, Any], path: str | Path) -> Path:
    """Persist prediction records as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return path


def prediction_status_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    """Summarize prediction success, interaction counts, and errors."""
    return pd.DataFrame(
        [
            {
                "pmid": pmid,
                "successful": isinstance(payload, dict)
                and not payload.get("error")
                and "interactions" in payload,
                "parsed_interactions": len(payload.get("interactions", []))
                if isinstance(payload, dict)
                else 0,
                "error": payload.get("error", "")
                if isinstance(payload, dict)
                else "Invalid prediction record",
            }
            for pmid, payload in sorted(predictions.items())
        ]
    )


def _prediction_metadata(
    model_key: str, model_id: str, prompt_name: str, prompt_version: str
) -> dict[str, str]:
    return {
        "model_key": model_key,
        "model_id": model_id,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
    }


def run_wags_predictions(
    payloads: list[dict[str, str]],
    predictions_path: str | Path,
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
    overwrite: bool = False,
    stop_on_error: bool = False,
    request_sleep_seconds: float = 0.25,
    allowed_pmids: Iterable[str] | None = None,
    progress: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> dict[str, Any]:
    """Run or resume WAGS predictions for task payloads."""
    predictions = load_predictions(predictions_path)
    selected_pmids = (
        [payload["pmid"] for payload in payloads]
        if allowed_pmids is None
        else allowed_pmids
    )
    allowed = {clean_value(pmid) for pmid in selected_pmids}

    if not enable_llm_calls:
        logger.info("ENABLE_LLM_CALLS is False. No Bedrock calls were made.")
        return predictions

    client = initialize_wags_client(
        model_id, bedrock_region, aws_profile_name, max_tokens, temperature
    )
    registry = make_prompt_registry(
        prompt_name,
        prompt_version,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )
    runner = make_wags_runner(client, registry, cache_max_entries)
    metadata = _prediction_metadata(model_key, model_id, prompt_name, prompt_version)

    for index, payload in enumerate(payloads, start=1):
        pmid = clean_value(payload["pmid"])
        if pmid not in allowed:
            continue
        if (
            pmid in predictions
            and "interactions" in predictions[pmid]
            and not overwrite
        ):
            if progress:
                logger.info(
                    "Skipping cached WAGS prediction %s of %s", index, len(payloads)
                )
            continue
        if progress:
            logger.info("Running WAGS prediction %s of %s", index, len(payloads))

        try:
            parsed = execute_wags_task(runner, payload, prompt_name, prompt_version)
            predictions[pmid] = {
                **metadata,
                "interactions": interaction_rows(parsed, pmid),
            }
        except Exception as exc:
            predictions[pmid] = {**metadata, "error": str(exc)}
            logger.exception("WAGS prediction failed for PMID %s", pmid)
            if stop_on_error:
                raise

        save_predictions(predictions, predictions_path)
        if request_sleep_seconds:
            time.sleep(request_sleep_seconds)
    return predictions


def raw_audit_wags_response(
    payload: Mapping[str, Any],
    client: Any,
    prompt_registry: Any,
    prompt_name: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Return raw and parsed WAGS output for one audit payload."""
    prompt = prompt_registry.get(prompt_name, prompt_version)
    response = client.invoke_json(
        system_prompt=prompt.build_system_prompt(),
        user_prompt=prompt.build_user_prompt(payload),
        json_schema=InteractionExtractionResponse.model_json_schema(),
    )
    parsed = InteractionExtractionResponse.model_validate(response.parsed_json)
    pmid = str(payload.get("pmid", ""))
    return {
        "pmid": pmid,
        "raw_response": response.raw_text,
        "parsed_json": response.parsed_json,
        "interactions": interaction_rows(parsed, pmid),
    }
