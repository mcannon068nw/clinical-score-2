from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from wags_llm.prompts import BasePromptTemplate


class InteractionClassificationResult(BaseModel):
    """LLM result for classifying one candidate drug-gene pair."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    drug: str
    gene: str
    interaction: bool
    evidence: str | None = None
    interaction_type: str | None = None
    directionality: str | None = None
    error_message: str | None = None


class RunResult:
    temperature: float
    run_idx: int
    pass


PROMPT_NAME = "dgilit_interaction_classification"


@dataclass
class InteractionClassificationPrompt(BasePromptTemplate):
    """Prompt for classifying whether one drug-gene pair is supported by text."""

    name = PROMPT_NAME
    PROMPT_DIR = Path(__file__).parent / "dgilit_prompts"

    def __init__(self, version: str):
        self.version = version

    @property
    def prompt_path(self) -> Path:
        return self.PROMPT_DIR / f"{self.version}.text"

    def build_system_prompt(self) -> str:
        if not self.prompt_path.exists():
            available_versions = sorted(
                p.stem for p in self.PROMPT_DIR.glob("*.text")
            )

            raise FileNotFoundError(
                f"Prompt version '{self.version}' not found. "
                f"Available versions: {available_versions}"
            )

        return self.prompt_path.read_text()

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        return (
            "Candidate Drug:\n"
            f"{payload['candidate_drug']}\n\n"
            "Candidate Gene:\n"
            f"{payload['candidate_gene']}\n\n"
            "Context:\n"
            f"{payload['context']}\n"
        )

    def build_payload(
        self,
        context: str,
        candidate_drug: str,
        candidate_gene: str,
    ) -> dict[str, Any]:
        return {
            "context": context,
            "candidate_drug": candidate_drug,
            "candidate_gene": candidate_gene,
        }