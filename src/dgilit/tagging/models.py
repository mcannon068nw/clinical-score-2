"""Pydantic models for dgiLIT entity pre-tagging.

These models are intentionally small and serializable so they can be passed
through WAGS prompt payloads, cached, written to parquet/jsonl, and reused for
post-LLM validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EntityType = Literal["drug", "gene", "disease"]


class NormalizedConcept(BaseModel):
    """Normalized biomedical concept returned by a concept normalizer."""

    model_config = ConfigDict(extra="forbid")

    concept_id: str | None = None
    concept_label: str | None = None
    match_type: int | str | None = None
    normalizer: str | None = None
    raw_response: dict[str, Any] | None = None
    error_message: str | None = None

    @property
    def is_grounded(self) -> bool:
        """Whether the mention normalized to a usable concept identifier."""
        return bool(self.concept_id) and not self.error_message


class EntityMention(BaseModel):
    """Single entity mention found in a text block."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    entity_type: EntityType
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    source: str = "biobert"
    concept: NormalizedConcept | None = None

    @field_validator("end")
    @classmethod
    def end_must_be_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("end must be nonnegative")
        return value

    @property
    def concept_id(self) -> str | None:
        return self.concept.concept_id if self.concept else None

    @property
    def concept_label(self) -> str | None:
        return self.concept.concept_label if self.concept else None

    @property
    def is_grounded(self) -> bool:
        return bool(self.concept and self.concept.is_grounded)

    def as_prompt_candidate(self) -> str:
        """Compact representation for LLM prompts."""
        label = self.concept_label or self.text
        identifier = self.concept_id or "un-normalized"
        return f"{label} | {identifier} | mention: {self.text} | span: {self.start}-{self.end}"


class TaggedTextBlock(BaseModel):
    """A text block plus all pre-tagged entity mentions."""

    model_config = ConfigDict(extra="forbid")

    context: str
    pmid: str | None = None
    block_id: str | None = None
    entities: list[EntityMention] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def drugs(self) -> list[EntityMention]:
        return [e for e in self.entities if e.entity_type == "drug"]

    @property
    def genes(self) -> list[EntityMention]:
        return [e for e in self.entities if e.entity_type == "gene"]

    @property
    def diseases(self) -> list[EntityMention]:
        return [e for e in self.entities if e.entity_type == "disease"]

    @property
    def grounded_drugs(self) -> list[EntityMention]:
        return [e for e in self.drugs if e.is_grounded]

    @property
    def grounded_genes(self) -> list[EntityMention]:
        return [e for e in self.genes if e.is_grounded]

    @property
    def should_run_llm(self) -> bool:
        """Cheap gate to prevent useless LLM calls."""
        return bool(self.grounded_drugs and self.grounded_genes)

    @property
    def skip_reason(self) -> str | None:
        if self.should_run_llm:
            return None
        if not self.grounded_drugs and not self.grounded_genes:
            return "No normalized drug or gene candidates found."
        if not self.grounded_drugs:
            return "No normalized drug candidates found."
        return "No normalized gene candidates found."

    def prompt_entity_block(self, include_diseases: bool = False) -> str:
        """Render normalized candidates in a compact prompt-friendly format."""
        sections: list[str] = []
        sections.append(_render_section("DRUGS", self.grounded_drugs))
        sections.append(_render_section("GENES", self.grounded_genes))
        if include_diseases:
            sections.append(_render_section("DISEASES", [e for e in self.diseases if e.is_grounded]))
        return "\n".join(sections)


def _render_section(title: str, mentions: list[EntityMention]) -> str:
    if not mentions:
        return f"{title}: none"
    unique: dict[tuple[str | None, str], EntityMention] = {}
    for mention in mentions:
        key = (mention.concept_id, mention.text.lower())
        unique.setdefault(key, mention)
    lines = [f"{title}:"]
    lines.extend(f"- {m.as_prompt_candidate()}" for m in unique.values())
    return "\n".join(lines)


class InteractionExtractionInput(BaseModel):
    """LLM input object after deterministic pre-processing."""

    model_config = ConfigDict(extra="forbid")

    context: str
    pmid: str | None = None
    block_id: str | None = None
    tagged_entities: list[EntityMention] = Field(default_factory=list)
    run_llm: bool = True
    skip_reason: str | None = None

    @classmethod
    def from_tagged_block(cls, block: TaggedTextBlock) -> "InteractionExtractionInput":
        return cls(
            context=block.context,
            pmid=block.pmid,
            block_id=block.block_id,
            tagged_entities=block.entities,
            run_llm=block.should_run_llm,
            skip_reason=block.skip_reason,
        )
