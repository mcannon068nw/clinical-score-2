"""Batch NER taggers for dgiLIT.

The HuggingFace pipelines are lazy-loaded and called in batches. Inputs are
sanitized so missing abstracts, NaN values, and non-string objects do not crash
transformers token-classification pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Iterable, Any
from tqdm.auto import tqdm

from .models import EntityMention, EntityType


@dataclass(frozen=True)
class TaggerConfig:
    batch_size: int = 16
    include_drugs: bool = True
    include_genes: bool = True
    include_diseases: bool = False
    aggregation_strategy: str = "first"
    device: int = -1
    num_workers: int = 0


class BioBertEntityTagger:
    """Lazy-loaded BioBERT entity tagger compatible with old novel.py models."""

    MODEL_BY_TYPE: dict[EntityType, str] = {
        "gene": "alvaroalon2/biobert_genetic_ner",
        "drug": "alvaroalon2/biobert_chemical_ner",
        "disease": "alvaroalon2/biobert_diseases_ner",
    }

    def __init__(self, config: TaggerConfig | None = None) -> None:
        self.config = config or TaggerConfig()
        self._pipelines = {}

    def tag_texts(
        self,
        texts: list[Any],
        pmids: list[Any] | None = None,
    ) -> list[list[EntityMention]]:
        """Return entity mentions for each input text, preserving input order.

        Blank or invalid inputs return an empty mention list. This avoids the
        common transformers error: "sentence must be an untokenized string".
        """
        clean_texts = [_coerce_text(text) for text in texts]
        mentions_by_text: list[list[EntityMention]] = [[] for _ in clean_texts]

        # Only send non-empty strings into HuggingFace. Keep original positions.
        valid_items = [(idx, text) for idx, text in enumerate(clean_texts) if text.strip()]
        if not valid_items:
            return mentions_by_text

        for entity_type in self._enabled_types():
            pipe = self._get_pipeline(entity_type)
            for offset in tqdm(
                range(0, len(valid_items), self.config.batch_size),
                total=(len(valid_items) + self.config.batch_size - 1) // self.config.batch_size,
                desc="Tagging text batches",
            ):                
                batch_items = valid_items[offset : offset + self.config.batch_size]
                batch_indices = [idx for idx, _ in batch_items]
                batch_texts = [text for _, text in batch_items]

                raw_batch = pipe(
                    batch_texts,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                )

                # transformers returns list[dict] for one string, list[list[dict]] for many strings.
                if batch_texts and raw_batch and isinstance(raw_batch[0], dict):
                    raw_batch = [raw_batch]

                for original_idx, raw_mentions in zip(batch_indices, raw_batch):
                    mentions_by_text[original_idx].extend(
                        self._convert_mentions(raw_mentions, entity_type)
                    )
        return mentions_by_text

    def tag_text(self, text: str, pmid: Any | None = None) -> list[EntityMention]:
        return self.tag_texts([text], pmids=[pmid] if pmid is not None else None)[0]

    def _enabled_types(self) -> Iterable[EntityType]:
        if self.config.include_drugs:
            yield "drug"
        if self.config.include_genes:
            yield "gene"
        if self.config.include_diseases:
            yield "disease"

    def _get_pipeline(self, entity_type: EntityType):
        if entity_type not in self._pipelines:
            from transformers import pipeline

            self._pipelines[entity_type] = pipeline(
                "token-classification",
                model=self.MODEL_BY_TYPE[entity_type],
                aggregation_strategy=self.config.aggregation_strategy,
                device=self.config.device,
            )
        return self._pipelines[entity_type]

    @staticmethod
    def _convert_mentions(raw_mentions: list[dict], entity_type: EntityType) -> list[EntityMention]:
        converted: list[EntityMention] = []
        for raw in raw_mentions or []:
            entity_group = str(raw.get("entity_group", ""))
            if entity_group == "0":
                continue
            word = raw.get("word") or raw.get("text")
            if not word:
                continue
            converted.append(
                EntityMention(
                    text=str(word),
                    entity_type=entity_type,
                    start=int(raw.get("start", 0)),
                    end=int(raw.get("end", 0)),
                    score=raw.get("score"),
                    source=f"biobert:{BioBertEntityTagger.MODEL_BY_TYPE[entity_type]}",
                )
            )
        return converted


def _coerce_text(value: Any) -> str:
    """Convert dataframe values into safe text for token-classification."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        try:
            if isnan(value):
                return ""
        except TypeError:
            pass
    # pandas.NA / numpy.nan often behave badly in boolean contexts.
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import EntityMention


@dataclass(frozen=True)
class PubTatorChemicalMention:
    pmid: str
    text: str
    concept_id: str | None = None


class PubTator3ChemicalTagger:
    """Tag drug candidates from precomputed PubTator3 chemical annotations.

    This is intended as an additive candidate source, not a replacement for
    BioBERT. It returns EntityMention objects with entity_type='drug'.
    """

    def __init__(
        self,
        chemicals_by_pmid: dict[str, list[PubTatorChemicalMention]],
        *,
        source: str = "pubtator3:chemical",
        case_sensitive: bool = False,
    ) -> None:
        self.chemicals_by_pmid = chemicals_by_pmid
        self.source = source
        self.case_sensitive = case_sensitive

    def tag_texts(
        self,
        texts: list[Any],
        pmids: list[Any] | None = None,
    ) -> list[list[EntityMention]]:
        """Return PubTator3 chemical mentions for each input text.

        `pmids` is optional for API compatibility, but PubTator3 matching needs
        PMID context. If omitted, all outputs are empty.
        """
        if pmids is None:
            return [[] for _ in texts]

        mentions_by_text: list[list[EntityMention]] = []

        for text_value, pmid_value in zip(texts, pmids, strict=False):
            text = _coerce_text(text_value)
            pmid = str(pmid_value) if pmid_value is not None else ""

            if not text.strip() or not pmid:
                mentions_by_text.append([])
                continue

            mentions_by_text.append(self._tag_text_for_pmid(text, pmid))

        # Preserve length even if pmids was shorter than texts.
        while len(mentions_by_text) < len(texts):
            mentions_by_text.append([])

        return mentions_by_text

    def tag_text(self, text: str, pmid: str | None = None) -> list[EntityMention]:
        if pmid is None:
            return []
        return self.tag_texts([text], [pmid])[0]

    def _tag_text_for_pmid(self, text: str, pmid: str) -> list[EntityMention]:
        candidates = self.chemicals_by_pmid.get(str(pmid), [])
        if not candidates:
            return []

        search_text = text if self.case_sensitive else text.lower()
        converted: list[EntityMention] = []
        seen: set[tuple[str, int, int]] = set()

        for candidate in candidates:
            mention_text = candidate.text.strip()
            if not mention_text:
                continue

            needle = mention_text if self.case_sensitive else mention_text.lower()
            start = 0

            while True:
                idx = search_text.find(needle, start)
                if idx == -1:
                    break

                end = idx + len(mention_text)
                key = (mention_text.lower(), idx, end)

                if key not in seen:
                    seen.add(key)
                    converted.append(
                        EntityMention(
                            text=text[idx:end],
                            entity_type="drug",
                            start=idx,
                            end=end,
                            score=None,
                            source=self.source,
                        )
                    )

                start = end

        return converted

    @classmethod
    def from_pubtator_file(cls, path: str | Path) -> "PubTator3ChemicalTagger":
        """Load PubTator-style chemical annotations.

        Expected annotation rows resemble:
            PMID<TAB>start<TAB>end<TAB>mention<TAB>Chemical<TAB>identifier

        Non-chemical rows and title/abstract rows are ignored.
        """
        chemicals_by_pmid: dict[str, list[PubTatorChemicalMention]] = defaultdict(list)

        with Path(path).open() as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line or "|t|" in line or "|a|" in line:
                    continue

                parts = line.split("\t")
                if len(parts) < 6:
                    continue

                pmid, _start, _end, mention, entity_type, concept_id = parts[:6]

                if entity_type.lower() != "chemical":
                    continue

                chemicals_by_pmid[pmid].append(
                    PubTatorChemicalMention(
                        pmid=pmid,
                        text=mention,
                        concept_id=concept_id or None,
                    )
                )

        return cls(dict(chemicals_by_pmid))
    
class CompositeEntityTagger:
    def __init__(self, *taggers):
        self.taggers = taggers

    def tag_texts(
        self,
        texts: list[Any],
        pmids: list[Any] | None = None,
    ) -> list[list[EntityMention]]:
        results: list[list[EntityMention]] = [[] for _ in texts]

        for tagger in self.taggers:
            tagged = tagger.tag_texts(texts, pmids=pmids)

            for i, mentions in enumerate(tagged):
                results[i].extend(mentions)

        return results

    def tag_text(
        self,
        text: Any,
        pmid: Any | None = None,
    ) -> list[EntityMention]:
        return self.tag_texts(
            [text],
            pmids=[pmid] if pmid is not None else None,
        )[0]