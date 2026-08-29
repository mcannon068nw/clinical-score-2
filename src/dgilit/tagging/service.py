"""Orchestration service for fast dgiLIT pre-tagging and normalization."""

from __future__ import annotations

from .models import EntityMention, TaggedTextBlock
from .normalizers import ViccNormalizer
from .taggers import BioBertEntityTagger, TaggerConfig

from tqdm.auto import tqdm


class EntityPreTaggingService:
    """Batch tag text blocks, normalize unique mentions, and return rich blocks."""

    def __init__(
        self,
        tagger: BioBertEntityTagger | None = None,
        normalizer: ViccNormalizer | None = None,
        keep_ungrounded: bool = True,
    ) -> None:
        self.tagger = tagger or BioBertEntityTagger(TaggerConfig(include_diseases=False))
        self.normalizer = normalizer or ViccNormalizer()
        self.keep_ungrounded = keep_ungrounded

    def _safe_text(value: object) -> str:
        if value is None:
            return ""
        return str(value)

    def tag_blocks(
        self,
        contexts: list[str],
        pmids: list[str | None] | None = None,
        block_ids: list[str | None] | None = None,
    ) -> list[TaggedTextBlock]:
        pmids = pmids or [None] * len(contexts)
        block_ids = block_ids or [None] * len(contexts)

        if not (len(contexts) == len(pmids) == len(block_ids)):
            raise ValueError("contexts, pmids, and block_ids must have the same length")

        contexts = ["" if context is None else str(context) for context in contexts]
        pmids = [None if pmid is None else str(pmid) for pmid in pmids]
        block_ids = [None if block_id is None else str(block_id) for block_id in block_ids]

        mentions_by_context = self.tagger.tag_texts(contexts)

        unique_strings: set[tuple[str, str]] = set()
        for mentions in mentions_by_context:
            for mention in mentions:
                unique_strings.add((mention.entity_type, mention.text))

        normalized = self.normalizer.normalize_many_unique(unique_strings)

        blocks: list[TaggedTextBlock] = []
        for context, pmid, block_id, mentions in zip(contexts, pmids, block_ids, mentions_by_context):
            enriched_mentions: list[EntityMention] = []
            for mention in mentions:
                enriched = mention.model_copy(
                    update={"concept": normalized.get((mention.entity_type, mention.text))}
                )
                if self.keep_ungrounded or enriched.is_grounded:
                    enriched_mentions.append(enriched)
            blocks.append(
                TaggedTextBlock(
                    context=context,
                    pmid=pmid,
                    block_id=block_id,
                    entities=enriched_mentions,
                )
            )
        return blocks

    def tag_block(
        self,
        context: str,
        pmid: str | None = None,
        block_id: str | None = None,
    ) -> TaggedTextBlock:
        return self.tag_blocks([context], [pmid], [block_id])[0]
