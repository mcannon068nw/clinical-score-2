"""
Public API for dgilit.tagging.
"""

from .models import (
    EntityMention,
    InteractionExtractionInput,
    NormalizedConcept,
    TaggedTextBlock,
)

from .normalizers import (
    SQLiteNormalizerCache,
    ViccNormalizer,
)

from .service import EntityPreTaggingService

from .taggers import (
    BioBertEntityTagger,
    CompositeEntityTagger,
    PubTator3ChemicalTagger,
    TaggerConfig,
)

__all__ = (
    "BioBertEntityTagger",
    "CompositeEntityTagger",
    "EntityMention",
    "EntityPreTaggingService",
    "InteractionExtractionInput",
    "NormalizedConcept",
    "PubTator3ChemicalTagger",
    "SQLiteNormalizerCache",
    "TaggedTextBlock",
    "TaggerConfig",
    "ViccNormalizer",
)