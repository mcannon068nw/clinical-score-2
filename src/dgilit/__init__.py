"""
dgiLIT: AI-assisted literature prioritization and drug-gene interaction
curation for DGIdb.
"""

from .dgilit_models import (
    InteractionClassificationPrompt,
    InteractionClassificationResult,
)

from .tagging import (
    BioBertEntityTagger,
    CompositeEntityTagger,
    EntityMention,
    EntityPreTaggingService,
    InteractionExtractionInput,
    NormalizedConcept,
    PubTator3ChemicalTagger,
    SQLiteNormalizerCache,
    TaggedTextBlock,
    TaggerConfig,
    ViccNormalizer,
)

__version__ = "0.1.0"

__all__ = (
    "BioBertEntityTagger",
    "CompositeEntityTagger",
    "EntityMention",
    "EntityPreTaggingService",
    "InteractionClassificationPrompt",
    "InteractionClassificationResult",
    "InteractionExtractionInput",
    "NormalizedConcept",
    "PubTator3ChemicalTagger",
    "SQLiteNormalizerCache",
    "TaggedTextBlock",
    "TaggerConfig",
    "ViccNormalizer",
)