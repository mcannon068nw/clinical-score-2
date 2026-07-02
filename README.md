# dgiLIT

Pre-print: https://www.biorxiv.org/content/10.64898/2026.01.16.699733v1

dgiLIT (Drug-Gene Interaction Literature Integration Tool) is a Python package for prioritizing biomedical literature and performing AI-assisted curation of drug–gene interactions for integration into the Drug-Gene Interaction Database (DGIdb).

The package combines lightweight natural language processing, named-entity recognition, biomedical concept normalization, and large language models (LLMs) to transform unstructured biomedical literature into structured, reviewable interaction data.

## Overview
The Drug-Gene Interaction Database (DGIdb) has a long history of curating drug–gene interactions from the biomedical literature. As the volume of published research continues to grow, maintaining comprehensive and up-to-date interaction data has become increasingly challenging.

dgiLIT addresses this problem by providing a modular pipeline that:

- identifies candidate drug and gene entities,
- normalizes entities to biomedical ontologies,
- extracts interactions using large language models, and
- produces structured outputs suitable for expert human review and downstream integration into DGIdb.

The workflow is designed to reduce the amount of literature requiring manual inspection while maintaining human oversight of the final curated interactions.

## Installation

Install the latest release from PyPI:

```bash
pip install dgilit
```

Or install the development version from source:

```bash
git clone https://github.com/dgidb/dgiLIT.git
cd dgiLIT
pip install -e .
```

For entity pre-tagging with PubTator3, download the required PubTator3 annotation files and configure the path in your environment or notebook.

---

## Example

The example below demonstrates the entity pre-tagging workflow, which identifies and normalizes candidate drug and gene mentions from biomedical text.

```python
from dgilit import (
    BioBertEntityTagger,
    CompositeEntityTagger,
    EntityPreTaggingService,
    PubTator3ChemicalTagger,
    SQLiteNormalizerCache,
    TaggerConfig,
    ViccNormalizer,
)

PUBTATOR_PATH = "/path/to/chemical2pubtator3"

# Configure entity taggers
tagger = CompositeEntityTagger(
    BioBertEntityTagger(
        TaggerConfig(
            batch_size=16,
            include_drugs=True,
            include_genes=True,
            include_diseases=False,
            device=-1,
        )
    ),
    PubTator3ChemicalTagger.from_pubtator_file(PUBTATOR_PATH),
)

# Configure concept normalization
pretagger = EntityPreTaggingService(
    tagger=tagger,
    normalizer=ViccNormalizer(
        cache=SQLiteNormalizerCache(".dgilit_normalizer_cache.sqlite")
    ),
)

text = """
Venetoclax inhibits BCL2 and induces apoptosis in leukemia cells.
"""

# Identify and normalize entities
result = pretagger.tag_block(text)

# Inspect grounded entities
for entity in result.entities:
    print(entity)

# LLM Curation Task of Choice
```

For complete examples including literature retrieval, AI-assisted interaction classification, and evaluation workflows, see the notebooks in the `notebooks/` directory.

