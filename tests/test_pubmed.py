from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from dgilit.pubmed import (
    PubMedClient,
    PubMedFetchConfig,
    fetch_pubmed_abstracts,
)


PUBMED_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <Abstract>
          <AbstractText Label="BACKGROUND">First section.</AbstractText>
          <AbstractText>Second section.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>456</PMID>
      <Article>
        <Abstract />
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class PubMedClientTest(unittest.TestCase):
    def test_fetch_abstract_texts_returns_notebook_friendly_mapping(self) -> None:
        client = PubMedClient(
            email="curator@example.org",
            batch_size=100,
            request_interval_seconds=0,
        )

        with patch.object(
            client,
            "_fetch_batch",
            return_value=ET.fromstring(PUBMED_XML),
        ) as fetch_batch:
            abstracts = client.fetch_abstract_texts(["123", "456", None, float("nan")])

        fetch_batch.assert_called_once_with(["123", "456"])
        self.assertEqual(
            abstracts,
            {
                "123": "BACKGROUND: First section.\nSecond section.",
                "456": None,
            },
        )

    def test_fetch_abstracts_returns_pydantic_models(self) -> None:
        config = PubMedFetchConfig(
            email="curator@example.org",
            request_interval_seconds=0,
        )
        client = PubMedClient(config)

        with patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)):
            abstracts = client.fetch_abstracts(["123"])

        abstract = abstracts["123"]
        self.assertEqual(abstract.pmid, "123")
        self.assertEqual(len(abstract.sections), 2)
        self.assertEqual(abstract.sections[0].label, "BACKGROUND")
        self.assertEqual(abstract.sections[0].text, "First section.")
        self.assertEqual(abstract.text, "BACKGROUND: First section.\nSecond section.")

    def test_batches_pmids_according_to_config(self) -> None:
        client = PubMedClient(batch_size=2, request_interval_seconds=0)

        with patch.object(
            client,
            "_fetch_batch",
            return_value=ET.fromstring("<PubmedArticleSet />"),
        ) as fetch_batch:
            client.fetch_abstracts(["1", "2", "3"])

        self.assertEqual(
            [call.args[0] for call in fetch_batch.call_args_list],
            [["1", "2"], ["3"]],
        )

    def test_convenience_function_matches_notebook_usage(self) -> None:
        with patch.object(
            PubMedClient,
            "fetch_abstract_texts",
            return_value={"123": "Example abstract."},
        ) as fetch_texts:
            abstracts = fetch_pubmed_abstracts(
                ["123"],
                email="curator@example.org",
            )

        fetch_texts.assert_called_once_with(["123"])
        self.assertEqual(abstracts["123"], "Example abstract.")


if __name__ == "__main__":
    unittest.main()
