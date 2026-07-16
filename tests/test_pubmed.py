from __future__ import annotations

import urllib.error
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from dgilit.pubmed import (
    PubMedArticle,
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
        <Journal>
          <Title>Example Journal</Title>
          <JournalIssue>
            <PubDate>
              <Year>2026</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Example title.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">First section.</AbstractText>
          <AbstractText Label="RESULTS">Second section.</AbstractText>
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

ELINK_XML = """
<eLinkResult>
  <LinkSet>
    <DbFrom>pubmed</DbFrom>
    <IdList>
      <Id>123</Id>
    </IdList>
    <LinkSetDb>
      <DbTo>pmc</DbTo>
      <LinkName>pubmed_pmc</LinkName>
      <Link>
        <Id>999</Id>
      </Link>
    </LinkSetDb>
    <LinkSetDb>
      <DbTo>pmc</DbTo>
      <LinkName>pubmed_pmc_refs</LinkName>
      <Link>
        <Id>111</Id>
      </Link>
    </LinkSetDb>
  </LinkSet>
</eLinkResult>
"""

ID_CONVERTER_JSON = {
    "status": "ok",
    "records": [
        {
            "pmid": 37953380,
            "pmcid": "PMC10767982",
            "requested-id": "37953380",
        }
    ],
}

PMC_XML = """
<pmc-articleset>
  <article>
    <article-id pub-id-type="pmcid">PMC999</article-id>
    <front>
      <article-meta>
        <abstract id="abs1">
          <sec id="abs-results">
            <title>Results</title>
            <p>PMC abstract results.</p>
          </sec>
        </abstract>
      </article-meta>
    </front>
    <body>
      <sec id="s1">
        <title>Results</title>
        <p>Body results paragraph.</p>
        <sec id="s1-1">
          <title>Secondary analysis</title>
          <p>Nested text should not be duplicated in the parent.</p>
        </sec>
      </sec>
      <sec id="s2">
        <title>Discussion</title>
        <p>Body discussion paragraph.</p>
      </sec>
      <sec id="s3">
        <title>Conclusion</title>
        <p>Body conclusion paragraph.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>
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
                "123": "BACKGROUND: First section.\nRESULTS: Second section.",
                "456": None,
            },
        )

    def test_fetch_articles_returns_pydantic_models_with_taggable_sections(self) -> None:
        config = PubMedFetchConfig(
            email="curator@example.org",
            request_interval_seconds=0,
        )
        client = PubMedClient(config)

        with patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)):
            articles = client.fetch_articles(["123"], include_full_text=False)

        article = articles["123"]
        self.assertIsInstance(article, PubMedArticle)
        self.assertEqual(article.pmid, "123")
        self.assertEqual(article.title, "Example title.")
        self.assertEqual(article.journal, "Example Journal")
        self.assertEqual(article.publication_year, "2026")
        self.assertEqual(len(article.sections), 2)
        self.assertEqual(article.sections[0].section_type, "abstract")
        self.assertEqual(article.sections[0].label, "BACKGROUND")
        self.assertEqual(article.sections[0].text, "First section.")
        self.assertEqual(article.sections[1].label, "RESULTS")
        self.assertEqual(article.abstract_sections, article.sections)
        self.assertEqual(
            article.text,
            "BACKGROUND: First section.\nRESULTS: Second section.",
        )

    def test_fetch_articles_enriches_with_pmc_full_text_sections_by_default(self) -> None:
        client = PubMedClient(request_interval_seconds=0)

        with (
            patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)),
            patch.object(
                client,
                "_fetch_pmc_links",
                return_value={"123": "999"},
            ) as fetch_links,
            patch.object(
                client,
                "_fetch_pmc_batch",
                return_value=ET.fromstring(PMC_XML),
            ) as fetch_pmc,
        ):
            articles = client.fetch_articles(["123"])

        fetch_links.assert_called_once_with(["123", "456"])
        fetch_pmc.assert_called_once_with(["999"])

        article = articles["123"]
        self.assertEqual(article.pmcid, "PMC999")
        self.assertTrue(article.full_text_available)
        self.assertEqual(
            [(section.section_type, section.label, section.source) for section in article.sections],
            [
                ("abstract", "RESULTS", "pmc"),
                ("results", "RESULTS", "pmc"),
                ("body", "Secondary analysis", "pmc"),
                ("discussion", "DISCUSSION", "pmc"),
                ("conclusions", "CONCLUSIONS", "pmc"),
            ],
        )
        self.assertEqual(article.sections[1].text, "Body results paragraph.")
        self.assertEqual(article.sections[2].parent_label, "RESULTS")
        self.assertEqual(article.sections[4].section_id, "s3")

    def test_fetch_articles_can_skip_full_text_enrichment(self) -> None:
        client = PubMedClient(request_interval_seconds=0)

        with (
            patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)),
            patch.object(client, "_fetch_pmc_links") as fetch_links,
        ):
            article = client.fetch_articles(["123"], include_full_text=False)["123"]

        fetch_links.assert_not_called()
        self.assertIsNone(article.pmcid)
        self.assertFalse(article.full_text_available)
        self.assertEqual([section.source for section in article.sections], ["pubmed", "pubmed"])

    def test_fetch_articles_falls_back_to_pubmed_when_no_pmc_link_exists(self) -> None:
        client = PubMedClient(request_interval_seconds=0)

        with (
            patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)),
            patch.object(client, "_fetch_pmc_links", return_value={}),
            patch.object(client, "_fetch_pmc_batch") as fetch_pmc,
        ):
            article = client.fetch_articles(["123"])["123"]

        fetch_pmc.assert_not_called()
        self.assertIsNone(article.pmcid)
        self.assertFalse(article.full_text_available)
        self.assertEqual(article.sections[0].source, "pubmed")

    def test_fetch_articles_falls_back_to_pubmed_when_pmc_link_lookup_errors(self) -> None:
        client = PubMedClient(request_interval_seconds=0)
        error = urllib.error.HTTPError(
            url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        with (
            patch.object(client, "_fetch_batch", return_value=ET.fromstring(PUBMED_XML)),
            patch.object(client, "_fetch_pmc_links", side_effect=error),
            patch.object(client, "_fetch_pmc_batch") as fetch_pmc,
        ):
            article = client.fetch_articles(["123"])["123"]

        fetch_pmc.assert_not_called()
        self.assertIsNone(article.pmcid)
        self.assertFalse(article.full_text_available)
        self.assertEqual(article.sections[0].source, "pubmed")

    def test_parse_pmid_to_pmc_link_mapping_from_id_converter(self) -> None:
        client = PubMedClient(request_interval_seconds=0)

        with patch.object(
            client,
            "_get_json",
            return_value=ID_CONVERTER_JSON,
        ) as get_json:
            links = client._fetch_pmc_links(["37953380"])

        self.assertEqual(links, {"37953380": "10767982"})
        get_json.assert_called_once()
        self.assertEqual(get_json.call_args.args[1]["ids"], "37953380")

    def test_parse_pmid_to_pmc_link_mapping_falls_back_to_elink(self) -> None:
        client = PubMedClient(request_interval_seconds=0)

        with (
            patch.object(client, "_get_json", return_value={"records": []}),
            patch.object(client, "_get_xml", return_value=ET.fromstring(ELINK_XML)) as get_xml,
        ):
            links = client._fetch_pmc_links(["123", "456"])

        self.assertEqual(links, {"123": "999"})
        self.assertEqual(
            [call.args[1]["id"] for call in get_xml.call_args_list],
            ["123", "456"],
        )
        self.assertNotIn("111", links.values())

    def test_batches_pmids_according_to_config(self) -> None:
        client = PubMedClient(batch_size=2, request_interval_seconds=0)

        with patch.object(
            client,
            "_fetch_batch",
            return_value=ET.fromstring("<PubmedArticleSet />"),
        ) as fetch_batch:
            client.fetch_articles(["1", "2", "3"], include_full_text=False)

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
