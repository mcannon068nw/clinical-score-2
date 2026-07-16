"""Utilities for fetching PubMed article records by PMID."""

from __future__ import annotations

import json
import math
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

SectionSource = Literal["pubmed", "pmc"]
BEST_EFFORT_PMC_ERRORS = (
    ET.ParseError,
    http.client.IncompleteRead,
    urllib.error.HTTPError,
    urllib.error.URLError,
)


class PubMedFetchConfig(BaseModel):
    """Configuration for PubMed E-utilities article fetches."""

    model_config = ConfigDict(extra="forbid")

    email: str | None = None
    batch_size: int = Field(default=100, ge=1)
    request_interval_seconds: float = Field(default=0.34, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    base_url: str = PUBMED_EFETCH_URL
    elink_url: str = NCBI_ELINK_URL
    id_converter_url: str = PMC_ID_CONVERTER_URL


class PubMedArticleSection(BaseModel):
    """One taggable section of a PubMed article record."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    section_type: str = Field(default="abstract", min_length=1)
    label: str | None = None
    section_id: str | None = None
    parent_label: str | None = None
    source: SectionSource = "pubmed"

    def render(self) -> str:
        if self.label:
            return f"{self.label}: {self.text}"
        return self.text


class PubMedArticle(BaseModel):
    """Structured PubMed article record returned for one PMID.

    PubMed EFetch returns citation metadata and abstract content. When a
    linked PMC record is available, this model can also include parsed
    full-text sections from PMC XML.
    """

    model_config = ConfigDict(extra="forbid")

    pmid: str = Field(..., min_length=1)
    pmcid: str | None = None
    title: str | None = None
    journal: str | None = None
    publication_year: str | None = None
    full_text_available: bool = False
    sections: list[PubMedArticleSection] = Field(default_factory=list)

    @field_validator("pmid", mode="before")
    @classmethod
    def normalize_pmid(cls, value: Any) -> str:
        return str(value).strip()

    @property
    def abstract_sections(self) -> list[PubMedArticleSection]:
        return [section for section in self.sections if section.section_type == "abstract"]

    @property
    def text(self) -> str | None:
        if not self.sections:
            return None
        return "\n".join(section.render() for section in self.sections)


class PubMedClient:
    """Small PubMed EFetch client for retrieving article records."""

    def __init__(self, config: PubMedFetchConfig | None = None, **kwargs: Any) -> None:
        if config and kwargs:
            raise ValueError("Pass either config or keyword options, not both")
        self.config = config or PubMedFetchConfig(**kwargs)

    def fetch_articles(
        self,
        pmids: Iterable[Any],
        include_full_text: bool = True,
    ) -> dict[str, PubMedArticle]:
        """Fetch structured PubMed article records keyed by PMID."""
        normalized_pmids = [_normalize_pmid(pmid) for pmid in pmids]
        normalized_pmids = [pmid for pmid in normalized_pmids if pmid is not None]

        articles: dict[str, PubMedArticle] = {}
        for batch in _batched(normalized_pmids, self.config.batch_size):
            root = self._fetch_batch(batch)
            for article_xml in root.findall(".//PubmedArticle"):
                article = _parse_pubmed_article(article_xml)
                if article:
                    articles[article.pmid] = article

            if self.config.request_interval_seconds:
                time.sleep(self.config.request_interval_seconds)

        if include_full_text and articles:
            self._enrich_with_pmc_full_text(articles)

        return articles

    def fetch_abstracts(
        self,
        pmids: Iterable[Any],
        include_full_text: bool = True,
    ) -> dict[str, PubMedArticle]:
        """Fetch structured PubMed article records keyed by PMID.

        Kept for compatibility with earlier abstract-focused usage.
        """
        return self.fetch_articles(pmids, include_full_text=include_full_text)

    def fetch_article_texts(
        self,
        pmids: Iterable[Any],
        include_full_text: bool = True,
    ) -> dict[str, str | None]:
        """Fetch rendered article section text keyed by PMID."""
        return {
            pmid: article.text
            for pmid, article in self.fetch_articles(
                pmids,
                include_full_text=include_full_text,
            ).items()
        }

    def fetch_abstract_texts(self, pmids: Iterable[Any]) -> dict[str, str | None]:
        """Fetch rendered abstract text keyed by PMID."""
        return self.fetch_article_texts(pmids, include_full_text=False)

    def _fetch_batch(self, pmids: list[str]) -> ET.Element:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.config.email:
            params["email"] = self.config.email

        url = f"{self.config.base_url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "dgilit/0.1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return ET.fromstring(response.read())

    def _enrich_with_pmc_full_text(self, articles: dict[str, PubMedArticle]) -> None:
        try:
            pmid_to_pmc_uid = self._fetch_pmc_links(list(articles))
        except BEST_EFFORT_PMC_ERRORS:
            return
        if not pmid_to_pmc_uid:
            return

        pmc_uid_to_pmid = {pmc_uid: pmid for pmid, pmc_uid in pmid_to_pmc_uid.items()}
        for batch in _batched(list(pmc_uid_to_pmid), self.config.batch_size):
            try:
                root = self._fetch_pmc_batch(batch)
            except BEST_EFFORT_PMC_ERRORS:
                continue
            for article_xml in root.findall(".//article"):
                pmc_uid = _extract_pmc_uid(article_xml)
                if not pmc_uid:
                    continue
                pmid = pmc_uid_to_pmid.get(pmc_uid)
                if not pmid or pmid not in articles:
                    continue
                article = articles[pmid]
                sections = _parse_pmc_sections(article_xml)
                if not any(section.section_type == "abstract" for section in sections):
                    sections = article.abstract_sections + sections
                if not sections:
                    articles[pmid] = article.model_copy(
                        update={"pmcid": f"PMC{pmc_uid}"}
                    )
                    continue
                articles[pmid] = article.model_copy(
                    update={
                        "pmcid": f"PMC{pmc_uid}",
                        "full_text_available": True,
                        "sections": sections,
                    }
                )

            if self.config.request_interval_seconds:
                time.sleep(self.config.request_interval_seconds)

    def _fetch_pmc_links(self, pmids: list[str]) -> dict[str, str]:
        links = self._fetch_pmc_links_from_id_converter(pmids)
        if links:
            return links
        return self._fetch_pmc_links_from_elink(pmids)

    def _fetch_pmc_links_from_id_converter(self, pmids: list[str]) -> dict[str, str]:
        pmid_to_pmc_uid: dict[str, str] = {}
        for batch in _batched(pmids, self.config.batch_size):
            params = {
                "ids": ",".join(batch),
                "format": "json",
                "tool": "dgilit",
            }
            if self.config.email:
                params["email"] = self.config.email

            try:
                payload = self._get_json(self.config.id_converter_url, params)
            except (json.JSONDecodeError, *BEST_EFFORT_PMC_ERRORS):
                continue

            for record in payload.get("records", []):
                pmid = _clean_text(str(record.get("pmid") or record.get("requested-id") or ""))
                pmcid = _clean_text(record.get("pmcid"))
                if pmid and pmcid:
                    pmid_to_pmc_uid[pmid] = pmcid.removeprefix("PMC")

            if self.config.request_interval_seconds:
                time.sleep(self.config.request_interval_seconds)

        return pmid_to_pmc_uid

    def _fetch_pmc_links_from_elink(self, pmids: list[str]) -> dict[str, str]:
        pmid_to_pmc_uid: dict[str, str] = {}
        for pmid in pmids:
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": pmid,
                "retmode": "xml",
            }
            if self.config.email:
                params["email"] = self.config.email

            try:
                root = self._get_xml(self.config.elink_url, params)
            except BEST_EFFORT_PMC_ERRORS:
                continue
            pmid_to_pmc_uid.update(_parse_pmc_links(root))

            if self.config.request_interval_seconds:
                time.sleep(self.config.request_interval_seconds)

        return pmid_to_pmc_uid

    def _fetch_pmc_batch(self, pmc_uids: list[str]) -> ET.Element:
        params = {
            "db": "pmc",
            "id": ",".join(pmc_uids),
            "retmode": "xml",
        }
        if self.config.email:
            params["email"] = self.config.email
        return self._get_xml(self.config.base_url, params)

    def _get_xml(self, base_url: str, params: dict[str, str]) -> ET.Element:
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "dgilit/0.1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return ET.fromstring(response.read())

    def _get_json(self, base_url: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "dgilit/0.1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return json.loads(response.read().decode())


def fetch_pubmed_abstracts(
    pmids: Iterable[Any],
    batch_size: int = 100,
    email: str | None = None,
) -> dict[str, str | None]:
    """Fetch complete PubMed abstract text for PMIDs using NCBI E-utilities."""
    client = PubMedClient(
        batch_size=batch_size,
        email=email,
    )
    return client.fetch_abstract_texts(pmids)


def _parse_pubmed_article(article: ET.Element) -> PubMedArticle | None:
    pmid = article.findtext(".//PMID")
    if not pmid:
        return None

    sections: list[PubMedArticleSection] = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        text = "".join(abstract_text.itertext()).strip()
        if text:
            sections.append(
                PubMedArticleSection(
                    label=abstract_text.attrib.get("Label"),
                    section_type="abstract",
                    source="pubmed",
                    text=text,
                )
            )

    return PubMedArticle(
        pmid=pmid,
        title=_clean_text(article.findtext(".//ArticleTitle")),
        journal=_clean_text(article.findtext(".//Journal/Title")),
        publication_year=_clean_text(article.findtext(".//PubDate/Year")),
        sections=sections,
    )


PubMedAbstractSection = PubMedArticleSection
PubMedAbstract = PubMedArticle


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _parse_pmc_links(root: ET.Element) -> dict[str, str]:
    links: dict[str, str] = {}
    for link_set in root.findall(".//LinkSet"):
        pmid = link_set.findtext("./IdList/Id")
        if not pmid:
            continue
        pmc_uid = None
        for link_set_db in link_set.findall("./LinkSetDb"):
            if link_set_db.findtext("./LinkName") != "pubmed_pmc":
                continue
            pmc_uid = link_set_db.findtext("./Link/Id")
            break
        if pmc_uid:
            links[pmid] = pmc_uid
    return links


def _parse_pmc_sections(article: ET.Element) -> list[PubMedArticleSection]:
    sections: list[PubMedArticleSection] = []

    for abstract in article.findall(".//front/article-meta/abstract"):
        label = _section_title(abstract)
        text = _section_body_text(abstract)
        if text:
            sections.append(
                PubMedArticleSection(
                    section_type="abstract",
                    label=_normalize_section_label(label),
                    section_id=abstract.attrib.get("id"),
                    source="pmc",
                    text=text,
                )
            )
        for sec in abstract.findall("./sec"):
            sections.extend(_parse_pmc_sec_tree(sec, "abstract", parent_label=label))

    for sec in article.findall(".//body/sec"):
        sections.extend(_parse_pmc_sec_tree(sec, "body"))

    return sections


def _parse_pmc_sec_tree(
    sec: ET.Element,
    section_type: str,
    parent_label: str | None = None,
) -> list[PubMedArticleSection]:
    sections: list[PubMedArticleSection] = []
    parsed = _parse_pmc_sec(sec, section_type, parent_label=parent_label)
    if parsed:
        sections.append(parsed)

    label = _section_title(sec)
    for child_sec in sec.findall("./sec"):
        sections.extend(
            _parse_pmc_sec_tree(
                child_sec,
                section_type,
                parent_label=label,
            )
        )
    return sections


def _parse_pmc_sec(
    sec: ET.Element,
    section_type: str,
    parent_label: str | None = None,
) -> PubMedArticleSection | None:
    label = _section_title(sec)
    text = _section_body_text(sec)
    if not text:
        return None
    return PubMedArticleSection(
        section_type=(
            "abstract"
            if section_type == "abstract"
            else _section_type_from_label(label, default=section_type)
        ),
        label=_normalize_section_label(label),
        section_id=sec.attrib.get("id"),
        parent_label=_normalize_section_label(parent_label),
        source="pmc",
        text=text,
    )


def _section_title(section: ET.Element) -> str | None:
    title = section.find("./title")
    if title is None:
        return None
    return _clean_text(" ".join(title.itertext()))


def _section_body_text(section: ET.Element) -> str | None:
    parts: list[str] = []
    for child in section:
        if _strip_namespace(child.tag) in {"title", "sec"}:
            continue
        text = _clean_text(" ".join(child.itertext()))
        if text:
            parts.append(text)
    return "\n".join(parts) or None


def _normalize_section_label(label: str | None) -> str | None:
    text = _clean_text(label)
    if not text:
        return None
    normalized = " ".join(text.upper().replace("-", " ").split())
    known = {
        "RESULT": "RESULTS",
        "RESULTS": "RESULTS",
        "DISCUSSION": "DISCUSSION",
        "CONCLUSION": "CONCLUSIONS",
        "CONCLUSIONS": "CONCLUSIONS",
    }
    return known.get(normalized, text)


def _section_type_from_label(label: str | None, default: str) -> str:
    normalized = _normalize_section_label(label)
    if normalized in {"RESULTS", "DISCUSSION", "CONCLUSIONS"}:
        return normalized.lower()
    return default


def _extract_pmc_uid(article: ET.Element) -> str | None:
    for article_id in article.findall(".//article-id"):
        if article_id.attrib.get("pub-id-type") in {"pmc", "pmcid", "pmcaid", "pmcaiid"}:
            pmcid = _clean_text(article_id.text)
            if pmcid:
                return pmcid.removeprefix("PMC")
    return None


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), batch_size):
        yield values[index:index + batch_size]
