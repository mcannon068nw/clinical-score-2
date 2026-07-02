"""Utilities for fetching PubMed abstracts by PMID."""

from __future__ import annotations

import math
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedFetchConfig(BaseModel):
    """Configuration for PubMed E-utilities abstract fetches."""

    model_config = ConfigDict(extra="forbid")

    email: str | None = None
    batch_size: int = Field(default=100, ge=1)
    request_interval_seconds: float = Field(default=0.34, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    base_url: str = PUBMED_EFETCH_URL


class PubMedAbstractSection(BaseModel):
    """One labeled or unlabeled section of a PubMed abstract."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    label: str | None = None

    def render(self) -> str:
        if self.label:
            return f"{self.label}: {self.text}"
        return self.text


class PubMedAbstract(BaseModel):
    """Structured abstract text returned from PubMed for one PMID."""

    model_config = ConfigDict(extra="forbid")

    pmid: str = Field(..., min_length=1)
    sections: list[PubMedAbstractSection] = Field(default_factory=list)

    @field_validator("pmid", mode="before")
    @classmethod
    def normalize_pmid(cls, value: Any) -> str:
        return str(value).strip()

    @property
    def text(self) -> str | None:
        if not self.sections:
            return None
        return "\n".join(section.render() for section in self.sections)


class PubMedClient:
    """Small PubMed EFetch client for retrieving abstract text."""

    def __init__(self, config: PubMedFetchConfig | None = None, **kwargs: Any) -> None:
        if config and kwargs:
            raise ValueError("Pass either config or keyword options, not both")
        self.config = config or PubMedFetchConfig(**kwargs)

    def fetch_abstracts(self, pmids: Iterable[Any]) -> dict[str, PubMedAbstract]:
        """Fetch structured PubMed abstracts keyed by PMID."""
        normalized_pmids = [_normalize_pmid(pmid) for pmid in pmids]
        normalized_pmids = [pmid for pmid in normalized_pmids if pmid is not None]

        abstracts: dict[str, PubMedAbstract] = {}
        for batch in _batched(normalized_pmids, self.config.batch_size):
            root = self._fetch_batch(batch)
            for article in root.findall(".//PubmedArticle"):
                abstract = _parse_pubmed_article(article)
                if abstract:
                    abstracts[abstract.pmid] = abstract

            if self.config.request_interval_seconds:
                time.sleep(self.config.request_interval_seconds)

        return abstracts

    def fetch_abstract_texts(self, pmids: Iterable[Any]) -> dict[str, str | None]:
        """Fetch rendered abstract text keyed by PMID."""
        return {
            pmid: abstract.text
            for pmid, abstract in self.fetch_abstracts(pmids).items()
        }

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


def _parse_pubmed_article(article: ET.Element) -> PubMedAbstract | None:
    pmid = article.findtext(".//PMID")
    if not pmid:
        return None

    sections: list[PubMedAbstractSection] = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        text = "".join(abstract_text.itertext()).strip()
        if text:
            sections.append(
                PubMedAbstractSection(
                    label=abstract_text.attrib.get("Label"),
                    text=text,
                )
            )

    return PubMedAbstract(pmid=pmid, sections=sections)


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
