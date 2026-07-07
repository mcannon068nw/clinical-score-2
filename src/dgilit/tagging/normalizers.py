"""Cached VICC concept normalization utilities for dgiLIT pre-tagging."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode
from tqdm.auto import tqdm

import inflect
import requests

from .models import EntityType, NormalizedConcept

NormalizerKind = Literal["gene", "therapy", "disease"]


class SQLiteNormalizerCache:
    """Tiny persistent cache for normalizer responses.

    This avoids the largest avoidable runtime cost: repeatedly hitting VICC
    normalizers for the same strings across notebook runs.
    """

    def __init__(self, path: str | Path = ".dgilit_normalizer_cache.sqlite") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS normalizer_cache (
                normalizer_kind TEXT NOT NULL,
                query TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (normalizer_kind, query)
            )
            """
        )
        self.conn.commit()

    def get(self, normalizer_kind: str, query: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT response_json FROM normalizer_cache WHERE normalizer_kind = ? AND query = ?",
            (normalizer_kind, query),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, normalizer_kind: str, query: str, response: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO normalizer_cache
            (normalizer_kind, query, response_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (normalizer_kind, query, json.dumps(response), time.time()),
        )
        self.conn.commit()


class ViccNormalizer:
    """Client for VICC gene, therapy, and disease normalization endpoints."""

    BASE_URL = "https://normalize.cancervariants.org"

    def __init__(
        self,
        cache: SQLiteNormalizerCache | None = None,
        timeout_seconds: int = 20,
        singularize: bool = True,
    ) -> None:
        self.cache = cache or SQLiteNormalizerCache()
        self.timeout_seconds = timeout_seconds
        self.singularize = singularize
        self._inflector = inflect.engine()

    def normalize_entity_text(self, entity_type: EntityType, text: str) -> NormalizedConcept:
        kind = self._normalizer_kind(entity_type)
        query = self._clean_query(text)
        return self.normalize(kind, query)

    def normalize(self, kind: NormalizerKind, query: str) -> NormalizedConcept:
        cached = self.cache.get(kind, query)
        if cached is not None:
            return self._parse_response(kind, cached)

        url = self._build_url(kind, query)
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            payload = {"error_message": str(exc), "match_type": "Failure to Normalize"}

        self.cache.set(kind, query, payload)
        return self._parse_response(kind, payload)

    def normalize_many_unique(
        self, entity_texts: set[tuple[EntityType, str]]
    ) -> dict[tuple[EntityType, str], NormalizedConcept]:
        """Normalize unique entity strings once, then map back to mentions."""
        results: dict[tuple[EntityType, str], NormalizedConcept] = {}

        sorted_entities = sorted(entity_texts, key=lambda x: (x[0], x[1].lower()))

        for entity_type, text in tqdm(
            sorted_entities,
            desc="Normalizing unique entities",
            unit="entity",
        ):
            cleaned = self._clean_query(text)
            results[(entity_type, text)] = self.normalize_entity_text(entity_type, cleaned)

        return results

    def _clean_query(self, text: str) -> str:
        text = " ".join(text.strip().split())
        if self.singularize:
            return self._inflector.singular_noun(text) or text
        return text

    @staticmethod
    def _normalizer_kind(entity_type: EntityType) -> NormalizerKind:
        if entity_type == "gene":
            return "gene"
        if entity_type == "drug":
            return "therapy"
        return "disease"

    def _build_url(self, kind: NormalizerKind, query: str) -> str:
        if kind == "therapy":
            qs = urlencode({"q": query, "infer_namespace": "true"})
        else:
            qs = urlencode({"q": query})
        return f"{self.BASE_URL}/{kind}/normalize?{qs}"

    @staticmethod
    def _parse_response(kind: NormalizerKind, payload: dict[str, Any]) -> NormalizedConcept:
        if payload.get("error_message"):
            return NormalizedConcept(
                match_type=payload.get("match_type", "Failure to Normalize"),
                normalizer=f"vicc:{kind}",
                raw_response=payload,
                error_message=payload.get("error_message"),
            )

        match_type = payload.get("match_type")
        if not match_type:
            return NormalizedConcept(match_type=match_type, normalizer=f"vicc:{kind}", raw_response=payload)

        concept_key = {"gene": "gene", "therapy": "therapy", "disease": "disease"}[kind]
        concept = payload.get(concept_key) or {}
        return NormalizedConcept(
            concept_id=concept.get("id"),
            concept_label=concept.get("name") or concept.get("label"),
            match_type=match_type,
            normalizer=f"vicc:{kind}",
            raw_response=payload,
        )
