"""Hybrid skill search engine (BM25 + embedding + lexical boost).

Implements the search pipeline:
  Phase 1: BM25 rough-rank over all candidates
  Phase 2: Vector scoring (embedding cosine similarity)
  Phase 3: Hybrid score = vector_score + lexical_boost
  Phase 4: Deduplication + limit

Used by MCP ``search_skills`` tool, ``retrieve_skill`` agent tool,
and potentially other search interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openspace.cloud")


def _check_safety(text: str) -> list[str]:
    """Lazy wrapper — avoids importing skill_engine at module load time."""
    from openspace.skill_engine.skill_utils import check_skill_safety
    return check_skill_safety(text)


def _is_safe(flags: list[str]) -> bool:
    from openspace.skill_engine.skill_utils import is_skill_safe
    return is_skill_safe(flags)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(value: str) -> list[str]:
    return _WORD_RE.findall(value.lower()) if value else []


def _lexical_boost(query_tokens: list[str], name: str, slug: str) -> float:
    """Compute lexical boost score based on exact/prefix token matching."""
    slug_tokens = _tokenize(slug)
    name_tokens = _tokenize(name)
    boost = 0.0

    # Slug exact / prefix
    if slug_tokens and all(
        any(ct == qt for ct in slug_tokens) for qt in query_tokens
    ):
        boost += 1.4
    elif slug_tokens and all(
        any(ct.startswith(qt) for ct in slug_tokens) for qt in query_tokens
    ):
        boost += 0.8

    # Name exact / prefix
    if name_tokens and all(
        any(ct == qt for ct in name_tokens) for qt in query_tokens
    ):
        boost += 1.1
    elif name_tokens and all(
        any(ct.startswith(qt) for ct in name_tokens) for qt in query_tokens
    ):
        boost += 0.6

    return boost


class SkillSearchEngine:
    """Hybrid BM25 + embedding search engine for skills.

    Usage::

        engine = SkillSearchEngine()
        results = engine.search(
            query="weather forecast",
            candidates=candidates,
            query_embedding=[...],  # optional
            limit=20,
        )
    """

    def search(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        query_embedding: Optional[List[float]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Run the full search pipeline on candidates.

        Each candidate dict should have at minimum:
          - ``skill_id``, ``name``, ``description``
          - ``_embedding`` (optional): pre-computed embedding vector
          - ``source``: "openspace-local" | "cloud"

        Args:
            query: Search query text.
            candidates: Candidate dicts to rank.
            query_embedding: Pre-computed query embedding (if available).
            limit: Max results to return.

        Returns:
            Sorted list of result dicts (highest score first).
        """
        q = query.strip()
        if not q or not candidates:
            return []

        query_tokens = _tokenize(q)
        if not query_tokens:
            return []

        # Phase 1: BM25 rough-rank
        filtered = self._bm25_phase(q, candidates, limit)

        # Phase 2+3: Vector + lexical scoring
        scored = self._score_phase(filtered, query_tokens, query_embedding)

        # Phase 4: Deduplicate and limit
        return self._dedup_and_limit(scored, limit)

    def _bm25_phase(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """BM25 rough-rank to keep top candidates for embedding stage."""
        from openspace.skill_engine.skill_ranker import SkillRanker, SkillCandidate

        ranker = SkillRanker(enable_cache=True)
        bm25_candidates = [
            SkillCandidate(
                skill_id=c.get("skill_id", ""),
                name=c.get("name", ""),
                description=c.get("description", ""),
                body="",
                metadata=c,
            )
            for c in candidates
        ]
        ranked = ranker.bm25_only(query, bm25_candidates, top_k=min(limit * 3, len(candidates)))

        ranked_ids = {sc.skill_id for sc in ranked}
        filtered = [c for c in candidates if c.get("skill_id") in ranked_ids]

        # If BM25 found nothing, fall back to all candidates
        return filtered if filtered else candidates

    def _score_phase(
        self,
        candidates: List[Dict[str, Any]],
        query_tokens: list[str],
        query_embedding: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        """Compute hybrid score = vector_score + lexical_boost."""
        from openspace.cloud.embedding import cosine_similarity

        scored = []
        for c in candidates:
            name = c.get("name", "")
            slug = c.get("skill_id", name).split("__")[0].replace(":", "-")

            # Vector score
            vector_score = 0.0
            if query_embedding:
                skill_emb = c.get("_embedding")
                if skill_emb and isinstance(skill_emb, list):
                    vector_score = cosine_similarity(query_embedding, skill_emb)

            # Lexical boost
            lexical = _lexical_boost(query_tokens, name, slug)

            final_score = vector_score + lexical

            entry: Dict[str, Any] = {
                "skill_id": c.get("skill_id", ""),
                "name": name,
                "description": c.get("description", ""),
                "source": c.get("source", ""),
                "score": round(final_score, 4),
            }
            if vector_score > 0:
                entry["vector_score"] = round(vector_score, 4)
            # Include optional fields
            for key in ("path", "visibility", "created_by", "origin", "tags", "quality", "safety_flags"):
                if c.get(key):
                    entry[key] = c[key]
            scored.append(entry)

        scored.sort(key=lambda x: -x["score"])
        return scored

    @staticmethod
    def _dedup_and_limit(
        scored: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Deduplicate by name and apply limit."""
        seen: set[str] = set()
        deduped = []
        for item in scored:
            name = item["name"]
            if name in seen:
                continue
            seen.add(name)
            deduped.append(item)
        return deduped[:limit]


def build_local_candidates(
    skills: list,
    store: Any = None,
) -> List[Dict[str, Any]]:
    """Build search candidate dicts from SkillRegistry skills.

    Args:
        skills: List of ``SkillMeta`` from ``registry.list_skills()``.
        store: Optional ``SkillStore`` instance for quality data enrichment.

    Returns:
        List of candidate dicts ready for ``SkillSearchEngine.search()``.
    """
    from openspace.cloud.embedding import build_skill_embedding_text

    candidates: List[Dict[str, Any]] = []
    for s in skills:
        # Read SKILL.md body
        readme_body = ""
        try:
            raw = s.path.read_text(encoding="utf-8")
            m = re.match(r"^---\n.*?\n---\n?", raw, re.DOTALL)
            readme_body = raw[m.end():].strip() if m else raw
        except Exception:
            pass

        embedding_text = build_skill_embedding_text(s.name, s.description, readme_body)

        # Safety check
        flags = _check_safety(embedding_text)
        if not _is_safe(flags):
            logger.info(f"BLOCKED local skill {s.skill_id} — {flags}")
            continue

        candidates.append({
            "skill_id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "source": "openspace-local",
            "path": str(s.path),
            "is_local": True,
            "safety_flags": flags if flags else None,
            "_embedding_text": embedding_text,
        })

    # Enrich with quality data
    if store and candidates:
        try:
            all_records = store.load_all(active_only=True)
            for c in candidates:
                rec = all_records.get(c["skill_id"])
                if rec:
                    c["quality"] = {
                        "total_selections": rec.total_selections,
                        "completion_rate": round(rec.completion_rate, 3),
                        "effective_rate": round(rec.effective_rate, 3),
                    }
                    c["tags"] = rec.tags
        except Exception as e:
            logger.warning(f"Quality lookup failed: {e}")

    return candidates


def build_cloud_candidates(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build search candidate dicts from cloud metadata items.

    Args:
        items: Items from ``OpenSpaceClient.fetch_metadata()``.

    Returns:
        List of candidate dicts (with safety filtering applied).
    """
    candidates: List[Dict[str, Any]] = []
    for item in items:
        name = item.get("name", "")
        desc = item.get("description", "")
        tags = item.get("tags", [])
        safety_text = f"{name}\n{desc}\n{' '.join(tags)}"
        flags = _check_safety(safety_text)
        if not _is_safe(flags):
            continue

        c_entry: Dict[str, Any] = {
            "skill_id": item.get("record_id", ""),
            "name": name,
            "description": desc,
            "source": "cloud",
            "visibility": item.get("visibility", "public"),
            "is_local": False,
            "created_by": item.get("created_by", ""),
            "origin": item.get("origin", ""),
            "tags": tags,
            "safety_flags": flags if flags else None,
        }
        # Carry pre-computed embedding
        platform_emb = item.get("embedding")
        if platform_emb and isinstance(platform_emb, list):
            c_entry["_embedding"] = platform_emb
        candidates.append(c_entry)

    return candidates


async def hybrid_search_skills(
    query: str,
    local_skills: list = None,
    store: Any = None,
    source: str = "all",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Shared cloud+local skill search with graceful fallback.

    Builds candidates, generates embeddings, runs ``SkillSearchEngine``.
    Cloud is attempted when *source* includes it; failures are silently
    skipped so the caller always gets local results at minimum.

    Args:
        query: Free-text search query.
        local_skills: ``SkillMeta`` list (from ``registry.list_skills()``).
        store: Optional ``SkillStore`` for quality enrichment.
        source: ``"all"`` | ``"local"`` | ``"cloud"``.
        limit: Maximum results.

    Returns:
        Ranked result dicts (same format as ``SkillSearchEngine.search()``).
    """
    from openspace.cloud.embedding import generate_embedding

    q = query.strip()
    if not q:
        return []

    candidates: List[Dict[str, Any]] = []

    if source in ("all", "local") and local_skills:
        candidates.extend(build_local_candidates(local_skills, store))

    if source in ("all", "cloud"):
        try:
            from openspace.cloud.auth import get_openspace_auth
            from openspace.cloud.client import OpenSpaceClient

            auth_headers, api_base = get_openspace_auth()
            if auth_headers:
                client = OpenSpaceClient(auth_headers, api_base)
                try:
                    from openspace.cloud.embedding import resolve_embedding_api
                    has_emb = bool(resolve_embedding_api()[0])
                except Exception:
                    has_emb = False
                items = await asyncio.to_thread(
                    client.fetch_metadata, include_embedding=has_emb, limit=200,
                )
                candidates.extend(build_cloud_candidates(items))
        except Exception as e:
            logger.warning(f"hybrid_search_skills: cloud unavailable: {e}")

    if not candidates:
        return []

    # query embedding (optional — key/URL resolved inside generate_embedding)
    query_embedding: Optional[List[float]] = None
    try:
        query_embedding = await asyncio.to_thread(generate_embedding, q)
        if query_embedding:
            for c in candidates:
                if not c.get("_embedding") and c.get("_embedding_text"):
                    emb = await asyncio.to_thread(
                        generate_embedding, c["_embedding_text"],
                    )
                    if emb:
                        c["_embedding"] = emb
    except Exception:
        pass

    engine = SkillSearchEngine()
    return engine.search(q, candidates, query_embedding=query_embedding, limit=limit)

