"""Cloud platform integration.

Provides:
  - ``ScionClient`` — HTTP client for the cloud API
  - ``get_scion_auth`` — credential resolution
  - ``SkillSearchEngine`` — hybrid BM25 + embedding search
  - ``generate_embedding`` — OpenAI embedding generation
"""

from scion.cloud.auth import get_scion_auth


def __getattr__(name: str):
    if name == "ScionClient":
        from scion.cloud.client import ScionClient

        return ScionClient
    if name == "SkillSearchEngine":
        from scion.cloud.search import SkillSearchEngine

        return SkillSearchEngine
    if name == "generate_embedding":
        from scion.cloud.embedding import generate_embedding

        return generate_embedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScionClient",
    "get_scion_auth",
    "SkillSearchEngine",
    "generate_embedding",
]
