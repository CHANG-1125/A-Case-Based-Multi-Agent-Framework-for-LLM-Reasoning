"""Experiment configuration from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str | None
    model: str
    generator_temperature: float = 0.3
    critic_temperature: float = 0.0
    judge_temperature: float = 0.0


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3


def get_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k=int(os.environ.get("TOP_K", "3")),
    )


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


# OpenAI-compatible endpoint for UniAPI (see https://docs.uniapi.ai/docs/use/api/openai_compatible)
_DEFAULT_UNIAPI_BASE = "https://api.uniapi.io/v1"


def _resolve_llm_endpoint() -> tuple[str, str | None]:
    """
    Returns (api_key, base_url).
    base_url None => default OpenAI official endpoint in the SDK.
    """
    key_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    key_uni = (os.environ.get("UNIAPI_API_KEY") or "").strip()
    api_key = key_openai or key_uni
    explicit_base = (os.environ.get("OPENAI_BASE_URL") or "").strip()

    if explicit_base:
        return api_key, explicit_base or None

    use_uniapi = _env_truthy("USE_UNIAPI") or (bool(key_uni) and not key_openai)
    if use_uniapi:
        uni_base = (os.environ.get("UNIAPI_BASE_URL") or "").strip() or _DEFAULT_UNIAPI_BASE
        return api_key, uni_base

    return api_key, None


def get_llm_config() -> LLMConfig:
    api_key, base_url = _resolve_llm_endpoint()
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY, or for UniAPI (https://uniapi.ai/dashboard/key) set "
            "USE_UNIAPI=1 with OPENAI_API_KEY / UNIAPI_API_KEY, and optionally "
            "UNIAPI_BASE_URL (default https://api.uniapi.io/v1; mainland: https://hk.uniapi.io/v1)."
        )
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        generator_temperature=float(os.environ.get("GENERATOR_TEMPERATURE", "0.3")),
        critic_temperature=float(os.environ.get("CRITIC_TEMPERATURE", "0")),
        judge_temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0")),
    )
