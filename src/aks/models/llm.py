"""LLM client — supports Gemini (default) and Anthropic."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class ModelConfig:
    model: str
    max_tokens: int
    temperature: float
    provider: str = "gemini"  # "gemini" | "anthropic"


# ---------------------------------------------------------------------------
# Gemini (google-genai SDK)
# ---------------------------------------------------------------------------

def _gemini_client() -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set")
    from google import genai
    return genai.Client(api_key=api_key)


def _gemini_contents(messages: list[dict]) -> list:
    from google.genai import types
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else m["role"],
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]


def _gemini_complete(client: Any, config: ModelConfig, system: str, messages: list[dict]) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=config.model,
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        ),
    )
    return response.text


def _gemini_stream(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> Iterator[str]:
    from google.genai import types

    for chunk in client.models.generate_content_stream(
        model=config.model,
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        ),
    ):
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_client() -> Any:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _anthropic_complete(client: Any, config: ModelConfig, system: str, messages: list[dict]) -> str:
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def _anthropic_stream(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> Iterator[str]:
    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=messages,
    ) as s:
        yield from s.text_stream


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding(text: str, provider: str = "gemini") -> list[float]:
    """Return a dense embedding vector for the given text."""
    if provider == "gemini":
        from google import genai
        from aks.utils.config import models_config
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
        model = models_config()["embeddings"]["model"]
        result = client.models.embed_content(model=model, contents=text)
        return list(result.embeddings[0].values)
    raise ValueError(f"Embeddings not supported for provider: {provider!r}")


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_clients: dict[str, Any] = {}


def get_client(provider: str = "gemini") -> Any:
    if provider not in _clients:
        if provider == "gemini":
            _clients[provider] = _gemini_client()
        elif provider == "anthropic":
            _clients[provider] = _anthropic_client()
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    return _clients[provider]


def complete(
    client: Any,
    config: ModelConfig,
    system: str,
    messages: list[dict],
) -> str:
    if config.provider == "gemini":
        return _gemini_complete(client, config, system, messages)
    return _anthropic_complete(client, config, system, messages)


def stream(
    client: Any,
    config: ModelConfig,
    system: str,
    messages: list[dict],
) -> Iterator[str]:
    """Yield text chunks as they arrive from the provider."""
    if config.provider == "gemini":
        yield from _gemini_stream(client, config, system, messages)
    else:
        yield from _anthropic_stream(client, config, system, messages)
