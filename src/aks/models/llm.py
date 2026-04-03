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


def _gemini_complete(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> tuple[str, int, int]:
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
    usage = response.usage_metadata
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    return response.text, in_tok, out_tok


def _gemini_stream(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> Iterator[tuple[str, int, int]]:
    """Yield (chunk, 0, 0) per chunk; final item is ("", in_tok, out_tok)."""
    from google.genai import types

    in_tok = out_tok = 0
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
            yield chunk.text, 0, 0
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
    yield "", in_tok, out_tok


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_client() -> Any:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _anthropic_complete(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> tuple[str, int, int]:
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=messages,
    )
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    return response.content[0].text, in_tok, out_tok


def _anthropic_stream(
    client: Any, config: ModelConfig, system: str, messages: list[dict]
) -> Iterator[tuple[str, int, int]]:
    """Yield (chunk, 0, 0) per chunk; final item is ("", in_tok, out_tok)."""
    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=messages,
    ) as s:
        for text in s.text_stream:
            yield text, 0, 0
        final = s.get_final_message()
    in_tok = final.usage.input_tokens
    out_tok = final.usage.output_tokens
    yield "", in_tok, out_tok


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding(text: str, provider: str = "gemini") -> list[float]:
    """Return a dense embedding vector for the given text."""
    if provider == "gemini":
        from google import genai
        from aks.utils.config import models_config
        from aks.utils.cost import CostLedger

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
        model = models_config()["embeddings"]["model"]
        result = client.models.embed_content(model=model, contents=text)

        usage = getattr(result, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        if in_tok:
            CostLedger().record("gemini", model, in_tok, 0)

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
    from aks.utils.cost import CostLedger

    if config.provider == "gemini":
        text, in_tok, out_tok = _gemini_complete(client, config, system, messages)
    else:
        text, in_tok, out_tok = _anthropic_complete(client, config, system, messages)
    if in_tok or out_tok:
        CostLedger().record(config.provider, config.model, in_tok, out_tok)
    return text


def stream(
    client: Any,
    config: ModelConfig,
    system: str,
    messages: list[dict],
) -> Iterator[str]:
    """Yield text chunks as they arrive; records cost after the last chunk."""
    from aks.utils.cost import CostLedger

    raw = _gemini_stream(client, config, system, messages) if config.provider == "gemini" \
        else _anthropic_stream(client, config, system, messages)

    in_tok = out_tok = 0
    try:
        for chunk, it, ot in raw:
            if chunk:
                yield chunk
            if it:
                in_tok = it
            if ot:
                out_tok = ot
    finally:
        if in_tok or out_tok:
            CostLedger().record(config.provider, config.model, in_tok, out_tok)
