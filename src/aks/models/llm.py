"""LLM client — provider fallback chain + embeddings."""
from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Fallback-chain completion
# ---------------------------------------------------------------------------

def complete_with_fallback(
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Try providers in order. Returns (content, 'provider/model').
    Skips providers whose API key env var is not set.
    Checks and records cost via CostLedger."""
    from aks.utils.config import get_fallback_chain
    from aks.utils.cost import CostLedger

    ledger = CostLedger()
    ledger.check_cap()

    chain = get_fallback_chain()
    errors: list[str] = []

    for cfg in chain:
        api_key = os.getenv(cfg["api_key_env"], "")
        if not api_key:
            continue
        try:
            if cfg["type"] == "gemini":
                content, in_tok, out_tok = _gemini_complete(
                    api_key, cfg["model"], system, messages, max_tokens, temperature
                )
            else:
                content, in_tok, out_tok = _openai_compat_complete(
                    cfg["base_url"], api_key, cfg["model"],
                    system, messages, max_tokens, temperature,
                )
            ledger.record(cfg["name"], cfg["model"], in_tok, out_tok)
            return content, f"{cfg['name']}/{cfg['model']}"
        except Exception as e:
            errors.append(f"{cfg['name']}: {e}")

    raise RuntimeError("All providers exhausted:\n" + "\n".join(errors))


def _gemini_complete(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else m["role"],
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    usage = response.usage_metadata
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    return response.text, in_tok, out_tok


def _openai_compat_complete(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    import openai

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = response.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return response.choices[0].message.content, in_tok, out_tok


# ---------------------------------------------------------------------------
# Embeddings (always Gemini)
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
        embed_cfg = models_config()["embeddings"]
        model = embed_cfg["model"]
        result = client.models.embed_content(model=model, contents=text)

        # Best-effort token count from usage metadata (may not always be present)
        usage = getattr(result, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        if in_tok:
            CostLedger().record("gemini-embedding", model, in_tok, 0)

        return list(result.embeddings[0].values)
    raise ValueError(f"Embeddings not supported for provider: {provider!r}")
