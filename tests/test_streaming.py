"""Unit tests for streaming LLM output."""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aks.models.llm import ModelConfig, stream
from aks.agents.base import AgentMessage, BaseAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gemini_config() -> ModelConfig:
    return ModelConfig(model="gemini-test", max_tokens=256, temperature=0.0, provider="gemini")


def _anthropic_config() -> ModelConfig:
    return ModelConfig(model="claude-test", max_tokens=256, temperature=0.0, provider="anthropic")


def _make_gemini_chunks(texts: list[str]) -> list[MagicMock]:
    chunks = []
    for t in texts:
        c = MagicMock()
        c.text = t
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# llm.stream() — Gemini
# ---------------------------------------------------------------------------

class TestGeminiStream:
    def test_yields_text_chunks(self):
        client = MagicMock()
        client.models.generate_content_stream.return_value = _make_gemini_chunks(
            ["Hello", ", ", "world", "!"]
        )
        result = list(stream(client, _gemini_config(), "sys", [{"role": "user", "content": "hi"}]))
        assert result == ["Hello", ", ", "world", "!"]

    def test_skips_empty_chunks(self):
        client = MagicMock()
        chunks = _make_gemini_chunks(["a", "", "b"])
        chunks[1].text = ""  # empty string is falsy
        client.models.generate_content_stream.return_value = chunks
        result = list(stream(client, _gemini_config(), "sys", [{"role": "user", "content": "hi"}]))
        assert result == ["a", "b"]

    def test_skips_none_text_chunks(self):
        client = MagicMock()
        chunks = _make_gemini_chunks(["x"])
        none_chunk = MagicMock()
        none_chunk.text = None
        client.models.generate_content_stream.return_value = [chunks[0], none_chunk]
        result = list(stream(client, _gemini_config(), "sys", [{"role": "user", "content": "hi"}]))
        assert result == ["x"]

    def test_passes_system_and_messages_to_client(self):
        client = MagicMock()
        client.models.generate_content_stream.return_value = _make_gemini_chunks(["ok"])
        list(stream(client, _gemini_config(), "my system", [{"role": "user", "content": "query"}]))
        call_kwargs = client.models.generate_content_stream.call_args
        assert call_kwargs is not None

    def test_returns_iterator(self):
        client = MagicMock()
        client.models.generate_content_stream.return_value = _make_gemini_chunks(["hi"])
        result = stream(client, _gemini_config(), "sys", [{"role": "user", "content": "hi"}])
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_empty_response_yields_nothing(self):
        client = MagicMock()
        client.models.generate_content_stream.return_value = []
        result = list(stream(client, _gemini_config(), "sys", [{"role": "user", "content": "hi"}]))
        assert result == []


# ---------------------------------------------------------------------------
# llm.stream() — Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicStream:
    def _make_client(self, texts: list[str]) -> MagicMock:
        client = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.text_stream = iter(texts)
        client.messages.stream.return_value = ctx
        return client

    def test_yields_text_chunks(self):
        client = self._make_client(["Hi", " there", "!"])
        result = list(stream(client, _anthropic_config(), "sys", [{"role": "user", "content": "hey"}]))
        assert result == ["Hi", " there", "!"]

    def test_empty_stream_yields_nothing(self):
        client = self._make_client([])
        result = list(stream(client, _anthropic_config(), "sys", [{"role": "user", "content": "hey"}]))
        assert result == []

    def test_context_manager_is_used(self):
        client = self._make_client(["ok"])
        list(stream(client, _anthropic_config(), "sys", [{"role": "user", "content": "q"}]))
        client.messages.stream.return_value.__enter__.assert_called_once()
        client.messages.stream.return_value.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# BaseAgent.stream()
# ---------------------------------------------------------------------------

class TestBaseAgentStream:
    @pytest.fixture()
    def agent(self, monkeypatch):
        monkeypatch.setattr(
            "aks.agents.base.agent_config",
            lambda name: {"system_prompt": "You are helpful.", "description": "test"},
        )
        monkeypatch.setattr(
            "aks.agents.base.models_config",
            lambda: {"base": {"model": "gemini-test", "max_tokens": 256, "temperature": 0.0}},
        )
        monkeypatch.setattr("aks.agents.base.get_provider", lambda: "gemini")

        class ConcreteAgent(BaseAgent):
            name = "base"

        client = MagicMock()
        client.models.generate_content_stream.return_value = _make_gemini_chunks(["chunk1", "chunk2"])
        return ConcreteAgent(client)

    def test_returns_iterator_and_sources(self, agent):
        msg = AgentMessage(
            message_id="1", sender="orch", receiver="base",
            query="test query", context="", conversation_history=[],
        )
        chunks, sources = agent.stream(msg)
        assert hasattr(chunks, "__iter__")
        assert isinstance(sources, list)

    def test_chunks_match_client_output(self, agent):
        msg = AgentMessage(
            message_id="1", sender="orch", receiver="base",
            query="test query", context="", conversation_history=[],
        )
        chunks, _ = agent.stream(msg)
        assert list(chunks) == ["chunk1", "chunk2"]

    def test_sources_extracted_from_context(self, agent):
        context = "## Knowledge\n*Source: my-note.md | relevance: 0.9*\nsome content"
        msg = AgentMessage(
            message_id="1", sender="orch", receiver="base",
            query="q", context=context, conversation_history=[],
        )
        _, sources = agent.stream(msg)
        assert sources == ["my-note.md "]

    def test_conversation_history_prepended(self, agent):
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        msg = AgentMessage(
            message_id="1", sender="orch", receiver="base",
            query="follow up", context="", conversation_history=history,
        )
        chunks, _ = agent.stream(msg)
        list(chunks)  # consume to trigger the call
        call_args = agent.client.models.generate_content_stream.call_args
        # contents should include history + current query (3 messages)
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert len(contents) == 3
