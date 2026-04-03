"""Unit tests for multi-agent chaining (Phase 3)."""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aks.orchestrator.router import (
    VALID_CHAINS,
    Orchestrator,
    _parse_chain,
)


# ---------------------------------------------------------------------------
# _parse_chain
# ---------------------------------------------------------------------------

class TestParseChain:
    def test_single_agent(self):
        assert _parse_chain("code") == ["code"]

    def test_valid_chain(self):
        assert _parse_chain("code->writing") == ["code", "writing"]

    def test_valid_chain_pkm_planning(self):
        assert _parse_chain("pkm->planning") == ["pkm", "planning"]

    def test_unknown_agent_returns_empty(self):
        assert _parse_chain("unknown") == []

    def test_unsupported_chain_falls_back_to_first(self):
        # writing->code is not a VALID_CHAIN
        assert _parse_chain("writing->code") == ["writing"]

    def test_all_valid_chains_parse(self):
        for chain_str in VALID_CHAINS:
            parts = chain_str.split("->")
            assert _parse_chain(chain_str) == parts

    def test_garbage_input_returns_empty(self):
        assert _parse_chain("definitely not an agent") == []

    def test_partial_valid_chain_with_bad_second(self):
        assert _parse_chain("code->garbage") == []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_agent_config(name: str) -> dict:
    return {
        "system_prompt": f"You are the {name} agent.",
        "description": f"{name} description",
        "keywords": [],
    }


def _make_models_config() -> dict:
    base = {"model": "gemini-test", "max_tokens": 256, "temperature": 0.0}
    return {
        "orchestrator": base,
        "code": base,
        "pkm": base,
        "writing": base,
        "planning": base,
    }


@pytest.fixture()
def orchestrator(monkeypatch):
    monkeypatch.setattr("aks.agents.base.agent_config", _make_agent_config)
    monkeypatch.setattr("aks.agents.base.models_config", _make_models_config)
    monkeypatch.setattr("aks.agents.base.get_provider", lambda: "gemini")
    monkeypatch.setattr("aks.orchestrator.router.agent_config", _make_agent_config)
    monkeypatch.setattr("aks.orchestrator.router.models_config", _make_models_config)
    monkeypatch.setattr("aks.orchestrator.router.get_provider", lambda: "gemini")
    monkeypatch.setattr("aks.orchestrator.router.retrieve_context", lambda q, s: "")

    client = MagicMock()
    store = MagicMock()
    return Orchestrator(client=client, store=store)


# ---------------------------------------------------------------------------
# route_chain
# ---------------------------------------------------------------------------

class TestRouteChain:
    def test_force_agent_returns_single(self, orchestrator):
        chain = orchestrator.route_chain("anything", force_agent="pkm")
        assert chain == ["pkm"]

    def test_keyword_routing_single(self, orchestrator, monkeypatch):
        monkeypatch.setattr(
            "aks.orchestrator.router.agent_config",
            lambda name: {
                **_make_agent_config(name),
                "keywords": ["debug", "error"] if name == "code" else [],
            },
        )
        # Rebuild routing system and keyword config
        orchestrator._routing_system = "irrelevant"
        chain = orchestrator.route_chain("debug this error")
        assert chain == ["code"]

    def test_llm_routing_single_agent(self, orchestrator, monkeypatch):
        monkeypatch.setattr(
            "aks.orchestrator.router.complete",
            lambda *a, **kw: "writing",
        )
        chain = orchestrator.route_chain("write me an email")
        assert chain == ["writing"]

    def test_llm_routing_chain(self, orchestrator, monkeypatch):
        monkeypatch.setattr(
            "aks.orchestrator.router.complete",
            lambda *a, **kw: "code->writing",
        )
        chain = orchestrator.route_chain("document this function")
        assert chain == ["code", "writing"]

    def test_llm_routing_invalid_falls_back_to_default(self, orchestrator, monkeypatch):
        monkeypatch.setattr(
            "aks.orchestrator.router.complete",
            lambda *a, **kw: "garbage",
        )
        chain = orchestrator.route_chain("something")
        assert chain == ["code"]  # DEFAULT_AGENT


# ---------------------------------------------------------------------------
# stream_chain — single agent
# ---------------------------------------------------------------------------

class TestStreamChainSingleAgent:
    def test_returns_single_element_chain(self, orchestrator, monkeypatch):
        monkeypatch.setattr("aks.orchestrator.router.complete", lambda *a, **kw: "pkm")
        monkeypatch.setattr(
            "aks.agents.base.stream",
            lambda *a, **kw: iter(["chunk1", "chunk2"]),
        )
        chain, model, chunks, sources = orchestrator.stream_chain("find my notes")
        assert chain == ["pkm"]
        assert list(chunks) == ["chunk1", "chunk2"]

    def test_sources_returned(self, orchestrator, monkeypatch):
        monkeypatch.setattr("aks.orchestrator.router.complete", lambda *a, **kw: "code")
        monkeypatch.setattr(
            "aks.agents.base.stream",
            lambda *a, **kw: iter([]),
        )
        monkeypatch.setattr(
            "aks.orchestrator.router.retrieve_context",
            lambda q, s: "*Source: my-note.md | relevance: 0.9*\nsome content",
        )
        chain, _, chunks, sources = orchestrator.stream_chain("debug this")
        list(chunks)
        assert "my-note.md " in sources


# ---------------------------------------------------------------------------
# stream_chain — multi-agent
# ---------------------------------------------------------------------------

class TestStreamChainMultiAgent:
    def _patch_complete_for_chain(self, monkeypatch, chain_str: str, agent_response: str = "agent output"):
        """Patch complete() to return chain_str for routing, agent_response for agent runs."""
        call_count = {"n": 0}

        def fake_complete(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return chain_str  # routing call
            return agent_response  # agent run() call

        monkeypatch.setattr("aks.orchestrator.router.complete", fake_complete)
        monkeypatch.setattr("aks.agents.base.complete", lambda *a, **kw: agent_response)

    def test_chain_executes_two_agents(self, orchestrator, monkeypatch):
        self._patch_complete_for_chain(monkeypatch, "code->writing")
        streamed = ["final", " output"]
        monkeypatch.setattr("aks.agents.base.stream", lambda *a, **kw: iter(streamed))

        chain, model, chunks, sources = orchestrator.stream_chain("document this function")
        assert chain == ["code", "writing"]
        assert list(chunks) == streamed

    def test_handoff_query_contains_original_and_analysis(self, orchestrator, monkeypatch):
        """The final agent receives both the original query and the intermediate output."""
        captured_msgs = []

        def fake_agent_stream(agent_self, msg):
            captured_msgs.append(msg)
            return iter(["done"]), []

        self._patch_complete_for_chain(monkeypatch, "pkm->writing", "retrieved notes content")
        monkeypatch.setattr("aks.agents.base.BaseAgent.stream", fake_agent_stream)

        orchestrator.stream_chain("turn my notes into a blog post")
        assert len(captured_msgs) == 1  # only writing agent streams
        handoff = captured_msgs[0].query
        assert "Original request:" in handoff
        assert "pkm analysis" in handoff
        assert "retrieved notes content" in handoff

    def test_sources_aggregated_from_all_agents(self, orchestrator, monkeypatch):
        self._patch_complete_for_chain(monkeypatch, "code->writing")

        monkeypatch.setattr(
            "aks.orchestrator.router.retrieve_context",
            lambda q, s: "*Source: note-a.md | relevance: 0.9*\ncontent",
        )
        monkeypatch.setattr("aks.agents.base.stream", lambda *a, **kw: iter([]))

        chain, _, chunks, sources = orchestrator.stream_chain("doc this code")
        list(chunks)
        # Sources extracted from context appear for both agents
        assert any("note-a.md" in s for s in sources)

    def test_pkm_planning_chain(self, orchestrator, monkeypatch):
        self._patch_complete_for_chain(monkeypatch, "pkm->planning")
        monkeypatch.setattr("aks.agents.base.stream", lambda *a, **kw: iter(["plan output"]))

        chain, _, chunks, _ = orchestrator.stream_chain("create a plan from my notes")
        assert chain == ["pkm", "planning"]
        assert list(chunks) == ["plan output"]
