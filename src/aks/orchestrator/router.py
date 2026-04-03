"""Intent classification and agent routing."""
from __future__ import annotations

import uuid
from typing import Any, Iterator

from aks.agents.base import AgentMessage, AgentResponse, BaseAgent
from aks.agents.code_agent import CodeAgent
from aks.models.llm import ModelConfig, complete
from aks.retrieval.search import retrieve_context
from aks.knowledge.store import KnowledgeStore
from aks.utils.config import models_config, get_provider

# Phase 1: only code agent is active
ACTIVE_AGENTS: dict[str, type[BaseAgent]] = {
    "code": CodeAgent,
}
DEFAULT_AGENT = "code"

ROUTING_SYSTEM = """You are an intent classifier for a personal AI assistant.
Classify the user's query into exactly one of these agents: {agents}.

Rules:
- Reply with ONLY the agent name (one word, lowercase).
- When uncertain, reply with the default: {default}.

Agent descriptions:
{descriptions}
"""


def _build_routing_system() -> str:
    from aks.utils.config import agent_config
    descs = "\n".join(
        f"- {name}: {agent_config(name)['description']}"
        for name in ACTIVE_AGENTS
    )
    return ROUTING_SYSTEM.format(
        agents=", ".join(ACTIVE_AGENTS),
        default=DEFAULT_AGENT,
        descriptions=descs,
    )


class Orchestrator:
    def __init__(self, client: Any, store: KnowledgeStore) -> None:
        self.client = client
        self.store = store
        cfg = models_config()
        m = cfg["orchestrator"]
        provider = get_provider()
        self._routing_config = ModelConfig(
            model=m["model"],
            max_tokens=m["max_tokens"],
            temperature=m["temperature"],
            provider=provider,
        )
        self._agents: dict[str, BaseAgent] = {
            name: cls(client) for name, cls in ACTIVE_AGENTS.items()
        }
        self._routing_system = _build_routing_system()

    def route(self, query: str, force_agent: str | None = None) -> str:
        """Return the agent name to handle this query."""
        if force_agent and force_agent in self._agents:
            return force_agent
        if len(self._agents) == 1:
            return DEFAULT_AGENT
        raw = complete(
            self.client,
            self._routing_config,
            self._routing_system,
            [{"role": "user", "content": query}],
        ).strip().lower()
        return raw if raw in self._agents else DEFAULT_AGENT

    def run(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        force_agent: str | None = None,
    ) -> AgentResponse:
        agent_name = self.route(query, force_agent)
        context = retrieve_context(query, self.store)
        msg = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender="orchestrator",
            receiver=agent_name,
            query=query,
            context=context,
            conversation_history=conversation_history or [],
        )
        return self._agents[agent_name].run(msg)

    def stream(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        force_agent: str | None = None,
    ) -> tuple[str, str, Iterator[str], list[str]]:
        """Route the query and stream the response.

        Returns (agent_name, model, chunk_iterator, sources).
        Routing is still a blocking call; only the agent response streams.
        """
        agent_name = self.route(query, force_agent)
        context = retrieve_context(query, self.store)
        msg = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender="orchestrator",
            receiver=agent_name,
            query=query,
            context=context,
            conversation_history=conversation_history or [],
        )
        chunks, sources = self._agents[agent_name].stream(msg)
        return agent_name, self._agents[agent_name].model_config.model, chunks, sources
