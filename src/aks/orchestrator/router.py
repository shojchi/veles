"""Intent classification and agent routing."""
from __future__ import annotations

import uuid

from aks.agents.base import AgentMessage, AgentResponse, BaseAgent
from aks.agents.code_agent import CodeAgent
from aks.agents.pkm_agent import PKMAgent
from aks.agents.writing_agent import WritingAgent
from aks.agents.planning_agent import PlanningAgent
from aks.knowledge.store import KnowledgeStore
from aks.retrieval.search import retrieve_context
from aks.utils.config import models_config, agent_config

ACTIVE_AGENTS: dict[str, type[BaseAgent]] = {
    "code": CodeAgent,
    "pkm": PKMAgent,
    "writing": WritingAgent,
    "planning": PlanningAgent,
}
DEFAULT_AGENT = "code"

ROUTING_SYSTEM = """You are an intent classifier for a personal AI assistant.
Classify the user query into exactly one of these agents: {agents}.
Reply with ONLY the agent name — one word, lowercase. No explanation.

Agent roles:
{descriptions}

Routing rules:
- code     → writing/debugging/reviewing actual code, CLI commands, software architecture, DevOps, specific languages (Python, JS, SQL…)
- planning → learning strategies, roadmaps, "best way to…", how-to advice, step-by-step guides, project breakdowns, schedules, priorities
- pkm      → recalling or searching the user's own notes, summarizing research, finding connections across knowledge base
- writing  → drafting emails, documents, reports, translations, editing or proofreading existing text

Key distinction: "how to learn X" or "best way to do Y" → planning, NOT code (even if X is a programming topic).

Examples:
query: "how do i fix this python KeyError?" → code
query: "what is the best way to learn programming?" → planning
query: "review my docker-compose file" → code
query: "write an email to my manager about the delay" → writing
query: "what did i write about machine learning?" → pkm
query: "plan my week" → planning
query: "summarize my notes on kubernetes" → pkm
query: "translate this paragraph to ukrainian" → writing
query: "debug this bash script" → code
query: "how should i approach learning system design?" → planning

When uncertain, reply with: {default}
"""


def _build_routing_system() -> str:
    descs = "\n".join(
        f"- {name}: {agent_config(name)['description']}"
        for name in ACTIVE_AGENTS
    )
    return ROUTING_SYSTEM.format(
        agents=", ".join(ACTIVE_AGENTS),
        default=DEFAULT_AGENT,
        descriptions=descs,
    )


def _keyword_route(query: str) -> str | None:
    """Fast keyword pre-filter. Returns agent name if confident, else None."""
    q = query.lower()
    scores: dict[str, int] = {name: 0 for name in ACTIVE_AGENTS}
    for name in ACTIVE_AGENTS:
        cfg = agent_config(name)
        for kw in cfg.get("keywords", []):
            if kw.lower() in q:
                scores[name] += 1
    best_name = max(scores, key=lambda n: scores[n])
    best_score = scores[best_name]
    # Only trust keyword routing when there's a clear winner
    if best_score >= 2:
        return best_name
    rivals = [n for n, s in scores.items() if s == best_score and n != best_name]
    if best_score == 1 and not rivals:
        return best_name
    return None


class Orchestrator:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        m = models_config()["orchestrator"]
        self._routing_max_tokens: int = m["max_tokens"]
        self._routing_temperature: float = m["temperature"]
        self._agents: dict[str, BaseAgent] = {
            name: cls() for name, cls in ACTIVE_AGENTS.items()
        }
        self._routing_system = _build_routing_system()

    def route(self, query: str, force_agent: str | None = None) -> str:
        """Return the agent name to handle this query."""
        if force_agent and force_agent in self._agents:
            return force_agent
        if len(self._agents) == 1:
            return DEFAULT_AGENT
        # Fast path: keyword match before spending an LLM call
        keyword_pick = _keyword_route(query)
        if keyword_pick:
            return keyword_pick
        from aks.models.llm import complete_with_fallback
        raw, _ = complete_with_fallback(
            self._routing_system,
            [{"role": "user", "content": query}],
            self._routing_max_tokens,
            self._routing_temperature,
        )
        raw = raw.strip().lower()
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
