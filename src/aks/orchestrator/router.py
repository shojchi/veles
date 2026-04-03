"""Intent classification and agent routing."""
from __future__ import annotations

import uuid
from typing import Any, Iterator

from aks.agents.base import AgentMessage, AgentResponse, BaseAgent
from aks.agents.code_agent import CodeAgent
from aks.agents.pkm_agent import PKMAgent
from aks.agents.writing_agent import WritingAgent
from aks.agents.planning_agent import PlanningAgent
from aks.knowledge.store import KnowledgeStore
from aks.models.llm import ModelConfig, complete
from aks.retrieval.search import retrieve_context
from aks.utils.config import models_config, agent_config, get_provider

ACTIVE_AGENTS: dict[str, type[BaseAgent]] = {
    "code": CodeAgent,
    "pkm": PKMAgent,
    "writing": WritingAgent,
    "planning": PlanningAgent,
}
DEFAULT_AGENT = "code"

# Two-agent chains supported by the LLM router.
VALID_CHAINS: frozenset[str] = frozenset({
    "code->writing",    # technical analysis → document/explain in writing
    "pkm->writing",     # retrieve notes → compose email/doc
    "pkm->planning",    # retrieve notes → create plan
    "code->planning",   # technical breakdown → structured plan
})

ROUTING_SYSTEM = """You are an intent classifier for a personal AI assistant.
Classify the user query into a single agent or a two-agent chain.

Single agents: {agents}
Supported chains: {chains}

Reply with ONLY the agent name or chain — no explanation, no punctuation.

Agent roles:
{descriptions}

Routing rules:
- code     → writing/debugging/reviewing actual code, CLI commands, software architecture, DevOps, specific languages (Python, JS, SQL…)
- planning → learning strategies, roadmaps, "best way to…", how-to advice, step-by-step guides, project breakdowns, schedules, priorities
- pkm      → recalling or searching the user's own notes, summarizing research, finding connections across knowledge base
- writing  → drafting emails, documents, reports, translations, editing or proofreading existing text

Chain routing rules:
- code->writing   → analyse/debug code AND produce documentation, explanations, or written output
- pkm->writing    → retrieve from notes AND compose an email, document, or report from them
- pkm->planning   → retrieve from notes AND create a structured plan or roadmap
- code->planning  → analyse technical work AND produce a structured plan or breakdown

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
query: "document this function for me" → code->writing
query: "write docs for this code" → code->writing
query: "turn my notes on redis into a blog post" → pkm->writing
query: "email my team a summary of my kubernetes notes" → pkm->writing
query: "create a learning plan based on my notes" → pkm->planning
query: "break down this codebase into a sprint plan" → code->planning

When uncertain, reply with: {default}
"""


def _build_routing_system() -> str:
    descs = "\n".join(
        f"- {name}: {agent_config(name)['description']}"
        for name in ACTIVE_AGENTS
    )
    return ROUTING_SYSTEM.format(
        agents=", ".join(ACTIVE_AGENTS),
        chains=", ".join(sorted(VALID_CHAINS)),
        default=DEFAULT_AGENT,
        descriptions=descs,
    )


def _parse_chain(raw: str) -> list[str]:
    """Parse LLM routing output into a chain of agent names.

    "code"          → ["code"]
    "code->writing" → ["code", "writing"]
    "garbage"       → []
    """
    parts = [p.strip() for p in raw.split("->")]
    if not all(p in ACTIVE_AGENTS for p in parts):
        return []
    if len(parts) == 2 and "->".join(parts) not in VALID_CHAINS:
        # Unsupported chain — fall back to just the first agent
        return [parts[0]]
    return parts


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
    if best_score >= 2:
        return best_name
    rivals = [n for n, s in scores.items() if s == best_score and n != best_name]
    if best_score == 1 and not rivals:
        return best_name
    return None


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
        """Return the agent name to handle this query (single agent)."""
        return self.route_chain(query, force_agent)[0]

    def route_chain(self, query: str, force_agent: str | None = None) -> list[str]:
        """Return the ordered list of agent names to run for this query."""
        if force_agent and force_agent in self._agents:
            return [force_agent]
        if len(self._agents) == 1:
            return [DEFAULT_AGENT]
        keyword_pick = _keyword_route(query)
        if keyword_pick:
            return [keyword_pick]
        raw = complete(
            self.client,
            self._routing_config,
            self._routing_system,
            [{"role": "user", "content": query}],
        ).strip().lower()
        chain = _parse_chain(raw)
        return chain if chain else [DEFAULT_AGENT]

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
        """Route and stream. Returns (agent_name, model, chunks, sources)."""
        chain, model, chunks, sources = self.stream_chain(query, conversation_history, force_agent)
        return chain[0] if len(chain) == 1 else " -> ".join(chain), model, chunks, sources

    def stream_chain(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        force_agent: str | None = None,
    ) -> tuple[list[str], str, Iterator[str], list[str]]:
        """Route and stream, supporting multi-agent chains.

        Returns (chain, final_model, chunk_iterator, all_sources).
        For single-agent queries the chain has one element.
        For multi-agent chains every agent except the last runs to completion
        (blocking), then the final agent streams its output.
        """
        chain = self.route_chain(query, force_agent)
        context = retrieve_context(query, self.store)
        history = conversation_history or []
        all_sources: list[str] = []

        if len(chain) == 1:
            agent_name = chain[0]
            msg = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender="orchestrator",
                receiver=agent_name,
                query=query,
                context=context,
                conversation_history=history,
            )
            chunks, sources = self._agents[agent_name].stream(msg)
            all_sources.extend(sources)
            return chain, self._agents[agent_name].model_config.model, chunks, all_sources

        # Multi-agent chain: run all but the last agent to completion.
        handoff_query = query
        for agent_name in chain[:-1]:
            msg = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender="orchestrator",
                receiver=agent_name,
                query=handoff_query,
                context=context,
                conversation_history=history,
            )
            resp = self._agents[agent_name].run(msg)
            all_sources.extend(resp.sources_used)
            handoff_query = (
                f"Original request: {query}\n\n"
                f"[{agent_name} analysis]:\n{resp.content}"
            )

        # Stream the final agent.
        final_agent = chain[-1]
        final_msg = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender="orchestrator",
            receiver=final_agent,
            query=handoff_query,
            context=context,
            conversation_history=history,
        )
        chunks, final_sources = self._agents[final_agent].stream(final_msg)
        all_sources.extend(final_sources)
        return chain, self._agents[final_agent].model_config.model, chunks, all_sources
