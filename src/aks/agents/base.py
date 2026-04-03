"""Base class for all specialist agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from aks.models.llm import ModelConfig, complete, stream
from aks.utils.config import agent_config, models_config, get_provider


@dataclass
class AgentMessage:
    """JSON message envelope between orchestrator and agent."""
    message_id: str
    sender: str
    receiver: str
    query: str
    context: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    config_override: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent: str
    content: str
    model_used: str
    sources_used: list[str] = field(default_factory=list)
    confidence: str = "high"


class BaseAgent:
    name: str = "base"

    def __init__(self, client: Any) -> None:
        self.client = client
        self._agent_cfg = agent_config(self.name)
        cfg = models_config()
        m = cfg[self.name]
        provider = get_provider()
        self.model_config = ModelConfig(
            model=m["model"],
            max_tokens=m["max_tokens"],
            temperature=m["temperature"],
            provider=provider,
        )

    def run(self, msg: AgentMessage) -> AgentResponse:
        system = self._build_system(msg.context)
        messages = list(msg.conversation_history) + [
            {"role": "user", "content": msg.query}
        ]
        content = complete(self.client, self.model_config, system, messages)
        sources = self._extract_sources(msg.context)
        return AgentResponse(
            agent=self.name,
            content=content,
            model_used=self.model_config.model,
            sources_used=sources,
        )

    def stream(self, msg: AgentMessage) -> tuple[Iterator[str], list[str]]:
        """Yield text chunks from the LLM. Returns (chunk_iterator, sources)."""
        system = self._build_system(msg.context)
        messages = list(msg.conversation_history) + [
            {"role": "user", "content": msg.query}
        ]
        sources = self._extract_sources(msg.context)
        return stream(self.client, self.model_config, system, messages), sources

    def _build_system(self, context: str) -> str:
        base = self._agent_cfg["system_prompt"]
        if context:
            return f"{base}\n\n{context}"
        return base

    def _extract_sources(self, context: str) -> list[str]:
        import re
        return re.findall(r"\*Source: ([^|]+)", context)
