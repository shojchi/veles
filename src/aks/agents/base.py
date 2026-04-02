"""Base class for all specialist agents."""
from __future__ import annotations

from dataclasses import dataclass, field

from aks.utils.config import agent_config, models_config


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

    def __init__(self) -> None:
        self._agent_cfg = agent_config(self.name)
        m = models_config()[self.name]
        self._max_tokens: int = m["max_tokens"]
        self._temperature: float = m["temperature"]

    def run(self, msg: AgentMessage) -> AgentResponse:
        from aks.models.llm import complete_with_fallback
        system = self._build_system(msg.context)
        messages = list(msg.conversation_history) + [
            {"role": "user", "content": msg.query}
        ]
        content, provider_model = complete_with_fallback(
            system, messages, self._max_tokens, self._temperature
        )
        return AgentResponse(
            agent=self.name,
            content=content,
            model_used=provider_model,
            sources_used=self._extract_sources(msg.context),
        )

    def _build_system(self, context: str) -> str:
        base = self._agent_cfg["system_prompt"]
        if context:
            return f"{base}\n\n{context}"
        return base

    def _extract_sources(self, context: str) -> list[str]:
        import re
        return re.findall(r"\*Source: ([^|]+)", context)
