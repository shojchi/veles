#!/usr/bin/env python3
"""AKS — Agent Knowledge System CLI."""
from __future__ import annotations

import click
from dotenv import load_dotenv

load_dotenv()


def _get_orchestrator():
    from aks.models.llm import get_client
    from aks.knowledge.store import KnowledgeStore
    from aks.orchestrator.router import Orchestrator
    from aks.utils.config import get_provider

    client = get_client(get_provider())
    store = KnowledgeStore()
    return Orchestrator(client=client, store=store)


@click.group()
def cli() -> None:
    """AKS — your personal AI assistant grounded in your notes."""


@cli.command()
@click.argument("query")
@click.option("--agent", "-a", default=None, help="Force a specific agent (e.g. code)")
def ask(query: str, agent: str | None) -> None:
    """Ask a question. Usage: aks ask 'why is my code slow?'"""
    orchestrator = _get_orchestrator()
    response = orchestrator.run(query, force_agent=agent)

    click.echo(f"\n[{response.agent} | {response.model_used}]\n")
    click.echo(response.content)

    if response.sources_used:
        click.echo("\n--- Sources ---")
        for src in response.sources_used:
            click.echo(f"  • {src.strip()}")


@cli.command()
def status() -> None:
    """Show loaded agents, models, and config."""
    from aks.utils.config import system_config, models_config, get_provider
    from aks.orchestrator.router import ACTIVE_AGENTS

    sys_cfg = system_config()
    mdl_cfg = models_config()

    click.echo("\n=== AKS Status ===")
    click.echo(f"Provider: {get_provider()}")
    click.echo(f"Notes   : {sys_cfg['notes_dir']}")
    embed_model = mdl_cfg.get("embeddings", {}).get("model", "?") if sys_cfg["retrieval"]["embeddings_enabled"] else None
    click.echo(f"Embeds  : {'enabled (' + embed_model + ')' if embed_model else 'disabled'}")
    click.echo(f"Daily $ : ${sys_cfg['cost']['daily_cap_usd']:.2f} cap")
    click.echo("\nActive agents:")
    for name in ACTIVE_AGENTS:
        m = mdl_cfg.get(name, {})
        click.echo(f"  • {name:<10} {m.get('model', '?')}  (temp={m.get('temperature', '?')})")


@cli.command()
@click.argument("title")
@click.argument("body")
def save(title: str, body: str) -> None:
    """Save a note. Usage: aks save 'Title' 'Body text'"""
    from aks.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    path = store.save_note(title=title, body=body)
    click.echo(f"Saved → {path}")


@cli.command()
@click.argument("query")
def search(query: str) -> None:
    """Search notes. Usage: aks search 'query'"""
    from aks.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    results = store.search(query)
    if not results:
        click.echo("No results found.")
        return
    for r in results:
        click.echo(f"\n[{r.score:.2f}] {r.note.title}  ({r.note.path.name})")
        click.echo(f"  {r.snippet}")


if __name__ == "__main__":
    cli()
