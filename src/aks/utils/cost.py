"""Cost ledger — track token usage and enforce daily spend cap."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

from aks.utils.config import DATA_DIR, get_fallback_chain, models_config, system_config


def _pricing(provider: str) -> dict[str, float]:
    """Return pricing dict for a provider. Falls back to zero-cost."""
    for cfg in get_fallback_chain():
        if cfg["name"] == provider:
            return cfg.get("pricing", {"input_per_1m": 0.0, "output_per_1m": 0.0})
    if provider == "gemini-embedding":
        return models_config().get("embeddings", {}).get(
            "pricing", {"input_per_1m": 0.0, "output_per_1m": 0.0}
        )
    return {"input_per_1m": 0.0, "output_per_1m": 0.0}


def _compute_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    p = _pricing(provider)
    return (input_tokens * p["input_per_1m"] + output_tokens * p["output_per_1m"]) / 1_000_000


class CostLedger:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(DATA_DIR / "cost.db")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                provider      TEXT    NOT NULL,
                model         TEXT    NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd      REAL    NOT NULL
            )
        """)
        self._db.commit()

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Insert a usage row and return the cost in USD."""
        cost = _compute_cost(provider, input_tokens, output_tokens)
        self._db.execute(
            "INSERT INTO usage(ts, provider, model, input_tokens, output_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), provider, model, input_tokens, output_tokens, cost),
        )
        self._db.commit()
        return cost

    def check_cap(self) -> None:
        """Raise RuntimeError if today's spend has hit the daily cap."""
        cap = system_config()["cost"]["daily_cap_usd"]
        spent = self.today_usd()
        if spent >= cap:
            raise RuntimeError(
                f"Daily cost cap of ${cap:.2f} reached (spent ${spent:.4f} today). "
                "Run `aks cost` to review usage."
            )

    def today_usd(self) -> float:
        today = date.today().isoformat()
        row = self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE ts >= ?",
            (today,),
        ).fetchone()
        return row[0]

    def today_by_provider(self) -> list[dict]:
        today = date.today().isoformat()
        rows = self._db.execute(
            "SELECT provider, model, SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) "
            "FROM usage WHERE ts >= ? "
            "GROUP BY provider, model ORDER BY cost_usd DESC",
            (today,),
        ).fetchall()
        return [
            {
                "provider": r[0],
                "model": r[1],
                "input_tokens": r[2],
                "output_tokens": r[3],
                "cost_usd": r[4],
            }
            for r in rows
        ]

    def history(self, days: int = 7) -> list[dict]:
        rows = self._db.execute(
            "SELECT DATE(ts) AS day, SUM(cost_usd) "
            "FROM usage GROUP BY day ORDER BY day DESC LIMIT ?",
            (days,),
        ).fetchall()
        return [{"date": r[0], "cost_usd": r[1]} for r in rows]
