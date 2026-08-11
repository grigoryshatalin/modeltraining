"""Export tournament state to a public `leaderboard.json`.

This is the *producer* side of the web leaderboard: `tournament init` and every
`tournament run` regenerate this file from state. Commit + push it (or use
`tournament run --publish`) for the hosted panel to pick up the new numbers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .state import TournamentState

DEFAULT_LEADERBOARD_PATH = Path("leaderboard.json")


def build_leaderboard(state: TournamentState) -> dict:
    start = state.starting_capital
    today = datetime.now(timezone.utc).date().isoformat()
    ranked = sorted(state.contestants, key=lambda c: (c.alive, c.net_equity()), reverse=True)

    agents = []
    for rank, c in enumerate(ranked, 1):
        eq = c.equity()
        net = c.net_equity()
        agents.append(
            {
                "rank": rank,
                "id": c.id,
                "name": c.id,
                "provider": c.provider,
                "model": c.model,
                "description": c.strategy,
                "research": c.research,
                "alive": c.alive,
                "eliminated_on": c.eliminated_on,
                "starting_balance": round(start, 2),
                "balance": round(eq, 2),
                "pnl": round(eq - start, 2),
                "return_pct": round((eq / start - 1) * 100, 2) if start else 0.0,
                "net_balance": round(net, 2),
                "net_return_pct": round((net / start - 1) * 100, 2) if start else 0.0,
                "weekly_return_pct": round(c.weekly_return(today, start), 2),
                "api_cost": round(c.api_cost, 4),
                "last_action": c.last_action,
            }
        )

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "day_index": state.day_index,
        "starting_capital": round(start, 2),
        "symbols": state.symbols,
        "alive_count": len(state.alive()),
        "total_count": len(state.contestants),
        "agents": agents,
    }


def write_leaderboard(state: TournamentState, path: Path = DEFAULT_LEADERBOARD_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(build_leaderboard(state), indent=2) + "\n")
    return path
