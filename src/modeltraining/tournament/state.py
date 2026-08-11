"""Tournament state: persisted to JSON so the competition survives across days."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .ledger import PaperLedger
from .roster import Contestant

DEFAULT_STATE_PATH = Path("state/tournament.json")


@dataclass
class ContestantState:
    id: str
    provider: str
    model: str
    strategy: str
    research: bool
    alive: bool = True
    eliminated_on: str | None = None
    cash: float = 100.0
    positions: dict = field(default_factory=dict)  # sym -> {qty, avg_price}
    api_cost: float = 0.0
    equity_history: list = field(default_factory=list)  # [ [date, equity, net_equity] ]
    last_note: str = ""
    last_action: str = ""
    trades: list = field(default_factory=list)  # [ {date, symbol, side, qty, price} ]

    def ledger(self) -> PaperLedger:
        return PaperLedger.from_dict({"cash": self.cash, "positions": self.positions})

    def write_ledger(self, ledger: PaperLedger) -> None:
        d = ledger.to_dict()
        self.cash = d["cash"]
        self.positions = d["positions"]

    def equity(self) -> float:
        return self.equity_history[-1][1] if self.equity_history else self.cash

    def net_equity(self) -> float:
        return self.equity_history[-1][2] if self.equity_history else self.cash - self.api_cost

    def weekly_return(self, today: str, starting_capital: float, days: int = 7) -> float:
        """Net return (%) over the trailing `days`, vs the book `days` ago."""
        if not self.equity_history:
            return 0.0
        try:
            target = date.fromisoformat(today) - timedelta(days=days)
        except ValueError:
            return 0.0
        base = None
        for entry in self.equity_history:  # chronological
            try:
                if date.fromisoformat(entry[0]) <= target:
                    base = entry[2]
            except (ValueError, IndexError):
                continue
        if base is None:            # less than `days` of history -> compare to start
            base = starting_capital
        now = self.equity_history[-1][2]
        return (now / base - 1) * 100 if base else 0.0


@dataclass
class TournamentState:
    created: str
    starting_capital: float
    symbols: list[str]
    day_index: int = 0
    last_elimination: str | None = None
    contestants: list[ContestantState] = field(default_factory=list)

    # --- lifecycle ---
    @classmethod
    def new(cls, created: str, capital: float, symbols: list[str], roster: list[Contestant]) -> "TournamentState":
        contestants = [
            ContestantState(
                id=c.id,
                provider=c.provider,
                model=c.model,
                strategy=c.strategy,
                research=c.research,
                cash=capital,
            )
            for c in roster
        ]
        return cls(
            created=created,
            starting_capital=capital,
            symbols=list(symbols),
            last_elimination=created,
            contestants=contestants,
        )

    def alive(self) -> list[ContestantState]:
        return [c for c in self.contestants if c.alive]

    # --- persistence ---
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TournamentState":
        contestants = [ContestantState(**c) for c in d.get("contestants", [])]
        d = dict(d)
        d["contestants"] = contestants
        return cls(**d)

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "TournamentState":
        return cls.from_dict(json.loads(path.read_text()))

    @staticmethod
    def exists(path: Path = DEFAULT_STATE_PATH) -> bool:
        return path.exists()
