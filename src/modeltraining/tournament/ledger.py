"""A tiny long-only paper-trading ledger, marked to real daily closes.

No shorting, no leverage: you can only spend the cash you have and only sell
shares you hold. This is the simulated "$100 book" each contestant trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..context import AccountSnapshot, PositionSnapshot

_EPS = 1e-9


@dataclass
class Lot:
    qty: float
    avg_price: float


@dataclass
class PaperLedger:
    cash: float
    positions: dict[str, Lot] = field(default_factory=dict)

    # --- reads ---
    def qty(self, symbol: str) -> float:
        lot = self.positions.get(symbol)
        return lot.qty if lot else 0.0

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus positions marked at `marks` (falls back to avg price)."""
        total = self.cash
        for sym, lot in self.positions.items():
            total += lot.qty * marks.get(sym, lot.avg_price)
        return total

    def account_snapshot(self, marks: dict[str, float], last_equity: float) -> AccountSnapshot:
        eq = self.equity(marks)
        return AccountSnapshot(
            buying_power=self.cash,     # long-only, no margin: buying power == cash
            cash=self.cash,
            equity=eq,
            last_equity=last_equity,
            portfolio_value=eq,
        )

    def position_snapshot(self, symbol: str, mark: float) -> PositionSnapshot | None:
        lot = self.positions.get(symbol)
        if not lot or lot.qty <= 0:
            return None
        return PositionSnapshot(
            symbol=symbol,
            qty=lot.qty,
            avg_entry_price=lot.avg_price,
            current_price=mark,
            market_value=lot.qty * mark,
            unrealized_pl=(mark - lot.avg_price) * lot.qty,
        )

    # --- writes ---
    def buy(self, symbol: str, qty: int, fill_price: float) -> None:
        if qty <= 0:
            return
        cost = qty * fill_price
        if cost > self.cash + _EPS:
            raise ValueError(f"insufficient cash to buy {qty} {symbol} @ {fill_price:.2f}")
        self.cash -= cost
        lot = self.positions.get(symbol)
        if lot is None:
            self.positions[symbol] = Lot(qty=float(qty), avg_price=fill_price)
        else:
            new_qty = lot.qty + qty
            lot.avg_price = (lot.qty * lot.avg_price + qty * fill_price) / new_qty
            lot.qty = new_qty

    def sell(self, symbol: str, qty: int, fill_price: float) -> None:
        if qty <= 0:
            return
        lot = self.positions.get(symbol)
        held = lot.qty if lot else 0.0
        if qty > held + _EPS:
            raise ValueError(f"cannot sell {qty} {symbol}; only {held} held")
        self.cash += qty * fill_price
        lot.qty -= qty
        if lot.qty <= _EPS:
            del self.positions[symbol]

    # --- serialization ---
    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {s: {"qty": l.qty, "avg_price": l.avg_price} for s, l in self.positions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperLedger":
        positions = {
            s: Lot(qty=v["qty"], avg_price=v["avg_price"]) for s, v in d.get("positions", {}).items()
        }
        return cls(cash=d["cash"], positions=positions)
