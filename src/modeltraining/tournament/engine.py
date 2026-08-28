"""Runs the tournament: one daily decision cycle, plus periodic elimination."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

from ..ai.schema import TradeAction
from ..config import Settings
from ..context import MarketContext
from ..data.market_data import MarketDataClient
from ..risk.guardrails import RiskManager
from . import brains, pricing
from .leaderboard import DEFAULT_LEADERBOARD_PATH, write_leaderboard
from .roster import Contestant, default_roster
from .state import DEFAULT_STATE_PATH, ContestantState, TournamentState

log = logging.getLogger(__name__)


class TournamentEngine:
    def __init__(
        self,
        settings: Settings,
        state_path: Path = DEFAULT_STATE_PATH,
        leaderboard_path: Path = DEFAULT_LEADERBOARD_PATH,
    ) -> None:
        self._s = settings
        self._path = state_path
        self._lb_path = leaderboard_path
        self._data = MarketDataClient(settings)
        # Risk limits scaled so only cash binds; keep no-shorting + confidence gate.
        self._risk = RiskManager(
            settings.model_copy(
                update={
                    "max_trade_notional": 1e12,
                    "max_position_notional": 1e12,
                    "max_daily_loss": 1e12,
                    "min_confidence": settings.tournament_min_confidence,
                }
            )
        )
        self._slip = settings.tournament_slippage_bps / 10_000.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def init(self, roster: list[Contestant] | None = None, force: bool = False) -> TournamentState:
        if TournamentState.exists(self._path) and not force:
            raise RuntimeError(
                f"A tournament already exists at {self._path}. Use --force to reset it."
            )
        roster = roster or default_roster()
        state = TournamentState.new(
            created=date.today().isoformat(),
            capital=self._s.tournament_capital,
            symbols=self._s.symbol_list,
            roster=roster,
        )
        state.save(self._path)
        write_leaderboard(state, self._lb_path, self._benchmark())
        return state

    def standings(self) -> TournamentState:
        return TournamentState.load(self._path)

    # ------------------------------------------------------------------ #
    # A daily cycle                                                      #
    # ------------------------------------------------------------------ #
    def run_day(self, force_eliminate: bool = False) -> dict:
        state = TournamentState.load(self._path)
        alive = state.alive()
        if not alive:
            return {"day_index": state.day_index, "spend": 0.0, "eliminated": None, "rows": [], "note": "no contestants alive"}

        self._require_keys(alive)
        clients = self._build_clients(alive)
        marks, bars_by_sym = self._market_snapshot(state.symbols)
        if not marks:
            raise RuntimeError("No market data available for any watchlist symbol.")

        today = datetime.now(timezone.utc).date().isoformat()
        day_spend = 0.0
        capped = False

        # A single pre-trade standings snapshot so every contestant sees the same
        # rival books going into today (equity as of the last recorded close).
        snapshot = [(c.id, c.model, c.alive, c.equity()) for c in state.contestants]

        for c in alive:
            if capped:
                self._mark_only(c, marks, today, "skipped (daily spend cap)")
                continue
            competitors = self._competitor_text(snapshot, state.starting_capital, c.id)
            spent = self._run_contestant(
                c, state.symbols, marks, bars_by_sym, clients, today, competitors
            )
            day_spend += spent
            if day_spend >= self._s.tournament_max_daily_spend:
                capped = True
                log.warning("daily spend cap $%.2f reached; remaining contestants skipped",
                            self._s.tournament_max_daily_spend)

        state.day_index += 1
        eliminated = self._maybe_eliminate(state, force_eliminate)
        state.save(self._path)
        write_leaderboard(state, self._lb_path, self._benchmark())

        return {
            "day_index": state.day_index,
            "spend": day_spend,
            "eliminated": eliminated.id if eliminated else None,
            "rows": self._rows(state),
            "capped": capped,
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _require_keys(self, alive: list[ContestantState]) -> None:
        needed = {c.provider for c in alive}
        missing = []
        if "claude" in needed and not (self._s.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
            missing.append("ANTHROPIC_API_KEY (Claude contestants)")
        if "openai" in needed and not (self._s.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            missing.append("OPENAI_API_KEY (OpenAI contestants)")
        if missing:
            raise RuntimeError(
                "Missing API keys: " + "; ".join(missing) + ". Add them to .env, or "
                "edit the roster to remove those contestants."
            )

    def _build_clients(self, alive: list[ContestantState]) -> dict:
        clients: dict[str, object] = {}
        providers = {c.provider for c in alive}
        if "claude" in providers:
            import anthropic

            clients["claude"] = anthropic.Anthropic(api_key=self._s.anthropic_api_key or None)
        if "openai" in providers:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI contestants are in the field but the 'openai' package "
                    "isn't installed. Run: pip install '.[openai]'  (or start the "
                    "tournament with: tournament init --providers claude)"
                ) from exc
            clients["openai"] = OpenAI(api_key=self._s.openai_api_key or None)
        return clients

    def _benchmark(self) -> dict | None:
        """S&P 500 (SPY) YTD reference for the public leaderboard. Never raises —
        a data hiccup just drops the reference line, it does not sink the run."""
        sym = self._s.benchmark_symbol
        if not sym:
            return None
        try:
            pct = self._data.get_ytd_return(sym)
        except Exception as exc:  # network / data issue must not fail the write
            log.warning("benchmark %s YTD unavailable: %s", sym, exc)
            return None
        if pct is None:
            return None
        return {"name": self._s.benchmark_name, "symbol": sym, "return_pct": round(pct, 2)}

    def _market_snapshot(self, symbols: list[str]) -> tuple[dict, dict]:
        marks: dict[str, float] = {}
        bars_by_sym: dict[str, list] = {}
        for sym in symbols:
            bars = self._data.get_recent_bars(sym, self._s.bar_lookback_days)
            if not bars:
                log.warning("no bars for %s; skipping it this cycle", sym)
                continue
            marks[sym] = bars[-1].close
            bars_by_sym[sym] = bars
        return marks, bars_by_sym

    def _run_contestant(self, c, symbols, marks, bars_by_sym, clients, today, competitors) -> float:
        spent = 0.0
        try:
            ledger = c.ledger()
            note = ""
            if c.research and c.provider == "claude":
                note, rcost = brains.research_claude(clients["claude"], c.model, symbols, today)
                c.api_cost += rcost
                spent += rcost

            actions: list[str] = []
            prev_eq = c.equity()  # yesterday's equity, for day_pnl
            for sym in [s for s in symbols if s in marks]:
                price = marks[sym]
                acct = ledger.account_snapshot(marks, prev_eq)
                pos = ledger.position_snapshot(sym, price)
                ctx = self._context(sym, price, bars_by_sym[sym], pos, acct)

                if c.provider == "claude":
                    dec, dcost = brains.decide_claude(
                        clients["claude"], c.model, c.strategy, note, ctx, competitors
                    )
                else:
                    dec, dcost = brains.decide_openai(
                        clients["openai"], c.model, c.strategy, note, ctx, competitors
                    )
                c.api_cost += dcost
                spent += dcost

                rr = self._risk.evaluate(dec, ctx, acct, pos)
                if rr.approved and rr.quantity > 0:
                    if rr.action == TradeAction.BUY:
                        fill = price * (1 + self._slip)
                        ledger.buy(sym, rr.quantity, fill)
                        side = "buy"
                    else:
                        fill = price * (1 - self._slip)
                        ledger.sell(sym, rr.quantity, fill)
                        side = "sell"
                    c.trades.append({"date": today, "symbol": sym, "side": side,
                                     "qty": rr.quantity, "price": round(fill, 4)})
                    actions.append(f"{side} {rr.quantity} {sym}")

                if spent >= self._s.tournament_max_daily_spend:
                    actions.append("(spend cap)")
                    break

            c.write_ledger(ledger)
            eq = ledger.equity(marks)
            c.equity_history.append([today, round(eq, 4), round(eq - c.api_cost, 4)])
            c.last_note = note[:500]
            c.last_action = ", ".join(actions) if actions else "hold"
        except Exception as exc:  # one bad contestant must not sink the run
            log.warning("%s errored: %s", c.id, exc)
            self._mark_only(c, marks, today, f"error: {exc}")
        return spent

    def _mark_only(self, c, marks, today, note: str) -> None:
        """Record marked equity without any API calls (spend cap / error path)."""
        try:
            eq = c.ledger().equity(marks)
        except Exception:
            eq = c.equity()
        c.equity_history.append([today, round(eq, 4), round(eq - c.api_cost, 4)])
        c.last_action = note

    def _competitor_text(self, snapshot, starting: float, self_id: str) -> str:
        """Format the live rival standings a contestant sees when deciding."""
        alive = [row for row in snapshot if row[2]]
        ranked = sorted(alive, key=lambda r: r[3], reverse=True)
        lines = []
        for i, (cid, model, _alive, eq) in enumerate(ranked, 1):
            ret = (eq / starting - 1) * 100 if starting else 0.0
            tag = "  <-- you" if cid == self_id else ""
            lines.append(f"  {i}. {cid} [{model}]: ${eq:,.2f} ({ret:+.1f}%){tag}")
        out = sum(1 for row in snapshot if not row[2])
        if out:
            lines.append(f"  ({out} already eliminated)")
        return "\n".join(lines)

    def _context(self, sym, price, bars, position, account) -> MarketContext:
        return MarketContext(
            symbol=sym,
            as_of=datetime.now(timezone.utc),
            latest_price=price,
            bid=price,
            ask=price,
            bars=bars,
            position=position,
            account=account,
        )

    def _maybe_eliminate(self, state: TournamentState, force: bool):
        """Eliminate the worst non-safe contestant, subject to the grace period.

        - No elimination for the first `grace_days` (default 3 weeks).
        - Thereafter, at most one per `elimination_days` (a "week").
        - A contestant whose trailing-week return exceeds `safe_weekly_return`
          is safe and cannot be cut that week; if everyone is safe, no cut.
        - `force` (manual override) ignores the grace period and safety.
        """
        alive = state.alive()
        if len(alive) <= 1:
            return None

        today = date.today()
        today_iso = today.isoformat()

        if not force:
            try:
                created = date.fromisoformat(state.created)
            except ValueError:
                created = today
            if (today - created).days < self._s.tournament_grace_days:
                return None  # still inside the 3-week grace period
            if state.last_elimination:
                try:
                    last = date.fromisoformat(state.last_elimination)
                    if (today - last).days < self._s.tournament_elimination_days:
                        return None  # already eliminated someone this week
                except ValueError:
                    pass

        if force:
            candidates = alive
        else:
            thr = self._s.tournament_safe_weekly_return
            period = self._s.tournament_elimination_days
            candidates = [
                c for c in alive
                if c.weekly_return(today_iso, state.starting_capital, period) <= thr
            ]
            if not candidates:
                log.info("no elimination: every contestant is safe (weekly return > %.1f%%)", thr)
                return None

        worst = min(candidates, key=lambda c: c.net_equity())
        worst.alive = False
        worst.eliminated_on = today_iso
        state.last_elimination = today_iso
        log.info("eliminated %s (net $%.2f)", worst.id, worst.net_equity())
        return worst

    def _rows(self, state: TournamentState) -> list[dict]:
        start = state.starting_capital
        rows = []
        for c in state.contestants:
            eq = c.equity()
            net = c.net_equity()
            rows.append(
                {
                    "id": c.id,
                    "model": c.model,
                    "alive": c.alive,
                    "equity": eq,
                    "net": net,
                    "return_pct": (eq / start - 1) * 100 if start else 0.0,
                    "net_return_pct": (net / start - 1) * 100 if start else 0.0,
                    "api_cost": c.api_cost,
                    "action": c.last_action,
                }
            )
        rows.sort(key=lambda r: (r["alive"], r["net"]), reverse=True)
        return rows
