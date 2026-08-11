"""Offline end-to-end test of the tournament engine.

Market data and the model calls are stubbed, so this exercises the real
orchestration (sizing, fills, scoring, state persistence, elimination) with no
network or API keys.
"""

from datetime import date, datetime, timedelta, timezone

import modeltraining.tournament.brains as brains
from modeltraining.ai.schema import TradeDecision
from modeltraining.config import Settings
from modeltraining.context import Bar
from modeltraining.tournament.engine import TournamentEngine
from modeltraining.tournament.roster import Contestant
from modeltraining.tournament.state import ContestantState, TournamentState


def _settings() -> Settings:
    return Settings(
        alpaca_api_key="x",
        alpaca_secret_key="y",
        anthropic_api_key="z",
        symbols="AAPL",
        tournament_capital=100.0,
        tournament_min_confidence=0.5,
        tournament_max_daily_spend=1000.0,
    )


def _bars(price: float) -> list[Bar]:
    now = datetime.now(timezone.utc)
    return [Bar(now, price, price, price, price, 1000) for _ in range(20)]


def _roster() -> list[Contestant]:
    return [
        Contestant("buyer", "claude", "claude-opus-5", "buy", research=False),
        Contestant("holder", "claude", "claude-haiku-4-5", "hold", research=False),
    ]


def _run(engine, monkeypatch, decision_by_id, mark=10.0):
    monkeypatch.setattr(engine, "_market_snapshot", lambda syms: ({"AAPL": mark}, {"AAPL": _bars(mark)}))

    def fake_decide(client, model, strategy, note, ctx, competitors=""):
        # strategy string doubles as the contestant's intent here
        return decision_by_id[strategy], 0.002

    monkeypatch.setattr(brains, "decide_claude", fake_decide)
    return engine.run_day()


def test_full_cycle_buys_marks_and_persists(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    engine = TournamentEngine(_settings(), state_path=path, leaderboard_path=tmp_path / "lb.json")
    engine.init(roster=_roster())

    decisions = {
        "buy": TradeDecision(action="buy", quantity=5, confidence=0.9, rationale="go"),
        "hold": TradeDecision(action="hold", quantity=0, confidence=0.9, rationale="wait"),
    }
    result = _run(engine, monkeypatch, decisions, mark=10.0)

    assert result["day_index"] == 1
    state = TournamentState.load(path)
    buyer = next(c for c in state.contestants if c.id == "buyer")
    holder = next(c for c in state.contestants if c.id == "holder")

    # buyer spent ~5 * 10 (+slippage) on AAPL; holder stayed in cash
    assert buyer.ledger().qty("AAPL") == 5
    assert buyer.cash < 100.0
    assert holder.ledger().qty("AAPL") == 0
    assert holder.cash == 100.0

    # API cost was charged, equity + net recorded
    assert buyer.api_cost > 0
    assert len(buyer.equity_history) == 1
    # equity ~ 100 minus slippage drag; net = equity - api_cost
    assert buyer.net_equity() < buyer.equity()


def test_weekly_elimination_drops_worst_net(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    engine = TournamentEngine(_settings(), state_path=path, leaderboard_path=tmp_path / "lb.json")
    engine.init(roster=_roster())

    # buyer buys into a rising mark (gains); holder sits in cash (flat, minus API cost)
    decisions = {
        "buy": TradeDecision(action="buy", quantity=9, confidence=0.9, rationale="go"),
        "hold": TradeDecision(action="hold", quantity=0, confidence=0.9, rationale="wait"),
    }
    _run(engine, monkeypatch, decisions, mark=10.0)  # buyer buys 9 @ ~10

    # next cycle at a higher mark, force elimination
    monkeypatch.setattr(engine, "_market_snapshot", lambda syms: ({"AAPL": 15.0}, {"AAPL": _bars(15.0)}))
    monkeypatch.setattr(brains, "decide_claude",
                        lambda *a, **k: (decisions[a[2]], 0.002))
    result = engine.run_day(force_eliminate=True)

    state = TournamentState.load(path)
    alive = [c.id for c in state.contestants if c.alive]
    # buyer rode the mark from 10 -> 15 and should outrank the cash-holder
    assert result["eliminated"] == "holder"
    assert alive == ["buyer"]


# --- grace period + "5% weekly return = safe" elimination rule --------------

def _elim_state(created_days_ago: int, last_elim_days_ago, histories: dict) -> TournamentState:
    today = date.today()
    created = (today - timedelta(days=created_days_ago)).isoformat()
    last = (today - timedelta(days=last_elim_days_ago)).isoformat() if last_elim_days_ago is not None else None
    contestants = []
    for cid, hist in histories.items():
        cs = ContestantState(id=cid, provider="claude", model="claude-opus-5",
                             strategy="s", research=False)
        cs.equity_history = hist
        contestants.append(cs)
    return TournamentState(created=created, starting_capital=100.0, symbols=["AAPL"],
                           day_index=30, last_elimination=last, contestants=contestants)


def _hist(week_ago: float, now: float) -> list:
    today = date.today()
    d8 = (today - timedelta(days=8)).isoformat()
    d1 = (today - timedelta(days=1)).isoformat()
    return [[d8, week_ago, week_ago], [d1, now, now]]


def _engine():
    return TournamentEngine(_settings(), state_path="/tmp/never_written.json")


def test_no_elimination_during_grace_period():
    # created 10 days ago (< 21-day grace) -> nobody goes, even the clear loser
    state = _elim_state(10, None, {
        "leader": _hist(100, 130),
        "loser": _hist(100, 90),
    })
    assert _engine()._maybe_eliminate(state, force=False) is None


def test_after_grace_eliminates_worst_non_safe():
    # past grace, cadence due; leader is safe (+10% wk), loser only +1% wk -> loser out
    state = _elim_state(30, 8, {
        "leader": _hist(100, 110),   # +10% weekly -> safe
        "loser": _hist(100, 101),    # +1% weekly  -> eligible, and worst
    })
    out = _engine()._maybe_eliminate(state, force=False)
    assert out is not None and out.id == "loser"
    assert not next(c for c in state.contestants if c.id == "loser").alive


def test_everyone_safe_means_no_elimination():
    # both up more than 5% this week -> all safe -> no cut, even though one is lower
    state = _elim_state(30, 8, {
        "a": _hist(100, 112),   # +12%
        "b": _hist(100, 108),   # +8% (lower, but still safe)
    })
    assert _engine()._maybe_eliminate(state, force=False) is None
    assert all(c.alive for c in state.contestants)


def test_weekly_cadence_blocks_second_cut_same_week():
    # past grace but only 3 days since last elimination -> not due yet
    state = _elim_state(30, 3, {
        "a": _hist(100, 101),
        "b": _hist(100, 100),
    })
    assert _engine()._maybe_eliminate(state, force=False) is None
