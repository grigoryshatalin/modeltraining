from datetime import datetime, timezone

from modeltraining.ai.schema import TradeAction, TradeDecision
from modeltraining.config import Settings
from modeltraining.context import AccountSnapshot, MarketContext, PositionSnapshot
from modeltraining.risk.guardrails import RiskManager


def make_settings(**over) -> Settings:
    base = dict(
        max_trade_notional=1000.0,
        max_position_notional=5000.0,
        max_daily_loss=500.0,
        min_confidence=0.6,
    )
    base.update(over)
    return Settings(**base)


def make_context(price: float = 100.0) -> MarketContext:
    return MarketContext(
        symbol="AAPL",
        as_of=datetime.now(timezone.utc),
        latest_price=price,
        bid=price - 0.01,
        ask=price + 0.01,
    )


def make_account(**over) -> AccountSnapshot:
    base = dict(
        buying_power=100_000.0,
        cash=100_000.0,
        equity=100_000.0,
        last_equity=100_000.0,
        portfolio_value=100_000.0,
    )
    base.update(over)
    return AccountSnapshot(**base)


def decision(action, quantity=10, confidence=0.9) -> TradeDecision:
    return TradeDecision(action=action, quantity=quantity, confidence=confidence, rationale="t")


def test_hold_is_never_an_order():
    rm = RiskManager(make_settings())
    r = rm.evaluate(decision(TradeAction.HOLD, quantity=0), make_context(), make_account(), None)
    assert not r.approved
    assert r.quantity == 0


def test_low_confidence_is_rejected():
    rm = RiskManager(make_settings(min_confidence=0.8))
    r = rm.evaluate(decision(TradeAction.BUY, confidence=0.5), make_context(), make_account(), None)
    assert not r.approved
    assert "confidence" in r.reason


def test_buy_clamped_by_per_trade_notional():
    # price 100, max_trade_notional 1000 -> at most 10 shares, even though model wants 50
    rm = RiskManager(make_settings())
    r = rm.evaluate(decision(TradeAction.BUY, quantity=50), make_context(100.0), make_account(), None)
    assert r.approved
    assert r.quantity == 10


def test_buy_clamped_by_buying_power():
    rm = RiskManager(make_settings())
    acct = make_account(buying_power=350.0)
    r = rm.evaluate(decision(TradeAction.BUY, quantity=10), make_context(100.0), acct, None)
    assert r.approved
    assert r.quantity == 3


def test_buy_blocked_when_position_at_notional_limit():
    rm = RiskManager(make_settings(max_position_notional=5000.0))
    pos = PositionSnapshot(
        symbol="AAPL", qty=50, avg_entry_price=100, current_price=100,
        market_value=5000.0, unrealized_pl=0.0,
    )
    r = rm.evaluate(decision(TradeAction.BUY, quantity=5), make_context(100.0), make_account(), pos)
    assert not r.approved
    assert "position notional" in r.reason


def test_daily_loss_kill_switch_blocks_buys():
    rm = RiskManager(make_settings(max_daily_loss=500.0))
    acct = make_account(equity=99_000.0, last_equity=100_000.0)  # day_pnl = -1000
    r = rm.evaluate(decision(TradeAction.BUY, quantity=1), make_context(100.0), acct, None)
    assert not r.approved
    assert "daily loss" in r.reason


def test_sell_without_position_is_blocked():
    rm = RiskManager(make_settings())
    r = rm.evaluate(decision(TradeAction.SELL, quantity=5), make_context(100.0), make_account(), None)
    assert not r.approved
    assert "shorting disabled" in r.reason


def test_sell_clamped_to_shares_held():
    rm = RiskManager(make_settings())
    pos = PositionSnapshot(
        symbol="AAPL", qty=5, avg_entry_price=90, current_price=100,
        market_value=500.0, unrealized_pl=50.0,
    )
    r = rm.evaluate(decision(TradeAction.SELL, quantity=20), make_context(100.0), make_account(), pos)
    assert r.approved
    assert r.quantity == 5
