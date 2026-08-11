import pytest

from modeltraining.tournament.ledger import PaperLedger


def test_buy_updates_cash_and_position():
    led = PaperLedger(cash=100.0)
    led.buy("AAPL", 2, 20.0)
    assert led.cash == pytest.approx(60.0)
    assert led.qty("AAPL") == 2
    assert led.positions["AAPL"].avg_price == pytest.approx(20.0)


def test_buy_averages_entry_price():
    led = PaperLedger(cash=100.0)
    led.buy("AAPL", 1, 10.0)
    led.buy("AAPL", 1, 30.0)
    assert led.qty("AAPL") == 2
    assert led.positions["AAPL"].avg_price == pytest.approx(20.0)


def test_cannot_overspend():
    led = PaperLedger(cash=15.0)
    with pytest.raises(ValueError):
        led.buy("AAPL", 1, 20.0)


def test_sell_returns_cash_and_closes_lot():
    led = PaperLedger(cash=100.0)
    led.buy("AAPL", 2, 20.0)   # cash 60, 2 shares
    led.sell("AAPL", 2, 25.0)  # +50 cash, position closed
    assert led.cash == pytest.approx(110.0)
    assert "AAPL" not in led.positions


def test_cannot_sell_more_than_held():
    led = PaperLedger(cash=100.0)
    led.buy("AAPL", 1, 20.0)
    with pytest.raises(ValueError):
        led.sell("AAPL", 2, 25.0)


def test_equity_marks_to_market():
    led = PaperLedger(cash=50.0)
    led.buy("AAPL", 2, 20.0)  # cash 10, 2 shares
    assert led.equity({"AAPL": 30.0}) == pytest.approx(10.0 + 60.0)


def test_round_trip_serialization():
    led = PaperLedger(cash=40.0)
    led.buy("MSFT", 3, 10.0)
    restored = PaperLedger.from_dict(led.to_dict())
    assert restored.cash == pytest.approx(led.cash)
    assert restored.qty("MSFT") == 3
