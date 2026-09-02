import numpy as np
import pandas as pd
import pytest
from market_regimes.data import load_prices
from market_regimes.features import market_features, future_diagnostics, MODEL_FEATURES


def synthetic_quotes():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2011-01-03", periods=100)
    prices = 100*np.exp(np.cumsum(rng.normal(0.001, 0.01, len(dates))))
    return pd.DataFrame({"nifty50": prices, "bank_nifty": prices*2, "india_vix": 15.}, index=dates)


def test_future_prices_cannot_change_earlier_features():
    original = synthetic_quotes()
    changed = original.copy()
    changed.loc[changed.index[70]:, "nifty50"] *= 2
    a,_ = market_features(original)
    b,_ = market_features(changed)
    pd.testing.assert_frame_equal(a.loc[:original.index[69], MODEL_FEATURES], b.loc[:original.index[69], MODEL_FEATURES])


def test_forward_volatility_uses_exactly_next_twenty_returns():
    quotes = synthetic_quotes()
    day = quotes.index[30]
    outcomes = future_diagnostics(quotes, quotes.index)
    returns = np.log(quotes.nifty50).diff()
    expected = returns.iloc[31:51].std(ddof=1)*np.sqrt(252)
    assert outcomes.loc[day, "next_20_volatility"] == pytest.approx(expected)
    assert outcomes.next_20_volatility.tail(20).isna().all()


def test_duplicate_dates_are_rejected(tmp_path):
    frame = synthetic_quotes().reset_index(names="date")
    frame = pd.concat([frame, frame.iloc[[0]]])
    path = tmp_path / "prices.csv"
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_prices(path)
