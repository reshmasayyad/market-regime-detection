"""Only current and prior closing observations enter modelling features."""
import numpy as np
import pandas as pd

MODEL_FEATURES = ["nifty_log_return", "log_volatility_20", "momentum_20", "log_vix"]
TRAIN_END = "2018-12-31"
VALIDATION_END = "2021-12-31"
ANALYSIS_START = "2011-01-01"
ANALYSIS_END = "2025-12-31"


def market_features(quotes, start=ANALYSIS_START, end=ANALYSIS_END):
    """Trailing 20 observed NIFTY sessions; data are available at that day's close.

    No forward fill, centered rolling window, or future-price reference enters X.
    Additional banking/correlation/drawdown columns are descriptive, not model X.
    """
    frame = quotes.copy()
    frame["nifty_log_return"] = np.log(frame.nifty50).diff()
    frame["bank_log_return"] = np.log(frame.bank_nifty).diff()
    frame["volatility_20"] = frame.nifty_log_return.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    frame["log_volatility_20"] = np.log(frame.volatility_20.where(frame.volatility_20.gt(0)))
    frame["momentum_20"] = np.log(frame.nifty50 / frame.nifty50.shift(20))
    frame["log_vix"] = np.log(frame.india_vix)
    frame["drawdown"] = frame.nifty50 / frame.nifty50.cummax() - 1
    frame["nifty_bank_correlation_20"] = frame.nifty_log_return.rolling(20, min_periods=20).corr(frame.bank_log_return)
    selected = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    valid = selected.dropna(subset=MODEL_FEATURES).copy()
    if not np.isfinite(valid[MODEL_FEATURES]).all().all():
        raise ValueError("Non-finite modelling feature")
    audit = {"rows_in_analysis_period": len(selected), "valid_feature_rows": len(valid),
             "rows_excluded_for_incomplete_features": len(selected)-len(valid),
             "excluded_feature_dates": selected.index.difference(valid.index).strftime("%Y-%m-%d").tolist(),
             "model_features": MODEL_FEATURES, "feature_timing": "available after current day's close"}
    return valid, audit


def chronological_split(features):
    train = features.loc[:TRAIN_END]
    validation = features.loc[pd.Timestamp(TRAIN_END)+pd.Timedelta(days=1):VALIDATION_END]
    test = features.loc[pd.Timestamp(VALIDATION_END)+pd.Timedelta(days=1):ANALYSIS_END]
    if any(len(part) < 100 for part in [train, validation, test]):
        raise ValueError("Each chronological split needs at least 100 valid observations")
    return train, validation, test


def future_diagnostics(quotes, feature_dates, horizon=20):
    """Future outcomes are computed separately and never included in model inputs.

    At date t, shifted trailing volatility at t+20 uses returns t+1 through t+20.
    The last horizon rows have missing outcomes and are excluded from summaries.
    Overlapping windows imply dependence; no independent-sample p-values are used.
    """
    returns = np.log(quotes.nifty50).diff()
    return pd.DataFrame({
        "next_20_volatility": returns.rolling(horizon, min_periods=horizon).std(ddof=1).shift(-horizon) * np.sqrt(252),
        "next_20_return": np.log(quotes.nifty50.shift(-horizon)/quotes.nifty50),
    }).reindex(feature_dates)
