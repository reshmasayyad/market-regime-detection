"""Validate supplied daily levels without silently filling missing observations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

INPUT_COLUMNS = ["date", "nifty50", "bank_nifty", "india_vix"]


def load_prices(path, analysis_end="2025-12-31"):
    path = Path(path)
    raw = pd.read_csv(path)
    missing = set(INPUT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = raw[INPUT_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.duplicated().any():
        raise ValueError("Duplicate dates require inspection; do not silently discard them")
    frame = frame.sort_values("date").set_index("date")
    for column in INPUT_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        observed = frame[column].dropna()
        if not np.isfinite(observed).all() or observed.le(0).any():
            raise ValueError(f"{column} must contain positive finite observed levels")
    within = frame.loc[:pd.Timestamp(analysis_end)]
    quotes = within.loc[within.nifty50.notna()].copy()
    if len(quotes) < 100:
        raise ValueError("Insufficient NIFTY quote history")
    audit = {
        "source_path": str(path.name), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "rows_after_end_excluded": len(frame)-len(within),
        "union_rows_without_nifty_excluded": int(within.nifty50.isna().sum()),
        "nifty_quote_rows": len(quotes), "quote_start": str(quotes.index.min().date()),
        "quote_end": str(quotes.index.max().date()),
        "missing_on_nifty_quote_calendar": quotes.isna().sum().astype(int).to_dict(),
        "sources": {"indices": "https://www.niftyindices.com/reports/historical-data",
                    "india_vix": "https://www.nseindia.com/reports-indices-historical-vix"},
        "source_verification": "Local extract attributed by supplied coverage notes; original downloads not independently reverified",
    }
    return quotes, audit
