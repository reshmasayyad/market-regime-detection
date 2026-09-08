"""Validate continuation against the held-out pipeline output, when built."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from score_prices import score


def test_saved_model_reproduces_heldout_labels(tmp_path):
    artifact=ROOT/"artifacts/regime_hmm.joblib"
    source=ROOT/"data/input/market_prices.csv"
    reference=ROOT/"data/processed/daily_regimes.csv"
    if not all(path.exists() for path in [artifact,source,reference]):
        pytest.skip("Run the full analysis with local input data first")
    result=score(source,tmp_path/"scored.csv",artifact)
    expected=pd.read_csv(reference,parse_dates=["date"]).set_index("date")
    heldout=expected.loc[expected.split.eq("test")]
    assert result.regime.tolist()==heldout.hmm_regime.tolist()
    for name in result.filter(regex="^probability_").columns:
        assert np.allclose(result[name],heldout[name])
