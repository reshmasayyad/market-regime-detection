import pandas as pd
from market_regimes.baselines import rule_labels


def test_rules_keep_risk_and_direction_separate():
    features = pd.DataFrame({"volatility_20": [.1,.1,.3,.3], "momentum_20": [.02,-.02,.02,-.02]})
    assert rule_labels(features,.2).tolist()==["Calm rising","Calm falling","Volatile rising","Volatile falling"]
