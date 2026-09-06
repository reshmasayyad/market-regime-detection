import pytest
from market_regimes.evaluation import sequence_summary


def test_runs_and_switch_denominator():
    result=sequence_summary([0,0,0,1,1,0])
    assert result["switches"]==2
    assert result["switch_rate"]==pytest.approx(2/5)
    assert result["median_run_observations"]==2
