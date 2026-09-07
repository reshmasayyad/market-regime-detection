"""Run from repository root: python scripts/run_analysis.py."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[name]="1"
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from market_regimes.pipeline import analyze

if __name__=="__main__": analyze(ROOT)
