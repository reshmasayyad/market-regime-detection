"""Score a complete post-fit quote sequence using the saved frozen HMM.

Input needs historical quotes for feature warmup and all valid observations from
the recorded next date onward. Filtered posteriors continue the fitted history.
"""
from pathlib import Path
import argparse
import sys
import joblib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from market_regimes.data import load_prices
from market_regimes.features import market_features,MODEL_FEATURES
from market_regimes.hmm import filter_hmm


def score(input_path,output_path,artifact_path,end="2025-12-31"):
    artifact=joblib.load(artifact_path)
    quotes,_=load_prices(input_path,analysis_end=end)
    features,_=market_features(quotes,end=end)
    after=features.loc[features.index>pd.Timestamp(artifact["fitted_through"])].copy()
    if after.empty: raise ValueError("No observations after the model fitting period")
    if str(after.index[0].date())!=artifact["expected_next_date"]:
        raise ValueError("Include all post-fit history from the model's expected next observation, with feature warmup quotes")
    values=artifact["scaler"].transform(after[MODEL_FEATURES])
    probability,scores=filter_hmm(artifact["model"],values,artifact["last_refit_posterior"])
    output=pd.DataFrame({"date":after.index,"regime":[artifact["names"][int(state)] for state in probability.argmax(axis=1)],
                         "max_filtered_probability":probability.max(axis=1),"conditional_log_score":scores})
    for state,name in artifact["names"].items(): output[f"probability_{name}"]=probability[:,state]
    output.to_csv(output_path,index=False)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv",type=Path)
    parser.add_argument("output_csv",type=Path)
    parser.add_argument("--model",type=Path,default=ROOT/"artifacts/regime_hmm.joblib")
    parser.add_argument("--end",default="2025-12-31",help="Last allowed quote date")
    args=parser.parse_args()
    output=score(args.input_csv,args.output_csv,args.model,args.end)
    print(output.tail().to_string(index=False))
