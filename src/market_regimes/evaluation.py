"""Descriptive state profiles and genuinely held-out chronological diagnostics."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from .features import MODEL_FEATURES
from .hmm import filter_hmm


def state_names(train_features, labels, prefix="R"):
    levels=train_features.assign(state=labels).groupby("state").volatility_20.mean().sort_values()
    return {int(state):f"{prefix}{rank+1}" for rank,state in enumerate(levels.index)}


def named_labels(raw_labels, dates, names, label_name="regime"):
    return pd.Series([names[int(label)] for label in raw_labels],index=dates,name=label_name)


def sequence_summary(labels):
    labels=pd.Series(np.asarray(labels))
    if len(labels)<2:
        return {"observations":len(labels),"switches":0,"switch_rate":np.nan,"median_run_observations":float(len(labels))}
    changes=labels.ne(labels.shift())
    lengths=labels.groupby(changes.cumsum()).size()
    switches=int(changes.iloc[1:].sum())
    return {"observations":len(labels),"switches":switches,"switch_rate":switches/(len(labels)-1),
            "median_run_observations":float(lengths.median()),"mean_run_observations":float(lengths.mean())}


def state_profiles(features, labels, outcomes=None):
    frame=features.join(labels)
    if outcomes is not None:
        frame=frame.join(outcomes)
    profiles=frame.groupby(labels.name).agg(
        observations=("nifty_log_return","size"),mean_daily_log_return=("nifty_log_return","mean"),
        mean_trailing_volatility=("volatility_20","mean"),median_vix=("india_vix","median"),
        mean_momentum_20=("momentum_20","mean"),mean_drawdown=("drawdown","mean"),
        mean_nifty_bank_correlation=("nifty_bank_correlation_20","mean"))
    profiles["share"]=profiles.observations/profiles.observations.sum()
    if outcomes is not None:
        grouped=frame.groupby(labels.name)
        profiles["future_outcome_count"]=grouped.next_20_volatility.count()
        profiles["mean_next_20_volatility"]=grouped.next_20_volatility.mean()
        profiles["median_next_20_volatility"]=grouped.next_20_volatility.median()
        profiles["mean_next_20_log_return"]=grouped.next_20_return.mean()
    return profiles


def initialization_robustness(bundle, refit_features, test_features):
    scaler=bundle["scaler"]
    reference_train,_=filter_hmm(bundle["model"],scaler.transform(refit_features[MODEL_FEATURES]))
    reference_test,_=filter_hmm(bundle["model"],scaler.transform(test_features[MODEL_FEATURES]),reference_train[-1])
    ref_labels=reference_test.argmax(axis=1)
    rows=[]
    for model,record in zip(bundle["refit_models"],bundle["refit_initializations"].to_dict("records")):
        fit_prob,_=filter_hmm(model,scaler.transform(refit_features[MODEL_FEATURES]))
        probability,scores=filter_hmm(model,scaler.transform(test_features[MODEL_FEATURES]),fit_prob[-1])
        rows.append({**record,"test_ari_to_selected_fit":adjusted_rand_score(ref_labels,probability.argmax(axis=1)),
                     "test_mean_log_score":scores.mean()})
    return pd.DataFrame(rows)


def ordered_transition_table(model, names):
    order=sorted(names,key=lambda key:names[key])
    labels=[names[state] for state in order]
    return pd.DataFrame(model.transmat_[np.ix_(order,order)],index=labels,columns=labels)


def expected_durations(model, names):
    rows=[]
    for state,name in sorted(names.items(),key=lambda item:item[1]):
        persistence=float(model.transmat_[state,state])
        rows.append({"regime":name,"self_transition_probability":persistence,
                     "model_expected_duration_observations":1/(1-persistence) if persistence<1 else np.inf})
    return pd.DataFrame(rows).set_index("regime")
