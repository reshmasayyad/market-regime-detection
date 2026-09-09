"""Chronological market regime workflow with held-out evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_prices
from .features import market_features,chronological_split,future_diagnostics,MODEL_FEATURES,VALIDATION_END
from .baselines import fit_kmeans,kmeans_labels,rule_threshold,rule_labels
from .selection import select_hmm
from .hmm import filter_hmm
from .evaluation import state_names,named_labels,state_profiles,sequence_summary,initialization_robustness
from .reporting import save_results,make_figures


def analyze(root,verbose=True):
    root=Path(root)
    def progress(message):
        if verbose: print(message,flush=True)
    quotes,data_audit=load_prices(root/"data/input/market_prices.csv")
    features,feature_audit=market_features(quotes)
    train,validation,test=chronological_split(features)
    refit=features.loc[:VALIDATION_END]
    progress(f"Chronological observations: train={len(train)}, validation={len(validation)}, test={len(test)}")
    km=fit_kmeans(train,validation,refit)
    progress("Fitting HMM candidates across five initialization seeds...")
    hmm=select_hmm(train,validation,refit)
    progress(f"Selected {hmm['selected_states']} HMM states; refit is frozen through {VALIDATION_END}.")
    result=evaluate_models(root,quotes,data_audit,features,feature_audit,train,validation,test,km,hmm)
    progress("Saved held-out diagnostics, seven figures, daily local assignments, and fitted HMM artifact.")
    return result


def evaluate_models(root,quotes,data_audit,features,feature_audit,train,validation,test,km,hmm):
    """Evaluate already fitted models; shared by the notebook and CLI workflow."""
    root=Path(root)
    refit=features.loc[:VALIDATION_END]
    refit_values=hmm["scaler"].transform(refit[MODEL_FEATURES])
    test_values=hmm["scaler"].transform(test[MODEL_FEATURES])
    refit_probability,_=filter_hmm(hmm["model"],refit_values)
    test_probability,test_scores=filter_hmm(hmm["model"],test_values,refit_probability[-1])
    names=state_names(refit,refit_probability.argmax(axis=1))
    if len(names)!=hmm["selected_states"]: raise RuntimeError("A fitted state lacks observations for interpretation")
    refit_labels=named_labels(refit_probability.argmax(axis=1),refit.index,names)
    test_labels=named_labels(test_probability.argmax(axis=1),test.index,names)
    test_probability_frame=pd.DataFrame({name:test_probability[:,component] for component,name in names.items()},index=test.index)
    test_probability_frame=test_probability_frame[sorted(names.values())]
    future=future_diagnostics(quotes,features.index)
    fit_profiles=state_profiles(refit,refit_labels)
    test_profiles=state_profiles(test,test_labels,future)
    threshold=rule_threshold(train)
    rules=rule_labels(test,threshold)
    km_test=kmeans_labels(km,test)
    comparisons=pd.DataFrame([
        {"method":"Rules",**sequence_summary(rules)},
        {"method":"KMeans",**sequence_summary(km_test)},
        {"method":"HMM",**sequence_summary(test_labels)}])
    gmm_scores=hmm["independent_gmm"].score_samples(test_values)
    yearly=pd.DataFrame({"year":test.index.year,"hmm_log_score":test_scores,"independent_gmm_log_score":gmm_scores}).groupby("year").agg(
        observations=("hmm_log_score","size"),hmm_mean_log_score=("hmm_log_score","mean"),gmm_mean_log_score=("independent_gmm_log_score","mean")).reset_index()
    robustness=initialization_robustness(hmm,refit,test)
    all_values=hmm["scaler"].transform(features[MODEL_FEATURES])
    smoothed=hmm["model"].predict_proba(all_values)[-len(test):]
    smoothing_disagreement=float((smoothed.argmax(axis=1)!=test_probability.argmax(axis=1)).mean())
    summary={"train_period":[str(train.index.min().date()),str(train.index.max().date())],
             "validation_period":[str(validation.index.min().date()),str(validation.index.max().date())],
             "test_period":[str(test.index.min().date()),str(test.index.max().date())],
             "split_observations":{"train":len(train),"validation":len(validation),"test":len(test)},
             "selected_hmm_states":hmm["selected_states"],"selected_kmeans_clusters":km["selected_k"],
             "hmm_covariance":"diagonal","selected_refit_seed":int(hmm["model"].random_state),
             "rule_annualized_volatility_threshold":threshold,
             "test_hmm_mean_conditional_log_score":float(test_scores.mean()),
             "test_independent_gmm_mean_log_score":float(gmm_scores.mean()),
             "test_hmm_mean_max_filter_probability":float(test_probability.max(axis=1).mean()),
             "test_filtered_smoothed_disagreement_share":smoothing_disagreement,
             "initialization_median_test_ari":float(robustness.test_ari_to_selected_fit.median()),
             "feature_timing":"available after current daily close",
             "likelihood_units":"nats per observation in the fixed standardized feature space"}
    timeline=features.copy()
    timeline["split"]=np.where(timeline.index<=pd.Timestamp("2018-12-31"),"train",np.where(timeline.index<=pd.Timestamp(VALIDATION_END),"validation","test"))
    timeline["hmm_regime"]=pd.concat([refit_labels,test_labels])
    all_probability=np.vstack([refit_probability,test_probability])
    for component,name in names.items(): timeline[f"probability_{name}"]=all_probability[:,component]
    timeline["rule_regime"]=rule_labels(features,threshold)
    timeline["kmeans_component"]=kmeans_labels(km,features)
    timeline=timeline.join(future)
    result={"quotes":quotes,"features":features,"train":train,"validation":validation,"refit":refit,"test":test,
            "data_audit":data_audit,"feature_audit":feature_audit,"km":km,"hmm":hmm,"names":names,
            "refit_probability":refit_probability,"test_probability_frame":test_probability_frame,
            "test_labels":test_labels,"refit_labels":refit_labels,"fit_profiles":fit_profiles,"test_profiles":test_profiles,
            "comparison":comparisons,"yearly_scores":yearly,"robustness":robustness,"summary":summary,"timeline":timeline}
    save_results(root,result)
    make_figures(root,result)
    return result
