"""Select configurations before examining the held-out test period."""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .features import MODEL_FEATURES
from .hmm import fit_hmm, filter_hmm

SEEDS = (7, 21, 42, 63, 84)


def best_training_fit(values, states, seeds=SEEDS):
    models, diagnostics = [], []
    for seed in seeds:
        model, record = fit_hmm(values, states, seed)
        models.append(model)
        diagnostics.append(record)
    table = pd.DataFrame(diagnostics)
    valid = table.loc[table.numerically_acceptable & ~table.iteration_limit_reached]
    if valid.empty:
        raise RuntimeError("No HMM run completed adequately; inspect initialization diagnostics")
    best_index = int(valid.train_total_log_likelihood.idxmax())
    return models[best_index], table, models


def select_hmm(train, validation, refit, counts=(3,4), seeds=SEEDS):
    scaler = StandardScaler().fit(train[MODEL_FEATURES])
    train_values = scaler.transform(train[MODEL_FEATURES])
    valid_values = scaler.transform(validation[MODEL_FEATURES])
    candidate_records, seed_records = [], []
    for states in counts:
        model, runs, _ = best_training_fit(train_values, states, seeds)
        seed_records.append(runs)
        train_probability, _ = filter_hmm(model, train_values)
        valid_probability, valid_scores = filter_hmm(model, valid_values, train_probability[-1])
        independent = GaussianMixture(n_components=states, covariance_type="diag", reg_covar=.001,
                                      n_init=10, max_iter=500, random_state=42).fit(train_values)
        _, sizes = np.unique(valid_probability.argmax(axis=1), return_counts=True)
        candidate_records.append({"states": states, "selected_training_seed": int(model.random_state),
                                  "validation_mean_conditional_log_score": valid_scores.mean(),
                                  "independent_gmm_validation_mean_log_score": independent.score(valid_values),
                                  "validation_states_observed": len(sizes),
                                  "minimum_validation_state_share": sizes.min()/len(valid_values)})
    candidates = pd.DataFrame(candidate_records)
    # Use all finite candidate log scores; do not reject a state just because a
    # future period never enters it. Occupancy is reported as a diagnostic.
    finite = candidates.loc[np.isfinite(candidates.validation_mean_conditional_log_score)]
    selected_states = int(finite.sort_values("validation_mean_conditional_log_score",ascending=False).iloc[0].states)
    final_scaler = StandardScaler().fit(refit[MODEL_FEATURES])
    values = final_scaler.transform(refit[MODEL_FEATURES])
    model, final_runs, final_models = best_training_fit(values, selected_states, seeds)
    independent = GaussianMixture(n_components=selected_states,covariance_type="diag",reg_covar=.001,
                                  n_init=10,max_iter=500,random_state=42).fit(values)
    return {"model": model, "scaler": final_scaler, "selected_states": selected_states,
            "independent_gmm": independent, "selection": candidates,
            "selection_initializations": pd.concat(seed_records,ignore_index=True),
            "refit_initializations": final_runs, "refit_models": final_models}
