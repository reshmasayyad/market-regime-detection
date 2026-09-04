"""Gaussian HMM fitting and explicit forward-only filtering."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM


def fit_hmm(values, states, seed):
    model = GaussianHMM(n_components=states, covariance_type="diag", random_state=seed,
                        n_iter=500, tol=1e-3, min_covar=1e-3, covars_prior=.01,
                        implementation="log")
    model.fit(values)
    history = list(model.monitor_.history)
    gain = float(history[-1]-history[-2]) if len(history)>1 else np.nan
    diagnostics = {"seed": seed, "states": states, "train_total_log_likelihood": float(model.score(values)),
                   "iterations": model.monitor_.iter, "last_log_likelihood_gain": gain,
                   "iteration_limit_reached": model.monitor_.iter >= model.n_iter,
                   "numerically_acceptable": bool(np.isfinite(model.score(values)) and np.isfinite(gain) and gain >= -.01)}
    return model, diagnostics


def filter_hmm(model, values, previous_posterior=None):
    """Return P(S_t | observations through t) and conditional log scores.

    First observation uses startprob_, or previous_posterior @ transmat_ when
    continuing an existing sequence. Future emissions never enter the recursion.
    Unlike predict_proba on a complete sequence, this is not smoothing.
    """
    values = np.asarray(values, dtype=float)
    states = model.n_components
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Observations must be a finite two-dimensional matrix")
    if len(values)==0:
        return np.empty((0,states)), np.empty(0)
    if previous_posterior is None:
        prior = np.asarray(model.startprob_, dtype=float)
    else:
        previous = np.asarray(previous_posterior, dtype=float)
        if previous.shape != (states,) or (previous<0).any() or not np.isclose(previous.sum(),1):
            raise ValueError("Invalid previous filtered posterior")
        prior = previous @ model.transmat_
    emissions = np.column_stack([multivariate_normal.logpdf(values, mean=model.means_[k], cov=model.covars_[k]) for k in range(states)])
    probabilities = np.empty((len(values),states))
    log_scores = np.empty(len(values))
    for t in range(len(values)):
        with np.errstate(divide="ignore"):
            weights = np.log(prior) + emissions[t]
        normalizer = logsumexp(weights)
        if not np.isfinite(normalizer):
            raise RuntimeError("No finite predictive probability for an observation")
        posterior = np.exp(weights-normalizer)
        probabilities[t] = posterior
        log_scores[t] = normalizer
        prior = posterior @ model.transmat_
    return probabilities, log_scores
