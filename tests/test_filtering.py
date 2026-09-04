import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from market_regimes.hmm import filter_hmm


def fixed_model():
    model = GaussianHMM(n_components=2,covariance_type="diag")
    model.n_features=1
    model.startprob_=np.array([.6,.4])
    model.transmat_=np.array([[.9,.1],[.2,.8]])
    model.means_=np.array([[-1.],[1.]])
    model.covars_=np.array([[.4],[.7]])
    return model


def test_forward_scores_match_library_sequence_likelihood():
    model=fixed_model()
    x=np.array([[-.8],[-.5],[.7],[1.2]])
    probabilities,scores=filter_hmm(model,x)
    assert scores.sum()==pytest.approx(model.score(x),abs=1e-10)
    assert np.allclose(probabilities.sum(axis=1),1)


def test_prefix_assignments_cannot_use_future_emissions():
    model=fixed_model()
    prefix=np.array([[-.8],[-.5],[.7]])
    a,_=filter_hmm(model,prefix)
    b,_=filter_hmm(model,np.vstack([prefix,[[20.],[-20.]]]))
    assert np.allclose(a,b[:len(prefix)])


def test_continuation_equals_one_pass_filtering():
    model=fixed_model()
    x=np.array([[-.8],[-.5],[.7],[1.2]])
    whole,whole_scores=filter_hmm(model,x)
    early,first_scores=filter_hmm(model,x[:2])
    late,last_scores=filter_hmm(model,x[2:],previous_posterior=early[-1])
    assert np.allclose(late,whole[2:])
    assert np.allclose(last_scores,whole_scores[2:])
