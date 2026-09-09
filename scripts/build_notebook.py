"""Generate the explanatory market regime notebook."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# Market Regime Detection: Indian Equity Markets

**Question:** Can we describe different market environments, model their persistence, and evaluate the results without future-data leakage?

This project uses NIFTY 50, NIFTY BANK and India VIX closing observations. It compares simple risk/trend rules, K-means, and a Gaussian Hidden Markov Model (HMM). An independent Gaussian mixture provides a density baseline for evaluating whether temporal modelling adds information.

There is no authoritative daily regime label. We therefore evaluate likelihood, persistence, initialization sensitivity and subsequent observed volatility rather than supervised accuracy. No trading profitability is claimed.

**Run:** extract the repository ZIP, install `requirements.txt`, and open this notebook inside the repository. The ZIP includes `data/input/market_prices.csv`; a Git clone requires supplying that local input. All computation is CPU-only. The saved outputs were produced by actually executing the notebook.
""")
code("""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "src/market_regimes").exists():
    ROOT = ROOT.parent
assert (ROOT / "src/market_regimes").exists(), "Open inside the extracted repository."
sys.path.insert(0, str(ROOT / "src"))
from market_regimes.data import load_prices
from market_regimes.features import market_features, chronological_split, MODEL_FEATURES, VALIDATION_END
from market_regimes.baselines import fit_kmeans, rule_threshold, rule_labels
from market_regimes.selection import select_hmm
from market_regimes.hmm import filter_hmm
from market_regimes.pipeline import evaluate_models
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")
md("""
## 1. Inspect the supplied daily market data

The supplied aligned extract is attributed by its coverage notes to [NIFTY Indices historical data](https://www.niftyindices.com/reports/historical-data) and [NSE historical India VIX](https://www.nseindia.com/reports-indices-historical-vix). This project validates the local extract; it does not independently redownload and verify every exchange quote.

We restrict observations to dates through 31 December 2025. NIFTY and BANK columns are index closing levels. India VIX is an implied-volatility index, not the historical return volatility we will calculate. No gold or bond data enters the model.

Dates must be unique. Non-positive or non-finite observed values are rejected. Union-calendar rows without NIFTY quotes are removed. Missing auxiliary observations remain missing; there is no forward fill.
""")
code("""
quotes, data_audit = load_prices(ROOT / "data/input/market_prices.csv")
display(quotes.head())
display(pd.Series(data_audit, name="Input audit"))
""")
md(r"""
## 2. Create features available at the current day's close

For NIFTY closing level $P_t$:

$$r_t=\log(P_t/P_{t-1}),\qquad
\sigma_{t,20}=\sqrt{252}\,\operatorname{sd}(r_{t-19},\ldots,r_t),\qquad
m_{t,20}=\log(P_t/P_{t-20}).$$

The four model inputs are current log return, log 20-session realized volatility, 20-session log momentum, and log India VIX. Log transforms compress skewed positive volatility values. Standardization will be fitted only to the permitted historical training period.

NIFTY–BANK rolling correlation and NIFTY drawdown are additional descriptive columns, not HMM fitting inputs. BANK data therefore enriches interpretation without increasing the model's fitted feature dimension.

Features use only closing observations on or before date $t$. They are unavailable before that close. Decisions made earlier must lag them. Overlapping rolling windows create serial dependence; some observed regime persistence is a consequence of this feature design.
""")
code("""
features, feature_audit = market_features(quotes)
display(features[MODEL_FEATURES + ["volatility_20", "nifty_bank_correlation_20"]].head())
display(pd.Series(feature_audit, name="Feature audit"))
""")
md("""
## 3. Fix chronological training, validation and test periods

| Period | Use |
|---|---|
| 2011–2018 | Initial model fitting and initial scaler estimation |
| 2019–2021 | Configuration selection and independent diagnostics |
| 2011–2021 | Refit the selected configurations before testing |
| 2022–2025 | Frozen-model held-out evaluation |

The 2019–2021 validation period contains unusually volatile observations, making extrapolation visible. We do not use a shuffled train/test split. Test data does not select the number of states, scaling, or initialization seed.

Fit-period profiles are retrospective because parameters are estimated using that whole fitting period. Only the post-2021 labels are genuinely held out. Filtered test probabilities update with current observations, while parameters remain frozen.
""")
code("""
train, validation, test = chronological_split(features)
refit = features.loc[:VALIDATION_END]
display(pd.DataFrame([
    {"split": name, "start": part.index.min().date(), "end": part.index.max().date(), "observations": len(part)}
    for name, part in [("train", train), ("validation", validation), ("test", test)]
]))
assert train.index.max() < validation.index.min() < test.index.min()
""")
md("""
## 4. Implement rules and a K-means baseline

The rules combine two dimensions: elevated trailing volatility versus lower volatility, and positive versus negative trailing momentum. The volatility threshold is the 70th percentile of **initial training** volatility. This yields calm/rising, calm/falling, volatile/rising and volatile/falling descriptions. Calm and bullish are not inherently mutually exclusive concepts.

K-means groups similar standardized feature observations independently of transition order. Compare three and four clusters using validation silhouette, subject to minimum validation cluster-size checks, then refit the selected K to the 2011–2021 period.

Silhouette describes geometric separation, not regime accuracy. Rules and K-means supply useful baselines even if their output changes more frequently than an HMM's.
""")
code("""
threshold = rule_threshold(train)
rules = rule_labels(test, threshold)
print("Training-calibrated annualized volatility threshold:", f"{threshold:.2%}")
display(rules.value_counts())
km = fit_kmeans(train, validation, refit)
display(km["selection"])
print("Selected K-means clusters:", km["selected_k"])
""")
md(r"""
## 5. Fit and select a Gaussian Hidden Markov Model

Let $S_t$ be an unobserved state and $X_t$ the standardized feature vector. An HMM assumes:

$$P(S_t=j\mid S_{t-1}=i)=A_{ij},\qquad
X_t\mid S_t=j\sim\mathcal{N}(\mu_j,\Sigma_j).$$

Unlike K-means, the HMM represents both emission distributions and state transitions. We use diagonal emission covariances to keep complexity manageable. Conditional independence and Gaussian emissions are approximations, especially because rolling features overlap and financial returns have heavy tails.

For each candidate state count, fit five initializations and select an adequately completed fit with the highest **training** likelihood. Select three versus four states using mean conditional validation log score. The validation sequence continues from the last training filtered posterior rather than restarting with an unrelated state prior.

Then refit the selected state count on 2011–2021 across five initializations. Again, select the final seed using fitting-period likelihood. A seed's subsequent test result is reported only as a diagnostic and cannot change that choice.

The independent Gaussian mixture uses the same transformed feature dimension and the same component count. It ignores temporal transitions and provides a predictive-density baseline. This is not an equal-parameter-count comparison: the HMM additionally estimates transition parameters.
""")
code("""
hmm = select_hmm(train, validation, refit)
display(hmm["selection"])
display(hmm["refit_initializations"])
print("Selected HMM states:", hmm["selected_states"])
print("Selected refit seed:", hmm["model"].random_state)
""")
md(r"""
## 6. Filter states without using future emissions

With $\alpha_{t-1}(i)=P(S_{t-1}=i\mid X_{1:t-1})$, first predict the state distribution:

$$q_t(j)=\sum_i\alpha_{t-1}(i)A_{ij}.$$

Then update it using the current feature observation:

$$\alpha_t(j)=\frac{q_t(j)\,p(X_t\mid S_t=j)}{\sum_k q_t(k)\,p(X_t\mid S_t=k)}.$$

The denominator also supplies the predictive observation likelihood. Computation uses log probabilities and log-sum-exp to avoid numerical underflow.

This is **filtering**, $P(S_t\mid X_{1:t})$. Calling `predict_proba` on a complete HMM sequence normally computes smoothed probabilities using future observations. Viterbi also finds a whole-sequence path. Those retrospective tools should not be presented as same-day live assignments.

Missing feature dates are excluded. The model's transitions and durations are measured between valid feature observations, not calendar days. A different treatment of missing emissions could model the extra hidden-state transition explicitly.
""")
code("""
refit_values = hmm["scaler"].transform(refit[MODEL_FEATURES])
test_values = hmm["scaler"].transform(test[MODEL_FEATURES])
refit_probability, refit_scores = filter_hmm(hmm["model"], refit_values)
test_probability, test_scores = filter_hmm(hmm["model"], test_values, refit_probability[-1])
assert np.allclose(test_probability.sum(axis=1), 1)
assert np.isclose(refit_scores.sum(), hmm["model"].score(refit_values))
prefix_probability, _ = filter_hmm(hmm["model"], test_values[:100], refit_probability[-1])
assert np.allclose(prefix_probability, test_probability[:100])
print("Verified probability normalization, sequence likelihood and prefix invariance.")
""")
md("""
## 7. Evaluate already fitted models on held-out observations

Regime IDs R1–R4 are ordered by their **fitting-period** average trailing volatility. This makes labels interpretable without assuming they correspond to four externally true market states. Later observations do not redefine the labels.

Compare hard-label switching and run lengths, mean conditional log scores, state occupancy and initialization sensitivity. Lower switching indicates smoother state descriptions but does not, by itself, prove better detection.

Future 20-session volatility and returns are calculated separately after fitting. They never enter the model input matrix. Because future windows overlap, their summaries are descriptive and no naive independent-sample significance test is reported.
""")
code("""
result = evaluate_models(ROOT, quotes, data_audit, features, feature_audit, train, validation, test, km, hmm)
display(pd.Series(result["summary"], name="Held-out summary"))
display(result["comparison"])
display(result["yearly_scores"])
""")
md("""
## 8. Interpret state profiles and subsequent volatility

Fitting-period profiles describe the state definitions. Held-out profiles describe what occurred when those definitions were applied later. A low average current volatility state can still contain an unusually large return: no Gaussian market-state model guarantees that every observation is typical.

Inspect subsequent volatility separately from current trailing volatility. A high current-volatility state need not have the highest future volatility: some high-volatility episodes are short-lived or followed by normalization. Rare-state outcomes are also based on small samples.
""")
code("""
display(result["fit_profiles"])
display(result["test_profiles"])
display(pd.read_csv(ROOT / "results/model_expected_durations.csv"))
""")
md("""
## 9. Examine initialization sensitivity and filtering versus smoothing

ARI compares partitions without requiring identical numeric state IDs. Each initialization is fitted on the same pre-test observations; its test assignment is compared with the training-selected reference fit. This is sensitivity to optimization initialization, not a bootstrap confidence interval or supervised accuracy metric.

The smoothed-versus-filtered disagreement share is reported as a diagnostic of how retrospective inference can differ from operational inference. Smoothed assignments are not used in the held-out performance summaries or the saved daily regime timeline.
""")
code("""
display(result["robustness"])
print("Filtered/smoothed test-label disagreement:",
      f"{result['summary']['test_filtered_smoothed_disagreement_share']:.2%}")
print("Selected seed was chosen using fitting-period likelihood, even if another seed scores better on test.")
""")
md("""
## 10. Inspect the figures

The first plot shows the full data overview with chronological boundaries. The market-state chart and probability chart use only genuinely held-out dates. The transition matrix and duration estimates come from the fitting period.
""")
code("""
for name in ["01_input_overview.png", "02_validation_selection.png", "03_heldout_market_regimes.png",
             "04_filtered_probabilities.png", "05_transition_matrix.png", "06_test_state_profiles.png",
             "07_persistence_and_robustness.png"]:
    display(Image(filename=str(ROOT / "results/figures" / name)))
""")
md("""
## 11. Use the saved frozen model

`artifacts/regime_hmm.joblib` contains the fitted model, scaler, state-name mapping, feature definitions, cutoff date and last fitting-period filtered posterior. To continue the sequence correctly, the input needs historical quotes for feature warmup and the entire post-fit observation history from the expected next date.

```bash
python scripts/score_prices.py data/input/market_prices.csv data/processed/scored_quotes.csv
```

This command scores observations without re-estimating parameters. It is an offline reproducible scoring interface, not a deployed trading service. Applying the model to much later or materially changed markets requires renewed validation.
""")
code("""
sys.path.insert(0, str(ROOT / "scripts"))
from score_prices import score
scored = score(ROOT / "data/input/market_prices.csv", ROOT / "data/processed/scored_quotes.csv",
               ROOT / "artifacts/regime_hmm.joblib")
display(scored.tail())
assert scored.regime.tolist() == result["test_labels"].tolist()
""")
md("""
## 12. Findings and practical limits

The following statements are produced from the actual results. This ensures that they update when the model or data changes.
""")
code("""
s = result["summary"]
switches = result["comparison"].set_index("method")
print(f"Selected {s['selected_hmm_states']} HMM states and {s['selected_kmeans_clusters']} K-means clusters.")
print(f"Evaluated {s['split_observations']['test']:,} held-out feature observations.")
print(f"Hard-label switching: HMM {switches.loc['HMM','switch_rate']:.2%}; K-means {switches.loc['KMeans','switch_rate']:.2%}; rules {switches.loc['Rules','switch_rate']:.2%}.")
print(f"Mean held-out density score: HMM {s['test_hmm_mean_conditional_log_score']:.3f}; independent mixture {s['test_independent_gmm_mean_log_score']:.3f} nats/observation.")
print(f"Median initialization ARI to selected fit: {s['initialization_median_test_ari']:.3f}.")
print("The highest current-volatility state does not necessarily imply the highest subsequent volatility.")
""")
md("""
**What the project demonstrates:** careful financial time-series preparation, chronological model selection, probabilistic state inference, explicit forward filtering, comparison with baselines, and meaningful robustness diagnostics.

**What remains uncertain:** the chosen states are statistical approximations. They depend on feature choice, Gaussian assumptions, the selected state-count range, initialization and the fitted historical period. Rolling features create dependence that can itself increase persistence. Posterior certainty is not independently calibrated confidence in a true economic state.

**Next extensions:** Student-t emissions, full-covariance sensitivity, walk-forward parameter refitting, missing-emission handling, different volatility horizons, and validation on another market. A regime-based investment strategy would be a separate project requiring implementable execution timing, an investable benchmark, turnover and transaction costs.

**References:**
- [NIFTY Indices historical data](https://www.niftyindices.com/reports/historical-data)
- [NSE historical India VIX](https://www.nseindia.com/reports-indices-historical-vix)
- [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)
- [hmmlearn API and decoding definitions](https://hmmlearn.readthedocs.io/en/stable/api.html)
""")

nb=nbf.v4.new_notebook(cells=cells)
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.12"}}
nbf.write(nb,ROOT/"notebooks/market_regime_detection.ipynb")
print(f"Created market regime notebook with {len(cells)} cells")
