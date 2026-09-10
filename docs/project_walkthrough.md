# Understanding the market regime project

## What is a regime?

A regime is a useful statistical description of an environment, such as a period of low volatility and positive momentum or high volatility and deep drawdowns. There is no supplied column containing the objectively true daily regime. We infer states from observed features and judge whether their distributions and transitions are useful descriptions.

Calm and bullish are different dimensions and can occur together. The rule baseline therefore separates volatility from direction rather than pretending those words are automatically exclusive classes.

## What does one observation contain?

At the end of each valid observation date, the model uses NIFTY's current log return, trailing 20-observation volatility, trailing momentum and India VIX. Logs are used for the positive volatility measures; all four inputs are then standardized using only the allowed historical fitting period.

The BANK series contributes a correlation diagnostic. It does not become a fifth HMM feature. NIFTY and BANK are price index levels, not returns until transformed; India VIX is implied volatility, not a historical return standard deviation.

## Why not shuffle the dates?

Shuffling would mix later market conditions into earlier model fitting and obscure how the model behaves when conditions change. The project fits initially on 2011–2018, selects configurations on 2019–2021, refits through 2021, and freezes parameters before evaluating 2022–2025.

A label attached to a fitting-period observation is retrospective because the fitted parameters saw the entire fitting period. Filtering solves future-emission leakage in state inference; it does not make in-sample parameter estimates magically historical or out of sample.

## Why compare rules, K-means and an HMM?

Rules supply an understandable risk/trend benchmark. K-means groups similar feature observations using distances. An HMM represents both state-specific distributions and transitions. It can retain a state when today's features are somewhat ambiguous but the previous filtered state was persistent.

Persistence is partly learned from overlapping rolling features. Fewer switches are therefore a property to inspect, not a universal definition of a better detector. A very sticky model could miss short-lived changes.

## What exactly is filtering?

Start with yesterday's probability of each state. Multiply by the transition matrix to predict today's state probabilities. Evaluate how likely today's feature vector is under each state. Multiply the predicted state probabilities by those likelihoods and normalize. Only observations through today enter that calculation.

Smoothing instead incorporates later observations when interpreting an earlier state. It is useful for retrospective descriptions but should not be presented as a signal available at the earlier date. The notebook explicitly compares the two and uses filtering for held-out assignments.

## What does likelihood measure?

The predictive density score describes how well the statistical model assigns probability density to the observed standardized feature vector. It is not the probability that the market will rise, nor a trading return. Scores can be compared here because models use the same transformation and feature dimension.

The independent Gaussian mixture is a density baseline without transitions. The HMM also has transition parameters, so this is not an equal-complexity contest. Held-out log scores are nevertheless a useful measure of predictive density performance on this particular feature representation.

## Why use several initialization seeds?

HMM fitting uses an iterative procedure that can end at different local optima. Five seeds make this visible. Selection uses fitting-period likelihood, not test scores. ARI compares the held-out state partitions without requiring the component numbers to have the same meaning.

This is initialization sensitivity, not a bootstrap confidence interval. Different features, covariance structures or emission families could change the result further.

## Why report future volatility separately?

State definitions use only current and prior features. The following 20-observation volatility is computed afterwards to examine what tended to happen next. It does not enter model fitting or state selection. Some currently high-volatility episodes normalize quickly, which is why R4's future volatility need not exceed every other state's.

Future windows overlap, so observations are not independent. A rare-state mean from 36 held-out observations is less secure than a mean based on hundreds. The project reports the sample counts and does not turn these averages into a causal investment recommendation.

## What should she be able to explain in an interview?

Explain the difference between price levels, returns, realized volatility and India VIX; the difference between an HMM and K-means; the chronological split; filtering versus smoothing; why five initializations were necessary; and why smooth state assignments are not the same as verified accuracy.

Be able to show the forward recursion and explain why appending future observations cannot change a previous filtered posterior when parameters are fixed. Also explain why that guarantee does not apply to refitting parameters using later data.

Use the measured density, switching and robustness results with their limitations. Do not claim stock-price forecasting accuracy, annual investment returns, or reliable early crisis prediction from this project.

## Potential extensions

Compare Student-t or other heavy-tailed emissions, repeat with different volatility windows, evaluate walk-forward parameter updates, handle missing feature emissions explicitly, and test another market. A trading strategy would require separate execution timing, an investable benchmark, transaction costs and robust economic evaluation.
