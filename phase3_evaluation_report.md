# Phase 3 Evaluation Report

## 1) Metrics Summary

| Model   |     RMSE |      MAE |   MAPE |   WAPE |   MDA |
|:--------|---------:|---------:|-------:|-------:|------:|
| Prophet |  89364.4 |  64160.6 |   4.07 |   4.12 |  0.44 |
| LSTM    | 106087   |  91002.6 |   5.97 |   5.85 |  0.44 |
| SARIMAX | 518491   | 480854   |  31.02 |  30.9  |  0.89 |

## 2) Residual Diagnostics
The 3x3 diagnostic grid was saved to `plots/phase3_residual_diagnostics.png`.

- Residuals are defined as `e_t = y_t - \hat{y}_t`.
- A stable model should show low-bias residuals with no obvious upward or downward drift.
- Residual histograms help assess whether the model errors are roughly centered and near-normal.

## 3) Model Comparison & Trade-Off Matrix

| Architecture   | Predictive Accuracy                                      | Interpretability                                            | Data Requirements                                                | Compute Cost                                                  | Maintenance                                                                            |
|:---------------|:---------------------------------------------------------|:------------------------------------------------------------|:-----------------------------------------------------------------|:--------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| Prophet        | Best on short weekly data (MAPE ~4.07%, RMSE ~89k)       | High: decomposable trend + seasonality + holiday components | Works well with limited history; strong on short horizons        | Low to moderate; fast training/inference                      | Low: transparent components and easier debugging                                       |
| PyTorch LSTM   | Competitive but behind Prophet (MAPE ~5.97%, RMSE ~106k) | Low: black-box sequence model                               | Needs more data; performs weaker on short histories              | High: GPU/CPU training and hyperparameter tuning              | Moderate to high: harder to debug drift and failure modes                              |
| SARIMAX        | Poor on this configuration (MAPE ~31.02%, RMSE ~518k)    | Moderate to high: explicit parametric structure             | Fragile when seasonal order is too large relative to sample size | Low to moderate; inference is cheap but training is sensitive | Moderate: statistical diagnostics are available but unstable under weak identification |

## 4) Technical Critique

## Technical Critique

### 1) Why Prophet outperformed (MAPE 4.07%):
Prophet scored the best because its GAM structure is well suited to short weekly series with a moderate amount of trend and recurring yearly structure. The model decomposes the target into a local linear trend component, a Fourier-series yearly seasonal term, and explicit external regressors such as holidays and exogenous drivers. In this dataset, holiday intensity and macro conditions matter, and Prophet captures those effects without requiring a high-dimensional autoregressive state vector. Because the history is only about 115 training weeks, the model benefits from strong regularization and a low effective parameter count. The yearly signal is still captured through a small Fourier basis instead of thousands of seasonal dummy terms, which avoids overfitting while preserving the recurring annual pattern.

### 2) Why SARIMAX failed so badly (MAPE 31.02%):
The root cause is not that SARIMAX is intrinsically bad — it is that this specification is statistically underpowered for the available data. The seasonal component uses s=52 with a seasonal order of (1, 1, 1, 52), meaning the state-space model is trying to learn a large seasonal structure with only ~115 training points. That is roughly 2 seasonal cycles, which is barely enough to identify the 52 seasonal state components, much less their interactions with trend and exogenous variables. In practice, this leads to parameter explosion, near-collinearity among seasonal states, weak identification, and numerical instability. The model then drifts badly on the test window because the estimated seasonal pattern is effectively a poor extrapolation from too few cycles. In short, the model is trying to estimate a seasonal system larger than the information content of the training sample.

### 3) Why LSTM lagged behind Prophet (MAPE 5.97%):
The LSTM is a strong general-purpose sequence learner, but it suffers from a classic time-series data-starvation problem. With only a few hundred weekly observations, the model does not see enough distinct temporal patterns to learn a stable long-horizon mapping from lagged sales and exogenous variables to weekly demand. Deep recurrent models are especially sample-hungry when the task is not dominated by long-range sequence structure but by sparse causal signals like seasonality, macroeconomic conditions, and holiday effects. Under these conditions, a regularized additive model like Prophet often wins because it encodes domain knowledge and imposes stronger structural priors. The LSTM can still learn useful patterns, but it is more prone to variance and weaker generalization in short-horizon, low-volume time series settings.

### 4) Why the error patterns matter:
The residual diagnostics reveal whether a model is merely inaccurate or systematically wrong. A residual mean near zero and a roughly symmetric distribution suggest unbiased forecasting, while persistent residual drift or heteroscedasticity indicates structural misspecification. Prophet’s residuals are the least noisy and most centered around zero, whereas SARIMAX shows large drift and volatility after the seasonal structure becomes unstable. LSTM residuals are smoother than SARIMAX but still more biased than Prophet, reflecting the model’s limited training sample and relatively weak inductive bias for this problem.

### 5) Production interpretation:
For a single-store, weekly demand forecast with ~115 observations, Prophet is the most operationally reliable choice. It is faster to debug, easier to explain to stakeholders, and more robust under short-series conditions. An LSTM becomes attractive when the pipeline scales to many products, stores, or related series, where information-sharing across series and larger data volumes justify a more flexible deep-learning architecture. SARIMAX should be reserved for cases with longer histories and more stable seasonal patterns, or when the analyst wants a transparent parametric benchmark rather than a production forecasting engine.

## 5) Interview Summary

## Executive Summary
- Recommendation: Deploy Prophet as the production baseline for this Store 1 weekly forecasting task because it has the lowest MAPE and the most stable residual behavior on a short-history dataset.
- SARIMAX root cause: the seasonal specification with s=52 is statistically underidentified on ~115 weeks of data, creating parameter explosion, collinearity, and unstable out-of-sample drift.
- LSTM threshold: switch from Prophet to an LSTM or Transformer architecture when the business scales to 10,000+ SKUs or many correlated series, where cross-series learning and larger data volumes justify deep learning.
- Operational insight: short-series forecasting favors regularized, interpretable additive models; deep learners become more attractive when data volume, cross-series sharing, and nonlinear interactions dominate the forecasting problem.

