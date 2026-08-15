# Walmart Demand Forecasting Benchmark

A professional time series forecasting project focused on weekly Walmart sales forecasting for Store 1, with a strict chronological evaluation strategy and comparison of three modeling paradigms:

- Prophet (additive decomposition model)
- PyTorch LSTM (deep learning sequence model)
- SARIMAX (classical statistical model)

## Project Objective

This project builds a rigorous benchmark for short-horizon weekly sales forecasting using real Walmart sales data with external economic and holiday features. The work follows a clear Phase 2 and Phase 3 evaluation pipeline:

1. Chronological train/test split with no future leakage.
2. Model training and benchmarking across three architectures.
3. Residual diagnostics, error analysis, and technical critique.

## Dataset

The project uses the engineered Walmart dataset with weekly sales and exogenous variables such as:

- Holiday_Flag
- Temperature
- Fuel_Price
- CPI
- Unemployment

## Core Results

The benchmark on the Store 1 test horizon produced:

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Prophet | 89,364.40 | 64,160.55 | 4.07% |
| LSTM | 106,086.94 | 91,002.57 | 5.97% |
| SARIMAX | 518,490.72 | 480,853.96 | 31.02% |

## Repository Contents

- `phase2_models.py` — end-to-end training and benchmarking of all three models.
- `phase3_evaluation.py` — residual diagnostics, WAPE, directional accuracy, and technical critique.
- `data/model_forecasts.csv` — stored actual and predicted test values.
- `plots/model_comparison_benchmark.png` — multi-model benchmark visualization.
- `plots/phase3_residual_diagnostics.png` — residual diagnostics dashboard.
- `phase3_evaluation_report.md` — detailed markdown report.

## Execution

Run the benchmark script:

```bash
python phase2_models.py
```

Run the evaluation pipeline:

```bash
python phase3_evaluation.py
```

## Environment

This project was developed in a Python environment with:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- prophet
- torch

## Author

Prajwal V Maralihalli

## Status

Production-quality forecasting benchmark for Store 1 weekly sales with rigorous model comparison and evaluation.
