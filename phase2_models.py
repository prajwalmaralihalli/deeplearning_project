"""Phase 2: Walmart Demand Forecasting Benchmark.

This script benchmarks three forecasting paradigms on Store 1 using a strict
chronological train/test split and an exogenous-variable setup suitable for
Phase 3 model evaluation.

The workflow is intentionally modular and designed for the "antigravity"
Python environment, where the following packages are expected to be installed:
- pandas, numpy, matplotlib, scikit-learn
- statsmodels
- prophet
- torch

Key assumptions:
1. The dataset is either `walmart_engineered.csv` or `Walmart.csv` in the same
   directory as this script.
2. The training objective is Store 1 weekly sales.
3. The exogenous variables used across models are:
   `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`.
4. Models are evaluated on the same 20% test period to ensure comparable
   performance under the same time horizon.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from prophet import Prophet
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Prophet is required. Please activate the antigravity environment and install it."
    ) from exc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Weekly_Sales"
EXOG_COLS = ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]


def locate_dataset() -> Path:
    """Find the Walmart dataset in the project directory."""
    for filename in ("walmart_engineered.csv", "Walmart.csv"):
        path = ROOT / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate walmart_engineered.csv or Walmart.csv in the project directory."
    )


def load_store1_dataset() -> pd.DataFrame:
    """Load the Walmart sales data, keep Store 1, and sort it chronologically."""
    path = locate_dataset()
    df = pd.read_csv(path, parse_dates=["Date"])

    if "Store" in df.columns:
        df = df[df["Store"] == 1].copy()

    df = df.sort_values("Date").reset_index(drop=True)

    if "Date" not in df.columns:
        raise ValueError("The dataset must contain a Date column.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in the dataset.")

    for col in EXOG_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required exogenous variable: {col}")

    return df


def chronological_split(df: pd.DataFrame, split_fraction: float = 0.8):
    """Create a strict chronological 80/20 train-test split with no future leakage."""
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("split_fraction must be between 0 and 1.")

    split_index = int(len(df) * split_fraction)
    train = df.iloc[:split_index].copy().reset_index(drop=True)
    test = df.iloc[split_index:].copy().reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError("The split produced an empty train or test set.")

    train_end = train["Date"].max()
    test_start = test["Date"].min()
    no_temporal_overlap = train_end < test_start

    print("=" * 80)
    print("1. CHRONOLOGICAL DATA PREPARATION & SPLIT")
    print("=" * 80)
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")
    print(f"Train date range: {train['Date'].min().date()} to {train_end.date()}")
    print(f"Test date range:  {test_start.date()} to {test['Date'].max().date()}")
    print(f"No temporal overlap: {no_temporal_overlap}")
    print(f"Strict chronological ordering check: {no_temporal_overlap}")

    assert no_temporal_overlap, "Train and test periods overlap in time."

    return train, test


def safe_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE for forecasting predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100.0
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def train_sarimax(train: pd.DataFrame, test: pd.DataFrame):
    """Train a SARIMAX model with exogenous regressors and generate test forecasts."""
    print("\n" + "=" * 80)
    print("2. MODEL 1: SARIMAX")
    print("=" * 80)

    exog_cols = EXOG_COLS
    model = SARIMAX(
        train[TARGET_COL],
        exog=train[exog_cols],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    exog_forecast = test[exog_cols]
    forecast = fitted.get_forecast(steps=len(test), exog=exog_forecast)
    predicted = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    result = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "Actual": test[TARGET_COL].values,
            "SARIMAX_Prediction": predicted.values,
            "SARIMAX_Lower_95": conf_int.iloc[:, 0].values,
            "SARIMAX_Upper_95": conf_int.iloc[:, 1].values,
        }
    )

    print(f"SARIMAX fit loglikelihood: {fitted.llf:.3f}")
    print(f"SARIMAX forecast sample count: {len(predicted)}")
    return fitted, result


def train_prophet(train: pd.DataFrame, test: pd.DataFrame):
    """Train a Prophet model with external regressors and produce a test forecast."""
    print("\n" + "=" * 80)
    print("3. MODEL 2: PROPHET")
    print("=" * 80)

    train_prophet = train[["Date", TARGET_COL] + EXOG_COLS].rename(
        columns={"Date": "ds", TARGET_COL: "y"}
    )
    test_prophet = test[["Date"] + EXOG_COLS].rename(columns={"Date": "ds"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    for column in ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI"]:
        model.add_regressor(column)

    model.fit(train_prophet)

    future = test_prophet.copy()
    forecast = model.predict(future)

    result = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "Actual": test[TARGET_COL].values,
            "Prophet_Prediction": forecast["yhat"].values,
            "Prophet_Lower_95": forecast["yhat_lower"].values,
            "Prophet_Upper_95": forecast["yhat_upper"].values,
        }
    )

    print(f"Prophet forecast sample count: {len(forecast)}")
    return model, result


class SequenceDataset(Dataset):
    """Dataset for a sliding-window sequence-to-one forecasting task."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 12):
        self.sequence_length = sequence_length
        self.X = X
        self.y = y

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length.")

    def __len__(self):
        return max(0, len(self.X) - self.sequence_length)

    def __getitem__(self, idx: int):
        seq_start = idx
        seq_end = idx + self.sequence_length
        x_seq = self.X[seq_start:seq_end]
        y_target = self.y[seq_end]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


class LSTMRegressor(nn.Module):
    """A modular LSTM regressor for weekly sales forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def train_lstm(train: pd.DataFrame, test: pd.DataFrame, sequence_length: int = 12):
    """Train a PyTorch LSTM model using normalized lagged sales and exogenous variables."""
    print("\n" + "=" * 80)
    print("4. MODEL 3: PYTORCH LSTM")
    print("=" * 80)

    feature_cols = [TARGET_COL] + EXOG_COLS
    train_features = train[feature_cols].copy()
    test_features = test[feature_cols].copy()

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_features)
    target_scaler.fit(train[[TARGET_COL]])

    train_scaled_features = feature_scaler.transform(train_features)
    test_scaled_features = feature_scaler.transform(test_features)

    train_scaled_target = target_scaler.transform(train[[TARGET_COL]]).ravel()
    test_scaled_target = target_scaler.transform(test[[TARGET_COL]]).ravel()

    # Construct sequence dataset using the latest 12 weeks of sales + exogenous variables.
    X_train = train_scaled_features.astype(np.float32)
    y_train = train_scaled_target.astype(np.float32)

    sequence_dataset = SequenceDataset(X_train, y_train, sequence_length=sequence_length)
    train_loader = DataLoader(sequence_dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]
    model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:02d}/50 | Loss: {epoch_loss / max(1, len(train_loader)):.6f}")

    # Recursive multi-step forecasting over test, using actual observed exogenous inputs
    # and previously observed sales values to build the sliding history window.
    history = X_train[-sequence_length:].copy()
    predictions = []
    actual_history = train[TARGET_COL].tolist()[-sequence_length:]
    for idx in range(len(test)):
        seq = history[-sequence_length:]
        seq_tensor = torch.tensor(seq.reshape(1, sequence_length, -1), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(seq_tensor).item()

        pred_value = target_scaler.inverse_transform(np.array([[pred_scaled]])).item()
        predictions.append(pred_value)

        current_row = test_scaled_features[idx]
        history = np.vstack([history, current_row])
        actual_history.append(test[TARGET_COL].iloc[idx])

    result = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "Actual": test[TARGET_COL].values,
            "LSTM_Prediction": predictions,
        }
    )

    print(f"LSTM test forecast count: {len(result)}")
    return model, result


def build_leaderboard(results: dict):
    """Assemble a model leaderboard with RMSE, MAE, and MAPE."""
    leaderboard = []
    for model_name, model_result in results.items():
        actual_col = "Actual" if "Actual" in model_result.columns else TARGET_COL
        pred_col = f"{model_name}_Prediction"
        if pred_col not in model_result.columns:
            pred_col = next(col for col in model_result.columns if col.endswith("Prediction"))

        metrics = safe_metrics(model_result[actual_col].values, model_result[pred_col].values)
        leaderboard.append(
            {
                "Model": model_name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("RMSE").reset_index(drop=True)
    print("\n" + "=" * 80)
    print("5. EVALUATION METRICS & COMPARATIVE VISUALIZATION")
    print("=" * 80)
    print(leaderboard_df.to_string(index=False))
    return leaderboard_df


def plot_comparison(actual: pd.Series, sarimax: pd.Series, prophet: pd.Series, lstm: pd.Series):
    """Plot actual sales vs all three model predictions on the test horizon."""
    plt.figure(figsize=(14, 8), dpi=200)
    plt.plot(actual.index, actual.values, label="Actual Historical Sales", color="black", linewidth=2)
    plt.plot(sarimax.index, sarimax.values, label="SARIMAX", color="tab:blue", linewidth=2)
    plt.plot(prophet.index, prophet.values, label="Prophet", color="tab:orange", linewidth=2)
    plt.plot(lstm.index, lstm.values, label="LSTM", color="tab:green", linewidth=2)

    plt.title("Walmart Weekly Sales Forecasting Benchmark (Store 1)")
    plt.xlabel("Test Week Index")
    plt.ylabel("Weekly Sales")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run the end-to-end Phase 2 benchmarking workflow."""
    df = load_store1_dataset()
    train, test = chronological_split(df)

    sarimax_model, sarimax_result = train_sarimax(train, test)
    prophet_model, prophet_result = train_prophet(train, test)
    lstm_model, lstm_result = train_lstm(train, test, sequence_length=12)

    merged = test[["Date", TARGET_COL]].copy()
    merged = merged.merge(sarimax_result[["Date", "SARIMAX_Prediction"]], on="Date", how="left")
    merged = merged.merge(prophet_result[["Date", "Prophet_Prediction"]], on="Date", how="left")
    merged = merged.merge(lstm_result[["Date", "LSTM_Prediction"]], on="Date", how="left")

    results = {
        "SARIMAX": merged[["Date", TARGET_COL, "SARIMAX_Prediction"]].copy(),
        "Prophet": merged[["Date", TARGET_COL, "Prophet_Prediction"]].copy(),
        "LSTM": merged[["Date", TARGET_COL, "LSTM_Prediction"]].copy(),
    }

    leaderboard = build_leaderboard(results)

    actual = merged[TARGET_COL].reset_index(drop=True)
    sarimax_predictions = merged["SARIMAX_Prediction"].reset_index(drop=True)
    prophet_predictions = merged["Prophet_Prediction"].reset_index(drop=True)
    lstm_predictions = merged["LSTM_Prediction"].reset_index(drop=True)

    plot_comparison(actual, sarimax_predictions, prophet_predictions, lstm_predictions)

    merged.to_csv(DATA_DIR / "model_forecasts.csv", index=False)
    print(f"\nSaved model predictions to: {DATA_DIR / 'model_forecasts.csv'}")
    print(f"Saved benchmark plot to: {PLOTS_DIR / 'model_comparison_benchmark.png'}")


if __name__ == "__main__":
    main()
