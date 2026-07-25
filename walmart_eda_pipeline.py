"""Comprehensive EDA and feature engineering pipeline for Walmart weekly sales.

This script loads the Walmart.csv dataset, performs a structured data quality audit,
visualizes key distributions and seasonal patterns, segments stores by performance,
explores external economic indicators, and engineers a leakage-safe feature set for
forecasting models.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def resolve_dataset_path(file_path: str = "Walmart.csv") -> Path:
    """Resolve the dataset path from common workspace and filesystem locations."""
    candidates = []

    if file_path:
        candidates.append(Path(file_path))
        candidates.append(Path.cwd() / file_path)
        candidates.append(Path(__file__).resolve().parent / file_path)

    for base in [Path.cwd(), Path(__file__).resolve().parent, Path("E:/"), Path("D:/")]:
        candidates.append(base / "Walmart.csv")
        candidates.append(base / "walmart.csv")

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved

    # Search recursively for Walmart.csv in likely locations
    for base in [Path.cwd(), Path(__file__).resolve().parent, Path("E:/"), Path("D:/")]:
        if base.exists():
            for match in base.rglob("Walmart.csv"):
                if match.is_file():
                    return match

    raise FileNotFoundError(
        f"Could not find Walmart.csv. Place the file in the workspace or provide its full path."
    )


def load_and_clean_data(file_path: str = "Walmart.csv") -> pd.DataFrame:
    """Load the dataset, parse dates, and audit basic data quality."""
    data_path = resolve_dataset_path(file_path)

    df = pd.read_csv(data_path, parse_dates=["Date"], date_format="%d-%m-%Y")

    # Basic consistency checks and initial cleaning
    original_rows = len(df)
    df = df.copy()
    quality_summary = {
        "row_count": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "date_range": {
            "min_date": df["Date"].min().strftime("%Y-%m-%d"),
            "max_date": df["Date"].max().strftime("%Y-%m-%d"),
        },
        "unique_stores": int(df["Store"].nunique()),
    }

    # Validate chronological order per store before sorting
    order_status = df.groupby("Store")["Date"].apply(lambda s: s.is_monotonic_increasing)
    quality_summary["chronological_order_by_store"] = {
        "all_in_order": bool(order_status.all()),
        "out_of_order_stores": [int(store) for store in order_status[~order_status].index.tolist()],
    }

    # Sort by store and date for time-series stability
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    print("=" * 70)
    print("1. DATA INGESTION & QUALITY AUDIT")
    print("=" * 70)
    print(f"Rows loaded: {original_rows}")
    print(f"Rows retained after cleaning: {len(df)}")
    print(f"Missing values: {df.isna().sum().to_dict()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Data types: {quality_summary['dtypes']}")
    print(f"Date range: {quality_summary['date_range']['min_date']} -> {quality_summary['date_range']['max_date']}")
    print(f"Unique stores: {quality_summary['unique_stores']}")
    print(
        "Chronological order per store: "
        f"{'All stores are in order' if quality_summary['chronological_order_by_store']['all_in_order'] else 'Some stores are not in order'}"
    )
    if quality_summary["chronological_order_by_store"]["out_of_order_stores"]:
        print(
            "Out-of-order stores: "
            f"{quality_summary['chronological_order_by_store']['out_of_order_stores']}"
        )

    return df


def profile_distributions(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Calculate summary statistics and visualize weekly sales distribution."""
    numeric_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    summary_stats = pd.DataFrame(
        {
            "Mean": df[numeric_cols].mean(),
            "Median": df[numeric_cols].median(),
            "Std": df[numeric_cols].std(),
            "IQR": df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
            "Min": df[numeric_cols].min(),
            "Max": df[numeric_cols].max(),
            "Skewness": df[numeric_cols].skew(),
        }
    ).T

    print("\n" + "=" * 70)
    print("2. SUMMARY STATISTICS & DISTRIBUTION PROFILING")
    print("=" * 70)
    print(summary_stats)

    sales_skew = float(summary_stats.loc["Skewness", "Weekly_Sales"])
    if sales_skew > 1.0:
        interpretation = (
            "Weekly_Sales is strongly right-skewed. A log1p transform will likely improve "
            "stability for linear and neural models."
        )
    elif sales_skew > 0.5:
        interpretation = (
            "Weekly_Sales is moderately skewed. A log1p transform may still help, especially "
            "for models sensitive to variance."
        )
    else:
        interpretation = "Weekly_Sales is fairly balanced; a log transform is optional."
    print(f"Sales skewness: {sales_skew:.3f}")
    print(interpretation)

    # Dual-panel figure: histogram + box plot by store
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Weekly_Sales"], kde=True, ax=axes[0], color="steelblue", bins=40)
    axes[0].set_title("Weekly Sales Distribution")
    axes[0].set_xlabel("Weekly Sales")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x="Store", y="Weekly_Sales", data=df, ax=axes[1], palette="viridis")
    axes[1].set_title("Weekly Sales Outliers by Store")
    axes[1].set_xlabel("Store")
    axes[1].set_ylabel("Weekly Sales")
    plt.tight_layout()
    distribution_path = output_dir / "eda_sales_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary_stats


def analyze_holidays_and_seasonality(df: pd.DataFrame, output_dir: Path) -> dict:
    """Analyze overall weekly sales trend, holiday effects, and monthly seasonality."""
    overall_sales = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    overall_sales = overall_sales.sort_values("Date")

    holiday_dates = set(df.loc[df["Holiday_Flag"] == 1, "Date"].tolist())
    holiday_sales = overall_sales[overall_sales["Date"].isin(holiday_dates)]

    # Holiday lift calculation
    holiday_mask = df["Holiday_Flag"] == 1
    non_holiday_mask = df["Holiday_Flag"] != 1
    holiday_avg = df.loc[holiday_mask, "Weekly_Sales"].mean()
    non_holiday_avg = df.loc[non_holiday_mask, "Weekly_Sales"].mean()
    holiday_lift_pct = ((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100

    print("\n" + "=" * 70)
    print("3. MACRO TIME SERIES & HOLIDAY SEASONALITY ANALYSIS")
    print("=" * 70)
    print(f"Holiday lift: {holiday_lift_pct:.2f}%")

    # Overall trend plot with holiday markers
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(overall_sales["Date"], overall_sales["Weekly_Sales"], color="navy", linewidth=1.5)
    ax.scatter(
        holiday_sales["Date"],
        holiday_sales["Weekly_Sales"],
        color="red",
        marker="*",
        s=140,
        label="Holiday Week",
        zorder=5,
    )
    ax.set_title("Company-wide Weekly Sales Trend with Holiday Weeks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Weekly Sales")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    trend_path = output_dir / "eda_trends.png"
    plt.savefig(trend_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Monthly average sales by year to highlight Q4 spikes
    monthly = df.copy()
    monthly["Year"] = monthly["Date"].dt.year
    monthly["Month"] = monthly["Date"].dt.month
    monthly_avg = monthly.groupby(["Year", "Month"])["Weekly_Sales"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for year in sorted(monthly_avg["Year"].unique()):
        subset = monthly_avg[monthly_avg["Year"] == year]
        ax.plot(subset["Month"], subset["Weekly_Sales"], marker="o", label=str(year), linewidth=1.5)
    ax.set_title("Monthly Average Sales by Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Weekly Sales")
    ax.set_xticks(range(1, 13))
    ax.legend(title="Year")
    plt.tight_layout()
    seasonality_path = output_dir / "eda_monthly_seasonality.png"
    plt.savefig(seasonality_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "holiday_lift_pct": holiday_lift_pct,
        "holiday_avg": holiday_avg,
        "non_holiday_avg": non_holiday_avg,
    }


def analyze_store_tiers(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Segment stores by average weekly sales and volatility."""
    store_stats = (
        df.groupby("Store")
        .agg(
            avg_weekly_sales=("Weekly_Sales", "mean"),
            volatility=("Weekly_Sales", lambda x: x.std() / x.mean()),
        )
        .reset_index()
    )

    # Rank-based tiering: top 20%, middle 60%, bottom 20%
    n = len(store_stats)
    top_n = max(1, int(np.floor(0.2 * n)))
    bottom_n = max(1, int(np.floor(0.2 * n)))
    store_stats["tier"] = "Medium Performers"
    store_stats["rank_desc"] = store_stats["avg_weekly_sales"].rank(ascending=False, method="first")
    store_stats["rank_asc"] = store_stats["avg_weekly_sales"].rank(ascending=True, method="first")
    store_stats.loc[store_stats["rank_desc"] <= top_n, "tier"] = "High Performers"
    store_stats.loc[store_stats["rank_asc"] <= bottom_n, "tier"] = "Low Performers"

    print("\n" + "=" * 70)
    print("4. STORE-LEVEL TIER SEGMENTATION")
    print("=" * 70)
    print(store_stats.groupby("tier").size().to_string())

    top5 = store_stats.sort_values("avg_weekly_sales", ascending=False).head(5)
    bottom5 = store_stats.sort_values("avg_weekly_sales", ascending=True).head(5)
    comparison = pd.concat([top5, bottom5], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=comparison, x="Store", y="avg_weekly_sales", hue="tier", ax=ax, palette="viridis")
    ax.set_title("Top 5 vs Bottom 5 Stores by Average Weekly Sales")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Weekly Sales")
    plt.tight_layout()
    tiers_path = output_dir / "eda_store_tiers.png"
    plt.savefig(tiers_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return store_stats


def analyze_correlations(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrices and visualize weekly sales against external indicators."""
    indicator_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    corr_pearson = df[["Weekly_Sales", *indicator_cols]].corr(method="pearson")
    corr_spearman = df[["Weekly_Sales", *indicator_cols]].corr(method="spearman")

    print("\n" + "=" * 70)
    print("5. EXTERNAL INDICATOR & CORRELATION ANALYSIS")
    print("=" * 70)
    print("Pearson correlations:")
    print(corr_pearson)
    print("\nSpearman correlations:")
    print(corr_spearman)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, indicator_cols):
        sns.regplot(
            data=df,
            x=col,
            y="Weekly_Sales",
            ax=ax,
            scatter_kws={"alpha": 0.2, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 2},
        )
        ax.set_title(f"Weekly Sales vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Weekly Sales")
    fig.delaxes(axes[-1])
    plt.tight_layout()
    correlations_path = output_dir / "eda_correlations.png"
    plt.savefig(correlations_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return corr_pearson, corr_spearman


def engineer_features(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create leakage-safe calendar and lag-based features for time-series modeling."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)

    # Lag features by store (avoid leakage by shifting only past observations)
    df["Weekly_Sales_Lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["Weekly_Sales_Lag_2"] = df.groupby("Store")["Weekly_Sales"].shift(2)
    df["Weekly_Sales_Lag_4"] = df.groupby("Store")["Weekly_Sales"].shift(4)
    df["Weekly_Sales_Lag_52"] = df.groupby("Store")["Weekly_Sales"].shift(52)

    # Rolling features from shifted values to avoid leakage
    df["Sales_Rolling_Mean_4"] = (
        df.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    )
    df["Sales_Rolling_Std_4"] = (
        df.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1).rolling(4, min_periods=4).std())
    )
    df["Sales_Rolling_Mean_12"] = (
        df.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1).rolling(12, min_periods=12).mean())
    )

    # Drop rows with NaN caused by lag/rolling windows
    feature_cols = [
        "Weekly_Sales_Lag_1",
        "Weekly_Sales_Lag_2",
        "Weekly_Sales_Lag_4",
        "Weekly_Sales_Lag_52",
        "Sales_Rolling_Mean_4",
        "Sales_Rolling_Std_4",
        "Sales_Rolling_Mean_12",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("6. TIME SERIES FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Engineered features added. Final rows: {len(df)}")

    engineered_path = output_dir.parent / "walmart_engineered.csv"
    df.to_csv(engineered_path, index=False)
    print(f"Saved engineered dataset to: {engineered_path}")

    return df


def run_eda(file_path: str = "Walmart.csv") -> None:
    """Run the complete EDA and feature engineering workflow."""
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = load_and_clean_data(file_path=file_path)
    profile_distributions(df, output_dir)
    analyze_holidays_and_seasonality(df, output_dir)
    analyze_store_tiers(df, output_dir)
    analyze_correlations(df, output_dir)
    engineer_features(df, output_dir)

    print("\nEDA completed successfully.")
    print(f"All plots saved in: {output_dir}")
    print(f"Engineered CSV saved in: {project_dir / 'walmart_engineered.csv'}")


if __name__ == "__main__":
    run_eda()
