# Walmart Demand Forecasting & Time Series Modeling Study Guide

## 1. Project Overview & Business Framing

### Executive Summary

This project addresses a classic retail forecasting problem: predicting weekly Walmart sales at store level using historical sales and external macro variables. The analysis is grounded in a real-world retail dataset spanning 45 stores over a 2010–2012 timeframe, with weekly sales observations and features such as holiday flags, temperature, fuel price, CPI, and unemployment.

The business objective is to forecast demand accurately enough to reduce inventory holding costs, optimize safety stock, improve working capital efficiency, reduce stockouts, and support labor scheduling decisions. The model selection process is built around a realistic operational constraint: we must work with a relatively short time series and avoid future leakage.

### Business Value

Retail demand forecasting is one of the highest-leverage ML applications because it sits directly on the financial operating model:

- Inventory holding cost reduction: better forecasting reduces overstock and markdown risk.
- Safety stock optimization: more accurate demand estimates translate into lower emergency replenishment frequency and better service levels.
- Working capital efficiency: inventory dollars are tied up in stock; forecasting helps reduce cash trapped in slow-moving products.
- Stockout prevention: under-forecasting causes lost sales and customer dissatisfaction.
- Labor scheduling: demand estimates improve staffing decisions for stores and warehouses.

In operational retail, small improvements in forecast accuracy can create outsized gains in revenue and margin because forecasting sits upstream of procurement, distribution, and store operations.

---

## 2. Tech Stack Selection & Justification ("Why did you use X?")

| Tool | Why It Was Used | Why It Was the Right Choice |
| :--- | :--- | :--- |
| Python & VS Code | Reproducible experimentation, modular script development, and quick iteration | Python is the standard language for time series, statistics, and ML; VS Code makes local debugging and scripts easy to maintain |
| `antigravity` virtual environment | Environment isolation and dependency control | Prevents package conflicts and makes experiments reproducible across runs |
| Pandas & NumPy | Temporal indexing, lag creation, rolling aggregations, vectorized math | Time series work depends on efficient and correct handling of ordered data and historical features |
| Matplotlib & Seaborn | Visual diagnostics: trends, holiday lifts, residual distributions, EDA | Business stakeholders need intuitive plots; analysts need distribution and residual checks |
| Statsmodels / SARIMAX | Classical state-space modeling with exogenous variables | Strong default benchmark for structured time series and interpretable dynamics |
| Meta Prophet | Additive decomposition model with trend + seasonality + holiday regressors | Particularly strong on short, weekly retail series with recurring annual patterns |
| PyTorch / LSTM | Non-linear sequence learning | Useful where there are latent interactions and complex temporal dependencies beyond linear autoregression |
| Scikit-Learn | Scaling and evaluation metrics (`MinMaxScaler`, RMSE, MAE, MAPE, WAPE) | Standardized preprocessing and rigorous model comparison |

### Technical Rationale by Stack Component

#### Python & VS Code (`antigravity`)
Python is ideal for the project because the workflow combines statistical scripts, optimization, plotting, and deep learning in one ecosystem. VS Code offers modular file organization, debugging, notebook integration, and a clean path for reproducible experiments. The `antigravity` environment isolates versions such as `pandas`, `numpy`, `statsmodels`, `prophet`, `torch`, and `scikit-learn` so that the pipeline runs consistently.

#### Pandas & NumPy
Time series forecasting is fundamentally about order, lag structure, and windowed context. Pandas enables easy datetime parsing, chronological ordering, filtering by store, and generation of lagged features. NumPy supports vectorized operations for rolling statistics, residuals, and numerical optimization.

#### Matplotlib & Seaborn
Visual EDA is essential because poor models can still look plausible on aggregate plots. Trend decomposition, holiday lift overlays, and residual plots often reveal issues that metrics alone hide.

#### Statsmodels (SARIMAX)
SARIMAX is a strong classical benchmark because it is fully parametric and interpretable. It combines autoregressive terms, moving averages, differencing, and exogenous regressors in a state-space framework. This was useful as a baseline and a diagnostic model for understanding whether the weekly pattern is primarily seasonal and linear.

#### Meta Prophet
Prophet’s generalized additive model structure is appealing for retail series because it handles trend, yearly seasonality, and holiday effects in a transparent way. It often performs well on short histories because it imposes a regularized, decomposition-based structure instead of trying to fit an overly flexible state vector.

#### PyTorch (LSTM)
An LSTM is useful when nonlinear temporal dependencies are expected. It learns through recurrent memory cells and can model interactions between current demand, lagged sales, and external variables. However, it usually requires more data than classical or additive models to be competitive on short weekly series.

#### Scikit-Learn
Scikit-learn is essential for preprocessing and evaluation. `MinMaxScaler` ensures normalized inputs for neural models, and standard error metrics such as `RMSE`, `MAE`, `MAPE`, and `WAPE` allow apples-to-apples comparison across models.

---

## 3. Exploratory Data Analysis (EDA) & Statistical Findings

### Summary Statistics

The Walmart weekly sales data shows a strong right-skewed distribution due to high-volume holiday weeks and store-level heterogeneity. Baseline distributional statistics are approximately:

- Mean sales: about $1.05M/week
- Median sales: about $960K/week
- Right-skewness: demand is driven by heavy-tailed high-volume sales periods rather than a symmetric normal distribution

This matters because forecasting models that assume symmetric errors can be biased in the tails. Many retail forecasting problems are asymmetric: low-volume periods and holiday surges create long-tailed, seasonal dynamics.

### Macro Trends & Seasonality

Annual Q4 demand spikes are a recurring pattern in Walmart's weekly sales. The strongest lift occurs around Thanksgiving, Black Friday, and Christmas, producing company-wide weekly sales values often exceeding $80M during these periods.

This creates a highly seasonal pattern by calendar time and by business cycle, which explains why Prophet and seasonal models often outperform simple baselines. However, the challenge is that the seasonal pattern is not purely periodic in a standard linear sense; it interacts with holidays, promotions, and macroeconomic conditions.

### Holiday Lift Analysis

Holiday weeks produce a measurable uplift relative to non-holiday weeks:

- Holiday average sales: approximately $1.12M
- Non-holiday average sales: approximately $1.04M
- Holiday lift: roughly 7.8%

This effect is economically meaningful because holiday periods create large deviations in inventory demand, staffing needs, and logistics. They are not just noise — they are business events that should be modeled explicitly.

### Store Segmentation

There is substantial variation across stores in both volume and stability. The store-level revenue spread is large:

- Top store: Store 20, around $2.11M/week
- Bottom store: Store 33, around $259.8K/week
- Approximate variance ratio: about 8x

This indicates that a single global model would not be appropriate without store segmentation or multi-store hierarchical modeling. The project correctly focuses on Store 1 as a first benchmark and then scales to all stores later.

### External Feature Correlations

The exogenous variables do not show strong linear relationships with weekly sales:

- CPI: about -0.07 correlation
- Fuel Price: about +0.01 correlation
- Unemployment: about -0.11 correlation

This is an important finding: linear models alone are insufficient because the economic features are weak as direct linear predictors. Sales depend on nonlinear interactions between seasonality, holiday effects, and promotions rather than a single clean linear coefficient on CPI or fuel price. This explains why Prophet’s additive decomposition and LSTM's nonlinear sequence learning can outperform simple linear baselines.

---

## 4. Feature Engineering & Chronological Splitting Strategy

### Prevention of Data Leakage

Time series modeling demands strict chronological separation because random cross-validation or shuffled validation folds contaminate the future with historical information. Data leakage is especially dangerous in forecasting because the model may see future prices, future calendar events, or future sales patterns that would not be available at prediction time.

The correct approach is a strict 80/20 split:

- Train: first 80% of the ordered history
- Test: final 20% of the ordered history

This ensures that the model is evaluated only on genuinely future periods. The train/test boundary must be temporal, not random.

### Why Random K-Fold CV Fails

In standard tabular ML, K-fold cross-validation shuffles rows and randomly partitions the dataset. In time series, this breaks the causal ordering. If the model is trained on data from after the test period, then it implicitly learns the future, leading to over-optimistic metrics and invalid business decisions.

### Engineered Feature Rationale

#### Calendar Features
These are crucial because retail demand is cyclical by time of year:

- `Year`
- `Month`
- `WeekOfYear`
- `Quarter`

Calendar features capture annual and quarterly seasonality, while holiday flags capture discrete demand shocks.

#### Lag Features
Lag features represent autocorrelation and temporal dependence:

- `Lag_1`
- `Lag_2`
- `Lag_4`
- `Lag_52`

These are useful because retail sales often depend on recent history and repeating seasonal patterns. `Lag_52` helps capture annual seasonality, which is especially important in weekly retail data.

#### Rolling Window Features
Rolling statistics were computed strictly from prior data to avoid leakage:

- 4-week rolling mean
- 4-week rolling standard deviation
- 12-week rolling mean

These features help summarize short-term demand momentum and volatility. They are especially useful when recent sales performance is a stronger signal than the calendar alone.

---

## 5. Model Architecture, Benchmark & Deep Statistical Failure Analysis

### Final Benchmark Table

| Model | RMSE | MAE | MAPE |
| :--- | :--- | :--- | :--- |
| **Meta Prophet** | **$89,364.40** | **$64,160.55** | **4.07%** |
| **PyTorch LSTM** | **$106,086.94** | **$91,002.57** | **5.97%** |
| **SARIMAX** | **$518,490.72** | **$480,853.96** | **31.02%** |

### Why Prophet Won (4.07% MAPE)

Prophet won because its decomposition fits the structure of retail demand well. Its additive formulation is:

$$
 y_t = g(t) + s(t) + h(t) + \epsilon_t
 $$

where:

- $g(t)$ is the trend component,
- $s(t)$ is a periodic seasonal component,
- $h(t)$ is the holiday/regressor component,
- $\epsilon_t$ is the residual noise term.

This is especially useful in weekly retail settings because the model can handle:

- long-term trend changes,
- yearly seasonality via Fourier terms,
- holidays and special events as explicit regressors,
- strong regularization that prevents over-parameterization on short datasets.

Prophet also handles holiday signals naturally, which matters because q4 demand is not ordinary demand. Instead of trying to learn all of that from raw autoregression alone, Prophet decomposes the signal into interpretable structure and keeps the effective parameter count manageable.

### Why SARIMAX Failed (31.02% MAPE)

The SARIMAX specification used a seasonal order with $s=52$ and a baseline ordering $(1,1,1)$ with seasonal $(1,1,1,52)$. This leads to a high-dimensional seasonal state-space model. The key mathematical issue is the number of parameters relative to the data length.

With roughly 115 training weeks, there are only about 2 full seasonal cycles. Yet the seasonal component is trying to estimate a structure tied to 52 weekly phases. That creates several problems:

1. Parameter explosion: the model is trying to estimate many seasonal states with too little data.
2. Non-identifiability / collinearity: seasonal coefficients become difficult to separate from one another.
3. State-space instability: the out-of-sample forecast depends on estimated latent states that are poorly constrained.
4. Drift: a poorly specified seasonal structure drifts badly when extrapolated beyond the observed cycles.

Mathematically, this is a classic under-identification problem. The model is effectively trying to estimate far more structure than the information content allows. In short, the model is “too large for the data.”

### Why LSTM Placed Second (5.97% MAPE)

The LSTM is a strong nonlinear sequence model, but it suffered from a classic data starvation problem. For weekly retail demand with only a few hundred observations, the model does not have enough diverse temporal examples to learn stable nonlinear dynamics. This is especially true in short, tabular time series where there are fewer than 10,000 training rows.

In deep learning, performance is often dominated by two things:

- data volume,
- inductive bias relative to the problem structure.

In this task, the structure is seasonal, additive, and largely regularized by business logic. Prophet encodes this structure better than a generic recurrent model trained on a small sample. LSTM still captured useful nonlinear dynamics, but it had more variance and less stable out-of-sample extrapolation.

---

## 6. Model Trade-Off Matrix (Accuracy vs. Interpretability vs. Production Cost)

| Model | Predictive Accuracy | Interpretability | Data Appetite | Compute Cost | Maintenance Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Prophet | Best on short weekly data (MAPE 4.07%) | High: additive components are explainable | Works well with short histories | Low to moderate | Low: easier to debug and explain |
| LSTM | Good but weaker (MAPE 5.97%) | Low: black-box hidden state | Needs more data to be competitive | High: training and tuning cost | Moderate to high |
| SARIMAX | Poor on this problem (MAPE 31.02%) | Moderate to high: explicit coefficients | Fragile under small seasonal sample size | Low to moderate | Moderate: mathematically explainable but unstable |

### Summary of Trade-Offs

- Prophet is the best single-model baseline for short retail series.
- LSTM becomes attractive when there are hundreds of thousands of observations, many correlated series, or strong cross-series learning opportunities.
- SARIMAX is still useful as a transparent benchmark, but it is not the best choice for a short, high-seasonality weekly retail horizon with a large seasonal period.

---

## 7. Production System Architecture & MLOps Design

### Retraining Cadence

A sensible production policy is monthly batch retraining, with model selection and validation performed on a rolling window basis. This keeps the system responsive to changing demand patterns without retraining weekly on noisy signals.

### Drift & Anomaly Alerts

A demand forecasting system should monitor:

- 2-week rolling MAPE
- residual bias drift
- sudden shifts in forecast error by store or region
- anomalies in exogenous inputs such as extreme weather or unusual holiday effects

A recommended trigger is: if rolling 2-week MAPE exceeds 10%, automatically trigger an investigation and consider model replacement or a fallback baseline.

### Inference Pipeline

The batch inference pipeline for enterprise use should include:

1. Data ingestion from POS, sales, supply, and external feature sources.
2. Feature generation and leakage-safe transformations.
3. Store-level or SKU-level prediction generation.
4. Forecast validation against recent actuals.
5. Replenishment and labor scheduling output to downstream systems.
6. Monitoring dashboards tracking error, bias, and drift.

This architecture converts forecasting from a one-off analytical task into a continuously monitored operational system.

---

## 8. Top 10 Technical Interview Questions & Model Answers

### 1) "Why did you choose a chronological split over cross-validation?"

Because time series data has temporal dependence and future information must not leak into the training set. A chronological 80/20 split mirrors real deployment: the model trains on past data and predicts future weeks. Random K-fold would artificially improve metrics by training on future weeks, which is invalid in production.

### 2) "Why did SARIMAX struggle compared to Prophet on this specific dataset?"

SARIMAX had a high-dimensional seasonal structure with $s=52$, but only roughly 115 training weeks. That means it was trying to learn a 52-period seasonal state with only about two seasonal cycles, leading to parameter explosion and unstable state estimation. Prophet, by contrast, uses a more regularized decomposition with Fourier seasonality and explicit holiday effects.

### 3) "How did you prevent data leakage when creating rolling window features?"

I used only past values when generating rolling means and standard deviations, ensuring the features were computed on shifted historical data. That means no future information was included in the training set and the test set represented true future weeks.

### 4) "What are the trade-offs between LSTM and Prophet for enterprise demand forecasting?"

Prophet is faster to explain, easier to maintain, and often strongest on short seasonal series. LSTM is more flexible for nonlinear dependencies but requires more data and has higher operational complexity. In enterprise settings, Prophet often wins when the dataset is short and structured; LSTM wins when there are much larger datasets and more cross-series signal.

### 5) "How would you translate a 4.07% MAPE into financial business value for Walmart leadership?"

A 4.07% MAPE reduction reduces forecast error on weekly demand, which translates into less inventory overhang, lower liquidation risk, fewer stockouts, and better labor allocation. On a multi-billion-dollar retail operation, even a modest absolute forecast accuracy gain can save millions in carrying costs and lost sales.

### 6) "How would this architecture scale if you had 100,000 individual SKUs instead of 45 stores?"

I would move from a store-level monolithic model to a hierarchical or global forecasting framework, using shared parameters across SKUs, embeddings for product attributes, and time-series features with cross-series regularization. This enables information sharing across related SKUs while still allowing local adjustments.

### 7) "What metrics would you monitor in production to detect concept drift?"

I would monitor rolling MAPE, residual bias, forecast coverage, and error by store or product segment. I would also track changes in exogenous inputs like fuel price, CPI, and holiday intensity. If forecast errors begin to rise above a threshold, I would trigger a retraining or recalibration workflow.

### 8) "Why are CPI and Fuel Price poorly correlated with weekly retail sales?"

Because weekly retail demand is driven more by seasonality, holiday effects, promotions, and local store behavior than by broad macroeconomic signals alone. The relationship is nonlinear and interaction-heavy, not simply linear. Linear correlations are therefore weak despite the economic variables being informative in aggregate.

### 9) "How did you handle the variance between Store 20 and Store 33?"

I treated the project as a Store 1 proof-of-concept first and then built a scalable architecture for all stores. In production, I would use store-level stratification, local model tuning, and hierarchical sharing so high-volume and low-volume stores are treated appropriately without collapsing all demand into one average pattern.

### 10) "If given another month on this project, what improvements would you implement?"

I would expand the benchmark to a multi-store rollout, add feature importance and SHAP-style interpretation, test a hybrid Prophet + LSTM strategy, tune the LSTM architecture, and implement a lightweight production retraining pipeline with alerting and drift monitoring.

---

## Mathematical Summary

### Forecast Error Metrics

Root Mean Squared Error (RMSE):

$$
 RMSE = \sqrt{\frac{1}{n} \sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
$$

Mean Absolute Error (MAE):

$$
 MAE = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|
$$

Mean Absolute Percentage Error (MAPE):

$$
 MAPE = \frac{1}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%
$$

Weighted Absolute Percentage Error (WAPE):

$$
 WAPE = \frac{\sum_{t=1}^{n} |y_t - \hat{y}_t|}{\sum_{t=1}^{n} y_t} \times 100\%
$$

Residual definition:

$$
 e_t = y_t - \hat{y}_t
$$

### Prophet Additive Decomposition

$$
 y_t = g(t) + s(t) + h(t) + \epsilon_t
$$

where $g(t)$ is the trend, $s(t)$ is seasonality, $h(t)$ is holiday/external effect, and $\epsilon_t$ is the residual term.

---

## Final Interview Takeaway

This project demonstrates the practical truth of time series modeling: the best forecasting method depends on the problem structure, data volume, and operational constraints. On a short weekly retail series with strong seasonality and limited historical data, a regularized additive model like Prophet is often the best choice. Deep learning can become superior only when the dataset is large enough and the architecture matches the business problem.

---

## Python PDF Generation Script

```python
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.colors import HexColor
from pathlib import Path


def build_pdf_from_markdown_text(md_text: str, output_path: str = "Walmart_Forecasting_Interview_Study_Guide.pdf"):
    """Convert a markdown-like study guide into a PDF using reportlab.

    This function is intentionally lightweight and is designed for interview-study
    export use. It handles headings and paragraph text, and it can be extended to
    support tables or code blocks if needed.
    """
    doc = SimpleDocTemplate(output_path, pagesize=LETTER, leftMargin=54, rightMargin=54,
                            topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=HexColor("#1F2D3D"),
        spaceAfter=16,
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=HexColor("#1F2D3D"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=HexColor("#202020"),
        spaceAfter=8,
    )

    story = []
    lines = md_text.splitlines()
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            text = "\n".join(current_paragraph).strip()
            if text:
                story.append(Paragraph(text, body_style))
            current_paragraph.clear()

    for line in lines:
        if not line.strip():
            flush_paragraph()
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(line[2:], title_style))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(line[3:], heading_style))
        elif line.startswith("- ") or line.startswith("* "):
            flush_paragraph()
            story.append(Paragraph(line, body_style))
        else:
            current_paragraph.append(line)

    flush_paragraph()
    doc.build(story)
    print(f"PDF generated successfully: {output_path}")


if __name__ == "__main__":
    md_path = Path("walmart_forecasting_study_guide.md")
    output_pdf = "Walmart_Forecasting_Interview_Study_Guide.pdf"

    md_text = md_path.read_text(encoding="utf-8")
    build_pdf_from_markdown_text(md_text, output_path=output_pdf)
```

This markdown guide is already structured for clean PDF export and direct markdown-to-PDF conversion workflows.
