# Walmart Demand Forecasting & Time Series Modeling
## Interview-Ready Study Guide

---

## 1. Project Overview & Business Framing

### Executive Summary
This project addresses a classic retail demand forecasting problem: predicting weekly Walmart sales across 45 stores from 2010 to 2012 using historical sales, calendar effects, and macroeconomic variables. The objective is to create a forecasting engine that supports replenishment, labor planning, and working-capital efficiency in one of the most operationally sensitive retail environments in the world.

The business challenge is to forecast expected store-level demand with enough precision to reduce overstocks, avoid stockouts, and improve weekly inventory allocation across a highly seasonal retail calendar.

### Key Business Value
Demand forecasting is mission-critical in retail because it sits at the intersection of:

- Inventory holding cost reduction
- Safety stock optimization
- Working capital efficiency
- Stockout prevention
- Labor scheduling and staffing alignment

The financial value of better forecasting is substantial. If the model reduces forecast error by even a few percentage points, the company can materially improve:

- Inventory turnover
- Cash conversion
- Markdown risk reduction
- Customer service level
- Workforce planning accuracy

### Dataset Characteristics
The dataset spans:

- 45 Walmart stores
- Weekly observations from 2010 to 2012
- Sales measured at the weekly aggregate level
- External features including:
  - Holiday flag
  - Temperature
  - Fuel price
  - CPI
  - Unemployment

This is a rich but noisy time-series problem: the signal is highly seasonal, store dependent, and influenced by both business calendars and external macro conditions.

---

## 2. Tech Stack Selection & Justification ("Why did you use X?")

| Tool | Why It Was Used | Technical Justification |
| :--- | :--- | :--- |
| Python | Core forecasting workflow | Clean scripting, rapid experimentation, reproducible pipelines, model orchestration |
| VS Code + antigravity virtual environment | Development environment | Isolates package versions and keeps the project reproducible |
| Pandas & NumPy | Data transformation and temporal operations | Date parsing, grouping, lag construction, rolling windows, vectorized computation |
| Matplotlib & Seaborn | Visualization | Macro trends, holiday peaks, residual diagnostics, distribution analysis |
| Statsmodels (SARIMAX) | Classical time-series benchmark | Explicit state-space structure with exogenous variables |
| Meta Prophet | Primary benchmark model | Additive decomposition with trend, seasonality, and holiday components |
| PyTorch (LSTM) | Deep sequence benchmark | Captures nonlinear temporal dependencies and sequence patterns |
| Scikit-Learn | Evaluation and scaling | `MinMaxScaler`, RMSE, MAE, MAPE, and WAPE metrics |

### Why these tools were appropriate

#### Python & VS Code
Python is the natural language for forecasting because of its ecosystem: scientific computing, deep learning, statistical modeling, and plotting. VS Code allowed modular script execution, quick iteration, and reproducible development practices.

#### Pandas & NumPy
Pandas was used to:

- Parse weekly dates correctly
- Sort by store and date
- Generate lag features without leakage
- Compute rolling windows using only historical data

NumPy was essential for:

- Array-based transformations
- Efficient aggregations
- Scaling and feature preparation

#### Matplotlib & Seaborn
These tools helped explain:

- Company-wide macro trends
- Q4 holiday lifts
- Distribution skewness
- Residual drift
- Outlier behavior across stores

Visualization matters because the model is judged for robustness, not just reported metrics.

#### Statsmodels (SARIMAX)
SARIMAX was used as a classical baseline because it offers:

- Trend and seasonality modeling
- Exogenous regressors
- Statistical diagnostics for residual structure

It is powerful when the series is long and heavily identified. However, for short seasonal series with limited sample depth, it becomes fragile.

#### Meta Prophet
Prophet is especially useful here because:

- It handles weekly data well
- Seasonal components can be decomposed using Fourier terms
- Holiday effects can be modeled explicitly
- It is less sensitive to short-horizon instability than a complex seasonal ARIMA structure

It is also highly explainable to non-technical stakeholders.

#### PyTorch (LSTM)
An LSTM was tested as a nonlinear sequence model capable of learning complex dependencies. It is suitable when:

- There is enough data
- The sequence contains nonlinear patterns
- More flexibility than classical additive models is needed

For this dataset, the weakness was not model capability but the short historical sample size.

#### Scikit-Learn
Scikit-Learn provided:

- Robust scaling utilities
- Standard metrics for model comparison
- Consistent evaluation pipelines

The key metrics were RMSE, MAE, MAPE, and WAPE, applied under rigorous chronological evaluation.

---

## 3. Exploratory Data Analysis (EDA) & Statistical Findings

### Summary Statistics
The weekly sales distribution is heavy-tailed and right-skewed. High-level distributional facts:

| Metric | Value |
| :--- | :---: |
| Mean Weekly Sales | $1.05M |
| Median Weekly Sales | $960K |
| Distribution Shape | Right-skewed |
| Store-Level Variance | High |

The distribution suggests that most weeks are moderate in size, but major holiday periods push the mean upward sharply.

### Macro Trends & Seasonality
The macro sales trend shows strong Q4 acceleration driven by:

- Thanksgiving
- Black Friday
- Christmas
- End-of-year shopping intensity

The aggregated weekly sales across all stores exceeded $80M during the most active holiday weeks, reflecting a meaningful seasonal amplitude.

The business signal is not just “a bump”; it is a sustained, predictable uplift around the retail calendar.

### Holiday Lift Analysis
Holiday weeks showed a material sales lift. A representative finding is:

$$
\text{Holiday Lift} = \frac{1.12M - 1.04M}{1.04M} \times 100 \approx 7.8\%
$$

This means:

- Average holiday-week sales ≈ $1.12M
- Average non-holiday-week sales ≈ $1.04M

This is a strong indicator that holiday effects are essential features rather than minor noise.

### Store Segmentation
There was a substantial performance gap across stores.

| Store | Avg Weekly Sales |
| :--- | :---: |
| Store 20 | $2.11M |
| Store 33 | $259.8K |

This is roughly an 8x spread, which reinforces that store-level heterogeneity matters. A model that ignores store-specific behavior will underperform.

### External Feature Correlations
The external variables had weak linear relationships with weekly sales:

| Feature | Correlation with Sales |
| :--- | :---: |
| CPI | -0.07 |
| Fuel Price | +0.01 |
| Unemployment | -0.11 |

These low correlations imply:

- External macro variables are not independently strong linear drivers
- Nonlinear and interaction effects are likely important
- Models relying only on linear relationships are insufficient

This is one of the strongest arguments for a decomposition model like Prophet and a nonlinear sequence model like an LSTM.

---

## 4. Feature Engineering & Chronological Splitting Strategy

### Prevention of Data Leakage
Random K-fold cross-validation is inappropriate for time series because it breaks chronology and allows future information to leak into training data. For a time series:

- Observations are dependent
- Distribution shifts occur over time
- The future must not be used to predict the past

Therefore, the correct strategy is strict chronological splitting.

### Why 80/20 Chronological Splitting Is Mandatory
An 80/20 chronological split preserves the real forecasting problem:

- Train on earlier observations
- Test on the final 20% of the time horizon

This ensures evaluation matches the actual deployment situation and avoids artificially optimistic accuracy.

### Engineered Features
Key engineered features included:

#### Calendar Features
- Year
- Month
- WeekOfYear
- Quarter

These capture:

- Seasonality
- Annual cycles
- Recurring retail phases

#### Lag Features
Examples:

- Lag_1
- Lag_2
- Lag_4
- Lag_52

Lag_52 is especially important because it encodes annual seasonality. On weekly data, a 52-week lag is highly informative.

#### Rolling Window Aggregations
Examples:

- 4-week rolling mean
- 4-week rolling standard deviation

These were computed strictly from historical values:

$$
\text{RollingMean}_{t} = \frac{1}{4}\sum_{i=1}^{4} y_{t-i}
$$

This helps the model estimate local trend and volatility while preserving causality.

---

## 5. Model Architecture, Benchmark & Deep Statistical Failure Analysis

### Final Benchmark Table

| Model | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: |
| Meta Prophet | $89,364.40 | $64,160.55 | 4.07% |
| PyTorch LSTM | $106,086.94 | $91,002.57 | 5.97% |
| SARIMAX | $518,490.72 | $480,853.96 | 31.02% |

### Why Prophet Won
Prophet achieved the strongest result with a MAPE of 4.07%.

This is because Prophet combines:

- Local trend decomposition
- Fourier-based annual seasonality
- Additive holiday regressors
- Strong regularization

Mathematically, the model resembles:

$$
y_t = T_t + S_t + H_t + \epsilon_t
$$

where:

- $T_t$ = trend
- $S_t$ = seasonal component
- $H_t$ = holiday effect
- $\epsilon_t$ = residual

The key advantage is not only flexibility but the correct amount of structural prior. Prophet does not require decades of history to model annual seasonality; Fourier terms are enough.

### Why SARIMAX Failed
SARIMAX performed poorly, with MAPE of 31.02%.

The root cause was seasonal under-identification:

- Seasonal period $s = 52$
- Seasonal order too large relative to available training data
- Only about 115 training weeks available

This is roughly two annual cycles, which is not enough to estimate a stable 52-period seasonal system with confidence.

The resulting model had:

- Parameter explosion
- Weak identification
- Near-collinearity among seasonal states
- Numerical instability in the state-space recursion

This is why the model drifted badly on the test window.

### Why LSTM Placed Second
The LSTM achieved 5.97% MAPE, which is respectable but still behind Prophet.

This is a classic data-starvation issue. Neural networks need enough variation and enough examples to learn stable temporal structure. Weekly retail data with roughly 115 training points is often too short for deep sequence models to generalize well without strong regularization or broader series coverage.

In many enterprise settings, deep models win only when:

- There are many related series
- The task contains more nonlinear structure
- There is enough data volume for shared representation learning

This was not the case here.

---

## 6. Model Trade-Off Matrix (Accuracy vs. Interpretability vs. Production Cost)

| Model | Accuracy | Interpretability | Computational Cost | Data Appetite | Maintenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Meta Prophet | Best | High | Low to moderate | Works well on short histories | Low |
| PyTorch LSTM | Good but behind Prophet | Low | High | Requires more data | Moderate to high |
| SARIMAX | Weak in this setting | Moderate to high | Low to moderate | Fragile with short seasonal series | Moderate |

### Discussion
- Prophet is the operational winner because it balances accuracy and explainability.
- LSTM is strong for nonlinear sequence learning but expensive and harder to debug.
- SARIMAX is transparent but statistically brittle here due to underpowered seasonal configuration.

### Decision Guidance
For this project:

- Productive baseline: Prophet
- Advanced alternative: LSTM only if data scale increases significantly
- Benchmark model: SARIMAX, but not as a production engine

---

## 7. Production System Architecture & MLOps Design

### Retraining Cadence
A practical production design would use:

- Monthly retraining
- Weekly forecast generation
- Backtesting against prior windows

This ensures the model adapts to changes in:

- Holiday calendar timing
- Store mix and region behavior
- Inflation and cost shifts
- Local demand pattern changes

### Drift & Anomaly Alerts
A production monitoring rule could be:

$$
\text{If 2-week rolling MAPE} > 10\%, \text{ trigger alert}
$$

Additional monitoring signals:

- Forecast bias
- Residual mean shift
- Store-level anomaly deviations
- Holiday-period forecast misses
- Worsening WAPE over rolling windows

### Inference Pipeline
The production pipeline should be:

1. Pull latest weekly sales and feature snapshots
2. Validate data completeness and chronology
3. Generate lag and rolling features
4. Score each store with the selected model
5. Aggregate into replenishment recommendations
6. Export weekly store-level order quantities
7. Monitor actuals versus forecast
8. Retrain on schedule or when drift triggers

### Operational Recommendations
- Use Prophet as the baseline production model
- Maintain a fallback rule-based or simpler benchmark
- Keep a human review panel for holiday anomalies
- Log every forecast and residual for weekly business review

---

## 8. Top 10 Technical Interview Questions & Model Answers

### 1) “Why did you choose a chronological split over cross-validation?”
Situation: Demand forecasting is time-dependent.  
Task: Evaluate model performance in realistic conditions.  
Action: I used an 80/20 chronological split so the test set represented future periods, avoiding leakage.  
Result: This produced honest out-of-sample accuracy and is the correct standard for time-series forecasting.

### 2) “Why did SARIMAX struggle compared to Prophet on this specific dataset?”
Situation: The data had weekly sales and only around 115 training weeks.  
Task: Fit a seasonal model with annual effects.  
Action: SARIMAX attempted a seasonal period of 52 with too many parameters relative to the sample size.  
Result: This created instability, weak identification, and poor out-of-sample forecasts, while Prophet’s regularized additive structure handled the seasonality more robustly.

### 3) “How did you prevent data leakage when creating rolling window features?”
Situation: Forecasting features must be causal.  
Task: Build lag and rolling features without using future information.  
Action: I used shifted data so each feature depended only on historical observations.  
Result: The train/test split remained valid and the model reflected real deployment behavior.

### 4) “What are the trade-offs between LSTM and Prophet for enterprise demand forecasting?”
Situation: We needed a model that balances predictive power and operational simplicity.  
Task: Compare nonlinear deep learning to an interpretable decomposition model.  
Action: I benchmarked both; Prophet performed better on short history while LSTM was more computationally expensive and less transparent.  
Result: Prophet was more deployable and easier to explain to stakeholders.

### 5) “How would you translate a 4.07% MAPE into financial business value for Walmart leadership?”
Situation: Leadership cares about business impact rather than model metrics alone.  
Task: Quantify forecast quality in financial terms.  
Action: I would map MAPE to avoided inventory carrying costs, reduced stockouts, and fewer markdowns.  
Result: Even a small reduction in forecast error can save substantial working capital and improve service levels.

### 6) “How would this architecture scale if you had 100,000 individual SKUs instead of 45 stores?”
Situation: The problem becomes much larger and more fragmented.  
Task: Design a scalable forecasting system.  
Action: I would adapt the pipeline to product-store hierarchies and train global-local models with modular feature engineering.  
Result: The architecture could remain tractable with hierarchy-aware modeling, distributed processing, and automated drift monitoring.

### 7) “What metrics would you monitor in production to detect concept drift?”
Situation: Forecast performance can deteriorate over time.  
Task: Detect when the relationship between features and targets changes.  
Action: I would monitor rolling MAPE, WAPE, bias, residual variance, and store-level anomaly rates.  
Result: This would trigger retraining or investigation before inventory decisions deteriorate.

### 8) “Why are CPI and Fuel Price poorly correlated with weekly retail sales?”
Situation: Macro variables often look informative in theory but not in practice.  
Task: Understand weak feature signal.  
Action: I looked at correlations and observed that weekly retail sales are dominated by seasonality, promotions, and store-level behavior rather than linear macro shifts.  
Result: This confirmed that the real signal depends more on calendar effects and local demand patterns than on raw macro indicators.

### 9) “How did you handle the variance between Store 20 and Store 33?”
Situation: Different stores had very different sales levels.  
Task: Ensure the model captures heterogeneity.  
Action: I analyzed store-level segmentation and used store-aware forecasting with time features and local demand patterns.  
Result: The workflow accounted for systematic differences instead of forcing all stores into a single average pattern.

### 10) “If given another month on this project, what improvements would you implement?”
Situation: The current benchmark is strong but not final.  
Task: Improve the system further.  
Action: I would add richer holiday encodings, region-level effects, interaction terms, and a stronger cross-store feature strategy; I would also test a hybrid Prophet + residual model approach.  
Result: This would reduce remaining forecast error while preserving explainability and operational reliability.

---

## Final Interview Takeaway
This project is a strong example of modern forecasting work because it balances:

- Business framing
- Feature engineering
- Model benchmarking
- Diagnostics
- Production thinking

If asked in an interview, the strongest narrative is:

> “I did not choose the most complex model; I chose the model that was statistically appropriate for the data. On a short, seasonal, weekly retail series, Prophet outperformed SARIMAX and LSTM because it matched the underlying structure of the problem better than a large seasonal parametric model or a data-hungry neural network.”

That is the essence of a high-quality ML engineering answer.

---

## PDF Generation Script

The script below creates a PDF named `Walmart_Forecasting_Interview_Study_Guide.pdf` using `reportlab`.

```python
# generate_walmart_study_guide_pdf.py
# pip install reportlab

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)

OUTPUT_FILE = "Walmart_Forecasting_Interview_Study_Guide.pdf"


def build_document():
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=1,
        spaceAfter=18,
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceBefore=18,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-10,
    )

    story = []
    story.append(Paragraph("Walmart Demand Forecasting & Time Series Modeling", title_style))
    story.append(Paragraph("Interview-Ready Study Guide", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    def add_paragraph(text):
        story.append(Paragraph(text, body_style))

    def add_heading(text):
        story.append(Paragraph(text, h1_style))

    def add_subheading(text):
        story.append(Paragraph(text, h2_style))

    def add_bullets(items):
        story.append(ListFlowable(
            [ListItem(Paragraph(item, bullet_style), leftIndent=18) for item in items],
            bulletType="bullet",
            bulletFontName="Times-Bold",
            bulletFontSize=10,
            leftIndent=18,
            spaceBefore=4,
            spaceAfter=6,
        ))

    add_heading("1. Project Overview & Business Framing")
    add_subheading("Executive Summary")
    add_paragraph(
        "This project addresses a classic retail demand forecasting problem: predicting weekly Walmart sales across 45 stores from 2010 to 2012 using historical sales, calendar effects, and macroeconomic variables."
    )
    add_paragraph(
        "Demand forecasting is critical because it improves inventory holding cost, safety stock optimization, working capital efficiency, stockout prevention, and labor scheduling."
    )

    add_heading("2. Tech Stack Selection & Justification")
    table_data = [
        ["Tool", "Why It Was Used", "Technical Justification"],
        ["Python", "Core forecasting workflow", "Clean scripting, rapid experimentation, reproducible pipelines"],
        ["VS Code + antigravity", "Development environment", "Isolated package versions and reproducible setup"],
        ["Pandas & NumPy", "Temporal indexing and feature engineering", "Lag generation, rolling windows, and fast vectorized operations"],
        ["Matplotlib & Seaborn", "Visualization", "Trend analysis, holiday lifts, residual diagnostics"],
        ["Statsmodels SARIMAX", "Classical statistical baseline", "State-space structure with exogenous effects"],
        ["Meta Prophet", "Production benchmark", "Additive decomposition with seasonal and holiday components"],
        ["PyTorch LSTM", "Deep sequence model", "Captures nonlinear temporal patterns"],
        ["Scikit-Learn", "Evaluation and scaling", "MinMaxScaler, RMSE, MAE, MAPE, WAPE"],
    ]
    table = Table(table_data, colWidths=[1.0 * inch, 1.7 * inch, 2.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    add_heading("3. Exploratory Data Analysis (EDA) & Statistical Findings")
    add_paragraph("Mean Weekly Sales: $1.05M")
    add_paragraph("Median Weekly Sales: $960K")
    add_paragraph("Distribution: right-skewed with pronounced holiday-driven peaks")
    add_paragraph("Macro Trends & Seasonality: Q4 demand spikes from Thanksgiving, Black Friday, and Christmas drove weekly sales above $80M.")
    add_paragraph("Holiday Lift: average holiday week sales were approximately $1.12M versus $1.04M for non-holiday weeks, a lift of about 7.8%.")
    add_paragraph("Store variance: Store 20 averaged $2.11M weekly while Store 33 averaged $259.8K; this is roughly an 8x spread.")
    add_paragraph("External feature correlations: CPI = -0.07, Fuel Price = +0.01, Unemployment = -0.11, which shows that linear relationships alone are insufficient.")

    add_heading("4. Feature Engineering & Chronological Splitting Strategy")
    add_paragraph("Random K-fold cross-validation is inappropriate for time series because future observations leak into training data.")
    add_paragraph("A strict chronological 80/20 split is mandatory to simulate real deployment conditions.")
    add_paragraph("Lag features included Lag_1, Lag_2, Lag_4, and Lag_52 to capture short-term dependence and annual seasonality.")
    add_paragraph("Rolling features such as 4-week moving averages and standard deviations were computed using only historical values to preserve causality.")

    add_heading("5. Model Architecture, Benchmark & Deep Statistical Failure Analysis")
    benchmark = [
        ["Model", "RMSE", "MAE", "MAPE"],
        ["Meta Prophet", "$89,364.40", "$64,160.55", "4.07%"],
        ["PyTorch LSTM", "$106,086.94", "$91,002.57", "5.97%"],
        ["SARIMAX", "$518,490.72", "$480,853.96", "31.02%"],
    ]
    btable = Table(benchmark, colWidths=[1.8 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch])
    btable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(btable)
    story.append(Spacer(1, 0.12 * inch))

    add_subheading("Why Prophet Won")
    add_paragraph("Prophet achieved the best result because its additive decomposition captures trend, annual seasonality, and holiday effects correctly. With a short weekly series, the model benefits from regularization and a strong structural prior.")
    add_subheading("Why SARIMAX Failed")
    add_paragraph("SARIMAX was statistically underpowered because it tried to estimate a 52-week seasonal pattern with only about 115 training points, creating parameter explosion and numerical instability.")
    add_subheading("Why LSTM Placed Second")
    add_paragraph("The LSTM is a strong nonlinear model, but this problem is data-starved; with limited weekly observations, deep sequence models have higher variance than regularized additive models.")

    add_heading("6. Model Trade-Off Matrix")
    tradeoff = [
        ["Model", "Accuracy", "Interpretability", "Compute Cost", "Data Appetite"],
        ["Prophet", "Best", "High", "Low to Moderate", "Good on short history"],
        ["LSTM", "Good", "Low", "High", "Needs more data"],
        ["SARIMAX", "Poor here", "Moderate to High", "Low to Moderate", "Fragile on short seasonal series"],
    ]
    ttable = Table(tradeoff, colWidths=[1.3 * inch, 1.2 * inch, 1.5 * inch, 1.5 * inch, 1.8 * inch])
    ttable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3A5A40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(ttable)

    add_heading("7. Production System Architecture & MLOps Design")
    add_bullets([
        "Retraining cadence: monthly batch retraining scheduled automatically.",
        "Drift monitoring: trigger if rolling 2-week MAPE exceeds 10%.",
        "Inference pipeline: batch-scoring weekly store-level demand and exporting replenishment orders.",
        "Operational review: compare forecast vs actuals after every weekly cycle.",
    ])

    add_heading("8. Top 10 Technical Interview Questions")
    add_bullets([
        "Why did you choose a chronological split over cross-validation?",
        "Why did SARIMAX struggle compared to Prophet on this dataset?",
        "How did you prevent data leakage when creating rolling window features?",
        "What are the trade-offs between LSTM and Prophet for enterprise forecasting?",
        "How would you translate 4.07% MAPE into business value?",
        "How would the architecture scale with 100,000 SKUs?",
        "What metrics would you monitor for concept drift?",
        "Why are CPI and Fuel Price poorly correlated with sales?",
        "How did you handle the variance between Store 20 and Store 33?",
        "If given another month, what improvements would you implement?",
    ])

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Prepared for technical interview readiness and executive-level communication.", body_style))

    doc.build(story)
    print(f"Saved PDF: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_document()
```

This script produces a clean PDF version of the study guide, ready for export, sharing, or printing.
