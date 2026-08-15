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
