# DemandOS: Walmart Demand Forecasting & Inventory Decision Engine

A production-style retail forecasting and inventory optimization project that turns historical sales and macroeconomic signals into actionable inventory decisions. Built for Walmart-style demand planning, this system combines time-series forecasting, uncertainty-aware planning, and an interactive decision dashboard to help retailers move from descriptive analytics to prescriptive action.

Repository: https://github.com/prajwalmaralihalli/deeplearning_project.git

## Why this project

Retail and supply-chain teams often have good historical data but limited visibility into what should be ordered next, how much safety stock is needed, and how uncertainty affects service levels. This project addresses that gap by combining:

- demand forecasting with probabilistic uncertainty bands,
- risk-aware inventory planning,
- model benchmark evaluation, and
- a business-facing dashboard for decision support.

It is designed as a portfolio-ready demonstration of how machine learning can support real operations decisions.

## Project overview

The workflow ingests weekly store-level sales data with external drivers such as CPI, fuel price, and unemployment, engineers leakage-safe features, evaluates several forecasting models, and then converts forecast uncertainty into inventory actions and risk scoring.

```text
Historical sales + demand drivers
        |
        v
Feature engineering and validation
        |
        v
Forecast generation (P10 / P50 / P90)
        |
        v
Inventory policy and reorder decisions
        |
        v
Dashboard, reports, and operational insights
```

## Key capabilities

### Forecasting and benchmarking

- Weekly demand forecasting for individual stores and future horizons
- P10, P50, and P90 forecast bands for uncertainty-aware planning
- Benchmarking across multiple models including LSTM, SARIMAX, and Prophet
- Chronological validation using RMSE, MAE, MAPE, WAPE, and directional accuracy
- Forecast drift detection and monitoring over time

### Inventory decision engine

- Safety-stock and reorder recommendation logic based on forecasted demand
- Scenario-based payoff analysis for inventory actions under multiple outcomes
- Holding cost and stockout penalty evaluation for operational trade-offs
- Reorder guidance tied to expected demand versus current stock levels
- Risk heatmaps to highlight lower-cost and more resilient inventory strategies

### Business-facing analytics

- Interactive Streamlit dashboard for scenario analysis and decision discovery
- Executive summary generation for business reporting
- Plotly visualizations for model comparison, residual diagnostics, and forecast logic
- PDF-ready reporting workflow for stakeholder communication
- Store and horizon selectors for operational planning workflows

## Tech stack

- Python
- Streamlit
- FastAPI
- PyTorch
- statsmodels
- Prophet
- pandas / NumPy / SciPy / scikit-learn
- Plotly / Matplotlib / Seaborn
- pdfkit + wkhtmltopdf
- pytest

## Repository structure

```text
.
├── app.py                          # Streamlit application entry point
├── api.py                         # Optional FastAPI service
├── pipeline.py                    # Forecasting, retraining, and pipeline orchestration
├── phase1_eda.py                  # Feature engineering and exploratory analysis
├── phase2_models.py               # Model training and benchmark generation
├── phase3_evaluation.py           # Forecast evaluation and diagnostics
├── inventory_engine.py             # Inventory economics and reorder recommendations
├── run_all.py                     # End-to-end local pipeline runner
├── run_system.py                  # System-level execution entry point
├── verify_backend.py              # Backend validation checks
├── requirements.txt               # Python dependencies
├── data/                          # Engineered data, forecasts, and evaluation outputs
├── models/                        # Trained model artifacts and metadata
├── plots/                         # Forecast and evaluation plots
├── src/                           # Shared service and UI logic
├── tests/                         # Automated tests
├── README.md                      # Project documentation
├── Dockerfile                     # Container build definition
├── docker-compose.yml             # Local orchestration setup
├── BACKEND_ML_PIPELINE.md          # Backend engineering notes
├── PROJECT_ROADMAP.md             # Project roadmap
├── SYSTEM_TECHNICAL_DOCUMENTATION.md
├── TECH_STACK.md                  # Implementation stack reference
├── DEMANDOS_UI_ARCHITECTURE.md    # Dashboard architecture notes
├── Walmart_Forecasting_Interview_Study_Guide.md
└── .github/                       # GitHub workflow and project automation
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/prajwalmaralihalli/deeplearning_project.git
cd deeplearning_project
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

The project expects Walmart-style retail inputs with weekly sales and exogenous drivers. A representative dataset should include fields such as:

```text
Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
```

Place the data in the project root or ensure the pipeline points to the intended file location before running the workflow.

## Running the project

### Full pipeline

```bash
python run_all.py
```

This runs the end-to-end workflow including feature engineering, model training, evaluation, and forecast generation.

### Stepwise execution

```bash
python phase1_eda.py
python phase2_models.py
python phase3_evaluation.py
python inventory_engine.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

The app will typically be available at:

```text
http://localhost:8501
```

### Start the API (optional)

```bash
uvicorn api:app --reload --port 8000
```

Open the API docs at:

```text
http://localhost:8000/docs
```

## Model logic and evaluation

This project follows a practical forecasting workflow for retail operations:

1. Clean and transform historical sales data
2. Engineer time-aware features and demand drivers
3. Train benchmark models on chronological splits
4. Compare performance on holdout periods
5. Generate forecast intervals and operational actions
6. Turn uncertainty into inventory planning decisions

Evaluation metrics include RMSE, MAE, MAPE, WAPE, and directional accuracy, with artifacts saved to the `data/` and `plots/` directories for review and reporting.

## Deployment notes

The project is configured for local execution and can be adapted to cloud environments such as Streamlit Community Cloud or a containerized deployment. For PDF export functionality, the environment should include the external `wkhtmltopdf` binary.

Windows example:

```text
C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe
```

If the binary is unavailable, the dashboard can still operate without PDF report generation.

## Validation

Check syntax:

```bash
python -m py_compile app.py
```

Run tests:

```bash
pytest
```

## License

This project is currently provided as an open portfolio and research project. Add an explicit license before public commercial use if required by your deployment or publishing policy.

## Acknowledgements

This project demonstrates an applied approach to forecasting, optimization, and business analytics in a retail setting. It is intended as a useful reference for data science, machine learning, and operations research workflows in demand planning.
