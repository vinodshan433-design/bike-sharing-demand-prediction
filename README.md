# Bike Sharing Demand Prediction & Analytics

**Student:** vinod
**Email:** vinodshan433@gmail.com

## Project overview
This project predicts hourly bike rental demand (`cnt`) using weather, calendar, and time-related variables. It creates cyclic hour/month features and commute-hour indicators, trains several regression models, and compares their performance using MAE, RMSE, and R².

## Project files
- `code.py` — feature engineering, model training, evaluation, and metrics export.
- `requirements.txt` — Python dependencies.
- `Project_Report.md` — project report and methodology.
- `README.md` — setup and usage guide.
- `data/hour.csv` — dataset file to add locally (not included in this package).
- `model_results.csv` — generated after a successful run.

## Dataset setup
Use the UCI Bike Sharing Dataset. Download and extract it, then place `hour.csv` in a folder named `data` beside `code.py`:

```text
project-folder/
├── code.py
├── requirements.txt
├── README.md
├── Project_Report.md
└── data/
    └── hour.csv
```

## Installation
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python code.py
```

The script prints a model comparison table and saves `model_results.csv` in the project folder. XGBoost and LightGBM are optional at runtime; if either package is unavailable, that model is skipped.

## Evaluation metrics
- **MAE:** average absolute difference between actual and predicted rental counts.
- **RMSE:** square root of the average squared prediction error; it penalizes large errors more strongly.
- **R²:** proportion of target variation explained by the model (higher is generally better).

## Important notes
- The dataset is not bundled; add `data/hour.csv` before running.
- The notebook supplied for this project referenced `results_df` without defining it. This Python script completes the model training and evaluation workflow.
- The target is transformed with `log1p` during training and converted back with `expm1` for evaluation.
