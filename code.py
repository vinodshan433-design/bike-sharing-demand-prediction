"""
Bike Sharing Demand Prediction & Analytics
Student: vinod | Email: vinodshan433@gmail.com

Place the UCI Bike Sharing Dataset's hour.csv file in the data/ folder,
then run: python code.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, VotingRegressor
)

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [ROOT / "data" / "hour.csv", ROOT.parent / "data" / "hour.csv"]

def load_data():
    for path in DATA_PATHS:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found. Download the Bike Sharing Dataset and place hour.csv "
        "inside the project's data/ folder. See README.md."
    )

def engineer_features(df):
    df = df.copy()
    required = {"hr", "mnth", "workingday", "temp", "hum", "cnt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["sin_hr"] = np.sin(2 * np.pi * df["hr"] / 24.0)
    df["cos_hr"] = np.cos(2 * np.pi * df["hr"] / 24.0)
    df["sin_mnth"] = np.sin(2 * np.pi * df["mnth"] / 12.0)
    df["cos_mnth"] = np.cos(2 * np.pi * df["mnth"] / 12.0)
    df["is_rush_hour_morning"] = df["hr"].isin([7, 8, 9]).astype(int)
    df["is_rush_hour_evening"] = df["hr"].isin([17, 18, 19]).astype(int)
    df["commute_peak"] = (
        (df["workingday"] == 1) &
        ((df["is_rush_hour_morning"] == 1) |
         (df["is_rush_hour_evening"] == 1))
    ).astype(int)
    df["temp_hum_ratio"] = df["temp"] / (df["hum"] + 0.01)
    return df

def get_models():
    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=42, n_jobs=-1
        )
    except ImportError:
        print("Optional package xgboost not installed; skipping XGBoost.")
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1
        )
    except ImportError:
        print("Optional package lightgbm not installed; skipping LightGBM.")
    return models

def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    df = engineer_features(df)

    drop_cols = ["instant", "dteday", "cnt", "casual", "registered"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["cnt"].astype(float)

    # Keep the same random split used by the original notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = []
    trained = {}
    for name, model in get_models().items():
        try:
            # Ridge uses a pipeline scaler; tree-based models need no scaling.
            model.fit(X_train, np.log1p(y_train))
            pred_log = model.predict(X_test)
            pred = np.maximum(0, np.expm1(pred_log))
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred)
            })
            trained[name] = model
        except Exception as exc:
            print(f"Skipping {name} due to training error: {exc}")

    if not results:
        raise RuntimeError("No models trained successfully.")
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== MODEL BENCHMARK RESULTS (lower MAE/RMSE is better) ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    results_df.to_csv(ROOT / "model_results.csv", index=False)
    print(f"\nSaved model metrics to: {ROOT / 'model_results.csv'}")
    print("\nNote: metrics depend on the dataset version and installed packages.")

if __name__ == "__main__":
    main()
