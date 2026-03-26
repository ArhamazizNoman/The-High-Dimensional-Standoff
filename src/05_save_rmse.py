from pathlib import Path
import pandas as pd
import os

# ============================
# BASE PATH (project root)
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================
# USER INPUT
# ============================

RMSE_FILE = BASE_DIR / "results" / "rmse_elasticnet.csv"
MODEL_NAME = "elasticnet"
OUTPUT_FILE = BASE_DIR / "estimatedRMSE_studentID1_studentID2.csv"


# ============================
# LOAD DATA
# ============================

if not os.path.exists(RMSE_FILE):
    raise FileNotFoundError(f"File not found: {RMSE_FILE}")

df = pd.read_csv(RMSE_FILE)

print("\nLoaded RMSE file:")
print(df.head())


# ============================
# FILTER BY MODEL
# ============================

df_model = df[df["model"] == MODEL_NAME]

if df_model.empty:
    raise ValueError(f"No rows found for model = {MODEL_NAME}")


# ============================
# SELECT BEST CONFIGURATION
# (minimum RMSE)
# ============================

best_row = df_model.loc[df_model["cv_rmse_mean"].idxmin()]

rmse_value = best_row["cv_rmse_mean"]

print("\nBest configuration found:")
print(best_row)

print(f"\nSelected RMSE: {rmse_value}")


# ============================
# SAVE FINAL OUTPUT
# (single value, no header)
# ============================

pd.DataFrame([rmse_value]).to_csv(OUTPUT_FILE, index=False, header=False)

print(f"\nFinal RMSE file saved: {OUTPUT_FILE}")