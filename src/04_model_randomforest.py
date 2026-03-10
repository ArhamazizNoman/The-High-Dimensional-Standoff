import os
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold


# =====================================
# Base paths
# =====================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CLEAN_DIR = BASE_DIR / "data_clean"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# =====================================
# Load target variable
# =====================================
y = pd.read_csv(DATA_CLEAN_DIR / "clean_y_train.csv").iloc[:, 0].values


# =====================================
# Define cleaned datasets for Random Forest
# (non-standardized, non-PCA)
# =====================================
datasets = {
    "clean1_median": {
        "train": DATA_CLEAN_DIR / "clean_1_median_train.csv",
        "test":  DATA_CLEAN_DIR / "clean_1_median_test.csv"
    },
    "clean2_knn": {
        "train": DATA_CLEAN_DIR / "clean_2_knn_train.csv",
        "test":  DATA_CLEAN_DIR / "clean_2_knn_test.csv"
    },
    "clean3_model": {
        "train": DATA_CLEAN_DIR / "clean_3_model_train.csv",
        "test":  DATA_CLEAN_DIR / "clean_3_model_test.csv"
    },
    "clean4_mean": {
        "train": DATA_CLEAN_DIR / "clean_4_mean_train.csv",
        "test":  DATA_CLEAN_DIR / "clean_4_mean_test.csv"
    }
}


# =====================================
# Cross-validation setup
# =====================================
cv = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_results = []


# =====================================
# Run Random Forest for each cleaning strategy
# =====================================
for dataset_name, paths in datasets.items():
    print("\n==============================")
    print(f"Running Random Forest for: {dataset_name}")
    print("==============================")

    # ---------------------------------
    # Check files exist
    # ---------------------------------
    if not paths["train"].exists():
        raise FileNotFoundError(f"Training file not found: {paths['train']}")
    if not paths["test"].exists():
        raise FileNotFoundError(f"Test file not found: {paths['test']}")

    # ---------------------------------
    # Load train/test data
    # ---------------------------------
    X_train = pd.read_csv(paths["train"])
    X_test = pd.read_csv(paths["test"])

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y shape      :", y.shape)

    # ---------------------------------
    # Safety checks
    # ---------------------------------
    if len(X_train) != len(y):
        raise ValueError(
            f"Mismatch between X_train rows ({len(X_train)}) and y length ({len(y)}) for {dataset_name}"
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            f"Train/Test column mismatch for {dataset_name}"
        )

    # ---------------------------------
    # Random Forest model
    # ---------------------------------
    model = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------
    # CV RMSE on train
    # ---------------------------------
    cv_scores = cross_val_score(
        model,
        X_train,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    rmse_mean = -cv_scores.mean()
    rmse_std = cv_scores.std()

    print("CV RMSE mean:", rmse_mean)
    print("CV RMSE std :", rmse_std)

    rmse_results.append({
        "model": "random_forest",
        "cleaning": dataset_name,
        "train_file": str(paths["train"]),
        "test_file": str(paths["test"]),
        "cv_rmse_mean": rmse_mean,
        "cv_rmse_std": rmse_std
    })

    # ---------------------------------
    # Fit final model on full train
    # ---------------------------------
    model.fit(X_train, y)

    # ---------------------------------
    # Predict on test set
    # ---------------------------------
    y_pred = model.predict(X_test)

    # ---------------------------------
    # Extract fitted model info
    # ---------------------------------
    print("n_estimators     :", model.n_estimators)
    print("max_features     :", model.max_features)
    print("min_samples_leaf :", model.min_samples_leaf)

    # ---------------------------------
    # Optional: feature importance summary
    # ---------------------------------
    feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_output = RESULTS_DIR / f"feature_importance_randomforest_{dataset_name}.csv"
    feature_importances.to_csv(importance_output, index=False)

    print(f"Saved feature importances to: {importance_output}")

    # ---------------------------------
    # Save predictions
    # ---------------------------------
    output_path = RESULTS_DIR / f"predictions_randomforest_{dataset_name}.csv"
    pd.DataFrame(y_pred).to_csv(output_path, index=False, header=False)

    print(f"Saved predictions to: {output_path}")


# =====================================
# Save RMSE summary
# =====================================
rmse_output = RESULTS_DIR / "rmse_randomforest.csv"
pd.DataFrame(rmse_results).to_csv(rmse_output, index=False)

print(f"\nSaved RMSE summary to: {rmse_output}")
print("All Random Forest prediction files created successfully.")