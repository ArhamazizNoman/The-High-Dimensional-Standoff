import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, KFold


# =====================================
# Create output folder if needed
# =====================================
os.makedirs("results", exist_ok=True)


# =====================================
# Load target variable
# =====================================
y = pd.read_csv("data_clean/clean_y_train.csv").iloc[:, 0].values


# =====================================
# Define standardized datasets for LASSO
# =====================================
datasets = {
    "clean1_median": {
        "train": "data_clean/clean_1_median_train_standardized.csv",
        "test":  "data_clean/clean_1_median_test_standardized.csv"
    },
    "clean2_knn": {
        "train": "data_clean/clean_2_knn_train_standardized.csv",
        "test":  "data_clean/clean_2_knn_test_standardized.csv"
    },
    "clean3_model": {
        "train": "data_clean/clean_3_model_train_standardized.csv",
        "test":  "data_clean/clean_3_model_test_standardized.csv"
    },
    "clean4_mean": {
        "train": "data_clean/clean_4_mean_train_standardized.csv",
        "test":  "data_clean/clean_4_mean_test_standardized.csv"
    }
}


# =====================================
# Cross-validation setup
# =====================================
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# To collect RMSE summary
rmse_results = []


# =====================================
# Run LASSO for each cleaning strategy
# =====================================
for dataset_name, paths in datasets.items():
    print("\n==============================")
    print(f"Running LASSO for: {dataset_name}")
    print("==============================")

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
    # LASSO model
    # ---------------------------------
    model = Pipeline([
        ("lasso", LassoCV(
            cv=cv,
            max_iter=20000,
            n_jobs=-1,
            random_state=42
        ))
    ])

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
        "model": "lasso",
        "cleaning": dataset_name,
        "train_file": paths["train"],
        "test_file": paths["test"],
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
    # Extract best parameters
    # ---------------------------------
    lasso = model.named_steps["lasso"]
    print("Best alpha:", lasso.alpha_)

    # Count non-zero coefficients
    non_zero = (lasso.coef_ != 0).sum()
    print("Non-zero coefficients:", non_zero)

    # ---------------------------------
    # Save predictions
    # ---------------------------------
    output_path = f"results/predictions_lasso_{dataset_name}.csv"
    pd.DataFrame(y_pred).to_csv(output_path, index=False, header=False)

    print(f"Saved predictions to: {output_path}")


# =====================================
# Save RMSE summary
# =====================================
rmse_output = "results/rmse_lasso.csv"
pd.DataFrame(rmse_results).to_csv(rmse_output, index=False)

print(f"\nSaved RMSE summary to: {rmse_output}")
print("All LASSO prediction files created successfully.")