import pandas as pd
from sklearn.preprocessing import StandardScaler


# =====================================
# Datasets to standardize
# =====================================
datasets = {
    "clean_1_median": {
        "train": "data_clean/clean_1_median_train.csv",
        "test":  "data_clean/clean_1_median_test.csv"
    },
    "clean_2_knn": {
        "train": "data_clean/clean_2_knn_train.csv",
        "test":  "data_clean/clean_2_knn_test.csv"
    },
    "clean_3_model": {
        "train": "data_clean/clean_3_model_train.csv",
        "test":  "data_clean/clean_3_model_test.csv"
    },
    "clean_4_mean": {
        "train": "data_clean/clean_4_mean_train.csv",
        "test":  "data_clean/clean_4_mean_test.csv"
    }
}


# =====================================
# Standardize each cleaned dataset
# =====================================
for dataset_name, paths in datasets.items():

    print("\n==============================")
    print(f"Standardizing: {dataset_name}")
    print("==============================")

    # Load data
    X_train = pd.read_csv(paths["train"])
    X_test = pd.read_csv(paths["test"])

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # Save column names
    columns = X_train.columns

    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)

    # New filenames
    train_out = paths["train"].replace(".csv", "_standardized.csv")
    test_out = paths["test"].replace(".csv", "_standardized.csv")

    # Save standardized datasets
    X_train_scaled.to_csv(train_out, index=False)
    X_test_scaled.to_csv(test_out, index=False)

    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")

print("\nAll datasets standardized successfully.")