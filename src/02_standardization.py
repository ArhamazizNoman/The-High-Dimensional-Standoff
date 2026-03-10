import pandas as pd
from sklearn.preprocessing import StandardScaler


# =====================================
# Datasets to standardize
# =====================================
datasets = {
    "clean1_mean": {
        "train": "data_clean/clean_1_X_train.csv",
        "test":  "data_clean/clean_1_X_test.csv"
    },
    "clean2_knn": {
        "train": "data_clean/clean_2_knn_train.csv",
        "test":  "data_clean/clean_2_knn_test.csv"
    },
    "clean3_model": {
        "train": "data_clean/clean_3_model_train.csv",
        "test":  "data_clean/clean_3_model_test.csv"
    },
    "clean4_median": {
        "train": "data_clean/clean_4_median_train.csv",
        "test":  "data_clean/clean_4_median_test.csv"
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

    print("Original train shape:", X_train.shape)
    print("Original test shape :", X_test.shape)

    # Keep original column names
    columns = X_train.columns

    # Fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)

    # Overwrite files in data_clean
    X_train_scaled.to_csv(paths["train"], index=False)
    X_test_scaled.to_csv(paths["test"], index=False)

    print(f"Overwritten train file: {paths['train']}")
    print(f"Overwritten test file : {paths['test']}")

print("\nAll cleaned X datasets were standardized successfully.")