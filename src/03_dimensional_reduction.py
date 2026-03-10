import os
import pandas as pd
from sklearn.decomposition import PCA


# =====================================
# Create output folder if it doesn't exist
# =====================================
os.makedirs("data_reduced", exist_ok=True)


# =====================================
# Datasets to process
# =====================================
datasets = [
    ("data_clean/clean_1_X_train.csv", "data_clean/clean_1_X_test.csv"),
    ("data_clean/clean_2_knn_train.csv", "data_clean/clean_2_knn_test.csv"),
    ("data_clean/clean_3_model_train.csv", "data_clean/clean_3_model_test.csv"),
    ("data_clean/clean_4_median_train.csv", "data_clean/clean_4_median_test.csv"),
]


# =====================================
# Run PCA
# =====================================
for train_path, test_path in datasets:

    print("\n==============================")

    train_name = os.path.basename(train_path)
    test_name = os.path.basename(test_path)

    print(f"Running PCA for {train_name}")

    # Load data
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # ---------------------------------
    # Fit PCA on TRAIN
    # ---------------------------------
    pca = PCA()
    pca.fit(X_train)

    # Choose number of components explaining 95% variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cumulative_variance >= 0.95).argmax() + 1

    print("Selected components:", n_components)

    # ---------------------------------
    # Fit PCA with optimal components
    # ---------------------------------
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # ---------------------------------
    # Convert to DataFrame
    # ---------------------------------
    columns = [f"PC{i+1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(X_train_pca, columns=columns)
    X_test_pca = pd.DataFrame(X_test_pca, columns=columns)

    # ---------------------------------
    # Save results with correct names
    # ---------------------------------
    train_output = f"data_reduced/pca_{train_name}"
    test_output = f"data_reduced/pca_{test_name}"

    X_train_pca.to_csv(train_output, index=False)
    X_test_pca.to_csv(test_output, index=False)

    print("Saved:", train_output)
    print("Saved:", test_output)


print("\nAll PCA datasets created successfully.")