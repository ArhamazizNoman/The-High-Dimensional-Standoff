import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =====================================
# Create output folder if it doesn't exist
# =====================================
os.makedirs("data_reduced", exist_ok=True)


# =====================================
# Standardized datasets to process
# =====================================
datasets = [
    (
        "data_clean/clean_1_median_train_standardized.csv",
        "data_clean/clean_1_median_test_standardized.csv"
    ),
    (
        "data_clean/clean_2_knn_train_standardized.csv",
        "data_clean/clean_2_knn_test_standardized.csv"
    ),
    (
        "data_clean/clean_3_model_train_standardized.csv",
        "data_clean/clean_3_model_test_standardized.csv"
    ),
    (
        "data_clean/clean_4_mean_train_standardized.csv",
        "data_clean/clean_4_mean_test_standardized.csv"
    ),
]


# =====================================
# Run PCA
# =====================================
for train_path, test_path in datasets:

    print("\n==============================")

    train_name = os.path.basename(train_path)
    test_name = os.path.basename(test_path)

    print(f"Running PCA for: {train_name}")

    # Load standardized data
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # ---------------------------------
    # Fit PCA on training data
    # ---------------------------------
    pca_full = PCA()
    pca_full.fit(X_train)

    cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

    # Select number of components for 95% explained variance
    n_components = (cumulative_variance >= 0.95).argmax() + 1

    print("Selected number of components:", n_components)
    print("Explained variance:", cumulative_variance[n_components-1])

    # ---------------------------------
    # Scree Plot (variance explained)
    # ---------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(cumulative_variance)
    plt.axhline(y=0.95)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"Scree Plot - {train_name}")
    plt.show()

    # ---------------------------------
    # Fit PCA with selected components
    # ---------------------------------
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # ---------------------------------
    # Convert back to DataFrame
    # ---------------------------------
    columns = [f"PC{i+1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(X_train_pca, columns=columns)
    X_test_pca = pd.DataFrame(X_test_pca, columns=columns)

    # ---------------------------------
    # Save in data_reduced with prefix PCA_
    # ---------------------------------
    train_output = os.path.join("data_reduced", f"PCA_{train_name}")
    test_output = os.path.join("data_reduced", f"PCA_{test_name}")

    X_train_pca.to_csv(train_output, index=False)
    X_test_pca.to_csv(test_output, index=False)

    print("Saved:", train_output)
    print("Saved:", test_output)


print("\nAll PCA datasets created successfully.")