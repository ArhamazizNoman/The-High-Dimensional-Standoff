import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# 1 Load Data
# Load Data

train = pd.read_csv("data_clean/case1Data.csv")
test = pd.read_csv("data_clean/case1Data_Xnew.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nFirst rows of training data")
print(train.head())








# 2 Check Missing Values
print("\nMissing values in training data:")
missing_train = train.isnull().sum().sort_values(ascending=False)
print(missing_train[missing_train > 0])

print("\nMissing values in test data:")
missing_test = test.isnull().sum().sort_values(ascending=False)
print(missing_test[missing_test > 0])

missing = test.isnull().sum()
missing = missing[missing > 0]
print(missing)

plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()




# 3 Handle Missing Values
# Feature / Target split
X = train.drop("y", axis=1)
y = train["y"]

# Mean imputation SOLO sulle feature
X_mean_train = X.fillna(X.mean())
X_mean_test = test.fillna(test.mean())

# Se vuoi salvare il train con y
train_mean = pd.concat([y, X_mean_train], axis=1)

print("\nMissing values after cleaning:")
print(train.isnull().sum().sum())




# 4 Outlier Detection
plt.figure(figsize=(6,4))
sns.boxplot(x=train["y"])
plt.title("Outlier Detection for Y")
plt.show()

# Optional: remove extreme outliers in y
Q1 = train["y"].quantile(0.25)
Q3 = train["y"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

train = train[(train["y"] >= lower) & (train["y"] <= upper)]

print("\nShape after removing outliers:", train.shape)


# 5 Feature / Target Split

X = train.drop("y", axis=1)
y = train["y"]
print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# 6 Handle Missing Values (Median Imputation)
print("\nApplying Median Imputation...")

y = train["y"]
X = train.drop("y", axis=1)

X_median_train = X.fillna(X.median())
X_median_test = test.fillna(test.median())


# 7 Knn Imputation Methods
from sklearn.impute import KNNImputer

print("\nApplying KNN Imputation...")

# Separate X and y
y = train["y"]
X = train.drop("y", axis=1)

# Apply KNN only to X
knn_imputer = KNNImputer(n_neighbors=5)

X_knn_train = pd.DataFrame(
    knn_imputer.fit_transform(X),
    columns=X.columns
)

X_knn_test = pd.DataFrame(
    knn_imputer.transform(test),
    columns=test.columns
)

print("KNN imputation completed")



# 8 Model-Based Imputation Methods
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
print("\nApplying Model-based Imputation...")
y = train["y"]
X = train.drop("y", axis=1)
model_imputer = IterativeImputer(random_state=42)
X_model_train = pd.DataFrame(
    model_imputer.fit_transform(X),
    columns=X.columns
)
X_model_test = pd.DataFrame(
    model_imputer.transform(test),
    columns=test.columns
)


# Initialize model-based imputer
model_imputer = IterativeImputer(max_iter=10, random_state=42)

# Apply imputation
X_model = pd.DataFrame(model_imputer.fit_transform(X), columns=X.columns)
test_model = pd.DataFrame(model_imputer.transform(test), columns=test.columns)
train_model = pd.concat([y, X_model], axis=1)
print("Model-based imputation completed")








# Save
X_mean_train.to_csv("data_clean/clean_4_mean_train.csv", index=False)
X_mean_test.to_csv("data_clean/clean_4_mean_test.csv", index=False)


# Save files
X_knn_train.to_csv("data_clean/clean_2_knn_train.csv", index=False)
X_knn_test.to_csv("data_clean/clean_2_knn_test.csv", index=False)

# Save
X_median_train.to_csv("data_clean/clean_1_median_train.csv", index=False)
X_median_test.to_csv("data_clean/clean_1_median_test.csv", index=False)


X_model_train.to_csv("data_clean/clean_3_model_train.csv", index=False)
X_model_test.to_csv("data_clean/clean_3_model_test.csv", index=False)



print("All cleaned datasets saved.")