
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# 1 Load Data
# Load Data

train = pd.read_csv("data/case1Data.csv")
test = pd.read_csv("data/case1Data_Xnew.csv")

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


# 3 Handle Missing Values
train = train.fillna(train.mean(numeric_only=True))
test = test.fillna(test.mean(numeric_only=True))

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


# 6 Save Cleaned Data
X.to_csv("clean_1_X_train.csv", index=False)
y.to_csv("clean_1_y_train.csv", index=False)
test.to_csv("clean_1_X_test.csv", index=False)

print("\nCleaned datasets saved successfully.")


print(train.info())
print(train.describe())



import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()
