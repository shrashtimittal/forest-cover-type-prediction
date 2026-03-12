import pandas as pd

# Load the CSV
df = pd.read_csv(r"data\train.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
