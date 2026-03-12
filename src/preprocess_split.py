import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv(r"data\train.csv")

# 2. Drop the Id column (not a predictive feature)
df = df.drop(columns=['Id'])

# 3. Separate target and features
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']

# 4. Identify numeric columns to scale
numeric_cols = [
    'Elevation','Aspect','Slope',
    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

# Binary columns (Wilderness & Soil) stay as 0/1
binary_cols = [c for c in X.columns if c not in numeric_cols]

# 5. Scale only numeric features
scaler = StandardScaler()
X_scaled_num = pd.DataFrame(
    scaler.fit_transform(X[numeric_cols]),
    columns=numeric_cols
)
X_scaled = pd.concat([X_scaled_num, X[binary_cols].reset_index(drop=True)], axis=1)

# 6. Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape :", X_train.shape, y_train.shape)
print("Test  shape :", X_test.shape,  y_test.shape)

# 7. Save preprocessed data (optional, for reuse)
import os
os.makedirs("artifacts", exist_ok=True)
X_train.to_parquet("artifacts/X_train.parquet", index=False)
X_test.to_parquet("artifacts/X_test.parquet", index=False)
y_train.to_csv("artifacts/y_train.csv", index=False)
y_test.to_csv("artifacts/y_test.csv", index=False)
print("Preprocessed datasets saved to artifacts/")
