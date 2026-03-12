import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load preprocessed data
print("Loading preprocessed datasets ...")
X_train = pd.read_parquet("artifacts/X_train.parquet")
X_test  = pd.read_parquet("artifacts/X_test.parquet")
y_train = pd.read_csv("artifacts/y_train.csv").values.ravel()
y_test  = pd.read_csv("artifacts/y_test.csv").values.ravel()
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 2. Shift labels to 0–6 (XGBoost requirement)
y_train_0 = y_train - 1
y_test_0  = y_test - 1

# 3. Train XGBoost model
print("\nTraining XGBoost model ...")
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=7,           # now matches 0–6
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train_0)

# 4. Predictions (convert back to 1–7)
train_preds_0 = xgb.predict(X_train)
test_preds_0  = xgb.predict(X_test)
train_preds = train_preds_0 + 1
test_preds  = test_preds_0 + 1

# 5. Metrics
train_acc = accuracy_score(y_train, train_preds)
test_acc  = accuracy_score(y_test, test_preds)

print(f"\nTraining Accuracy : {train_acc:.4f}")
print(f"Testing  Accuracy : {test_acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, test_preds))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=sorted(np.unique(y_test)),
            yticklabels=sorted(np.unique(y_test)))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Feature Importances
importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=False).head(10)
print("\nTop 10 important features:")
print(top10)

plt.figure(figsize=(8,4))
top10.sort_values().plot(kind="barh", color="purple")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

print("\n✅ Script finished successfully.")
