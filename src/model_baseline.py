import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load preprocessed data
X_train = pd.read_parquet("artifacts/X_train.parquet")
X_test  = pd.read_parquet("artifacts/X_test.parquet")
y_train = pd.read_csv("artifacts/y_train.csv").values.ravel()
y_test  = pd.read_csv("artifacts/y_test.csv").values.ravel()

# 2. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,    # number of trees
    max_depth=None,      # let trees expand fully
    random_state=42,
    n_jobs=-1            # use all CPU cores
)
rf.fit(X_train, y_train)

# 3. Evaluate
train_preds = rf.predict(X_train)
test_preds  = rf.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc  = accuracy_score(y_test, test_preds)

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing  Accuracy : {test_acc:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, test_preds))

# 4. Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(np.unique(y_test)),
            yticklabels=sorted(np.unique(y_test)))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# 5. Feature Importance (optional top 10)
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=False).head(10)
print("\nTop 10 important features:\n", top10)

plt.figure(figsize=(8,4))
top10.sort_values().plot(kind="barh", color="orange")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()
