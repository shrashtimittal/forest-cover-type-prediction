import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
X_train = pd.read_parquet("artifacts/X_train.parquet")
X_test  = pd.read_parquet("artifacts/X_test.parquet")
y_train = pd.read_csv("artifacts/y_train.csv").values.ravel()
y_test  = pd.read_csv("artifacts/y_test.csv").values.ravel()

# shift labels to 0–6 for XGBoost
y_train_0 = y_train - 1
y_test_0  = y_test - 1

# 2. Base model
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

# 3. Parameter grid for random search
param_dist = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# 4. Randomized search (change n_iter for more thorough search)
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter search ...")
search.fit(X_train, y_train_0)

print("\nBest parameters found:")
print(search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.4f}")

# 5. Evaluate on test set
best_xgb = search.best_estimator_
test_preds_0 = best_xgb.predict(X_test)
test_preds = test_preds_0 + 1   # shift back to 1–7
test_acc = accuracy_score(y_test, test_preds)
print(f"\nTest Accuracy of tuned model: {test_acc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, test_preds))

# 6. Save the final tuned model
joblib.dump(best_xgb, "artifacts/xgb_best_model.joblib")
print("\n✅ Tuned model saved to artifacts/xgb_best_model.joblib")
