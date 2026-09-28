import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = r"NASA_IMS_RAW\IMS_1st_test_RUL_dataset.csv"

df = pd.read_csv(file_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print()


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

features = [
    "RMS",
    "Std_Dev",
    "Kurtosis",
    "Skewness",
    "Peak",
    "Crest_Factor"
]

target = "Actual_RUL_hours"

X = df[features]
y = df[target]

print("Features:")
for f in features:
    print("-", f)

print()
print("Target:", target)
print()


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()


# ==========================================
# 4. GRADIENT BOOSTING MODEL
# ==========================================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    loss="huber"
)

print("Training Gradient Boosting model...")

model.fit(X_train, y_train)

print("Training completed.")
print()


# ==========================================
# 5. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. PERFORMANCE
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print("=================")
print(f"MAE  : {mae:.2f} hours")
print(f"RMSE : {rmse:.2f} hours")
print(f"R²   : {r2:.4f}")
print()


# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("FEATURE IMPORTANCE")
print("==================")
print(importance)
print()


# ==========================================
# 8. FEATURE IMPORTANCE GRAPH
# ==========================================

plt.figure(figsize=(9, 6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Bearing RUL Prediction - Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "Feature_Importance.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. ACTUAL VS PREDICTED
# ==========================================

comparison = pd.DataFrame({
    "Actual_RUL": y_test.values,
    "Predicted_RUL": y_pred
})

comparison = comparison.sort_values(
    "Actual_RUL"
).reset_index(drop=True)

plt.figure(figsize=(10, 6))

plt.plot(
    comparison["Actual_RUL"],
    label="Actual RUL"
)

plt.plot(
    comparison["Predicted_RUL"],
    label="Predicted RUL"
)

plt.xlabel("Test Sample")
plt.ylabel("RUL (hours)")
plt.title("Actual vs Predicted Bearing RUL")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Actual_vs_Predicted_RUL.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "bearing_rul_model.joblib"
)

print("Model saved as:")
print("bearing_rul_model.joblib")
print()


# ==========================================
# 11. SAVE PREDICTIONS
# ==========================================

results = df.loc[
    X_test.index,
    ["Bearing_ID", "Timestamp"]
].copy()

results["Actual_RUL_hours"] = y_test.values
results["Predicted_RUL_hours"] = y_pred

results = results.sort_values(
    ["Bearing_ID", "Timestamp"]
)

results.to_csv(
    "AI_Bearing_Predictions.csv",
    index=False
)

print("Prediction file saved as:")
print("AI_Bearing_Predictions.csv")
print()

print("AI MODEL PIPELINE COMPLETED.")