import pandas as pd


# ==========================================
# 1. LOAD DATA
# ==========================================

dataset_file = r"NASA_IMS_RAW\IMS_1st_test_RUL_dataset.csv"
prediction_file = "AI_Bearing_Predictions.csv"

df = pd.read_csv(dataset_file)
predictions = pd.read_csv(prediction_file)

print("Dataset loaded.")
print("Prediction file loaded.")
print()


# ==========================================
# 2. CONVERT TIMESTAMP
# ==========================================

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
predictions["Timestamp"] = pd.to_datetime(predictions["Timestamp"])


# ==========================================
# 3. MERGE FEATURES WITH PREDICTIONS
# ==========================================

feature_columns = [
    "RMS",
    "Std_Dev",
    "Kurtosis",
    "Skewness",
    "Peak",
    "Crest_Factor"
]

merge_columns = [
    "Bearing_ID",
    "Timestamp"
]

final_df = predictions.merge(
    df[merge_columns + feature_columns],
    on=merge_columns,
    how="left"
)


# ==========================================
# 4. HEALTH STATUS
# ==========================================
#
# These are PROJECT-DEFINED thresholds.
# They are not universal industrial standards.
#
# Predicted RUL > 500 h  -> Healthy
# Predicted RUL 100-500 h -> Warning
# Predicted RUL < 100 h   -> Critical
#

def classify_health(rul):

    if rul > 500:
        return "Healthy"

    elif rul >= 100:
        return "Warning"

    else:
        return "Critical"


final_df["Health_Status"] = (
    final_df["Predicted_RUL_hours"]
    .apply(classify_health)
)


# ==========================================
# 5. MAINTENANCE RECOMMENDATION
# ==========================================

def maintenance_action(status):

    if status == "Healthy":
        return "Continue operation and routine monitoring"

    elif status == "Warning":
        return "Schedule bearing inspection and increase monitoring"

    else:
        return "Plan bearing replacement and immediate inspection"


final_df["Recommended_Action"] = (
    final_df["Health_Status"]
    .apply(maintenance_action)
)


# ==========================================
# 6. ARRANGE COLUMNS
# ==========================================

final_columns = [
    "Bearing_ID",
    "Timestamp",
    "RMS",
    "Std_Dev",
    "Kurtosis",
    "Skewness",
    "Peak",
    "Crest_Factor",
    "Actual_RUL_hours",
    "Predicted_RUL_hours",
    "Health_Status",
    "Recommended_Action"
]

final_df = final_df[final_columns]


# ==========================================
# 7. SAVE FINAL AI OUTPUT
# ==========================================

output_file = "AI_Bearing_Prediction_Final.csv"

final_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 8. DISPLAY SUMMARY
# ==========================================

print("FINAL AI OUTPUT CREATED")
print("=======================")

print("Total predictions:", len(final_df))
print()

print("Health Status Distribution:")
print(
    final_df["Health_Status"]
    .value_counts()
)

print()

print("Sample output:")
print(
    final_df.head(10).to_string(index=False)
)

print()

print("Final file saved as:")
print(output_file)

print()
print("AI OUTPUT PIPELINE COMPLETED.")