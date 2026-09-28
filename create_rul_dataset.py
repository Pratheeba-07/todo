import pandas as pd

# Input feature dataset
INPUT_FILE = r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\IMS_1st_test_features.csv"

# Output RUL dataset
OUTPUT_FILE = r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\IMS_1st_test_RUL_dataset.csv"

# Read feature data
df = pd.read_csv(INPUT_FILE)

# Convert filename into timestamp
df["Timestamp"] = pd.to_datetime(
    df["File"],
    format="%Y.%m.%d.%H.%M.%S"
)

# The README states that Set 1 ends with failure
# of Bearing 3 and Bearing 4.
failure_time = df["Timestamp"].max()

print("Failure/reference timestamp:", failure_time)

datasets = []

# Bearing 3 → Channels 5 and 6
# Bearing 4 → Channels 7 and 8
bearing_info = {
    3: [5, 6],
    4: [7, 8]
}

for bearing, channels in bearing_info.items():

    ch1, ch2 = channels

    temp = pd.DataFrame(index=df.index)

    temp["Bearing_ID"] = bearing
    temp["Timestamp"] = df["Timestamp"]

    # Average the two sensor directions to create
    # one bearing-level value for each feature.
    temp["RMS"] = (
        df[f"B{bearing}_Ch{ch1}_RMS"] +
        df[f"B{bearing}_Ch{ch2}_RMS"]
    ) / 2

    temp["Std_Dev"] = (
        df[f"B{bearing}_Ch{ch1}_Std"] +
        df[f"B{bearing}_Ch{ch2}_Std"]
    ) / 2

    temp["Kurtosis"] = (
        df[f"B{bearing}_Ch{ch1}_Kurtosis"] +
        df[f"B{bearing}_Ch{ch2}_Kurtosis"]
    ) / 2

    temp["Skewness"] = (
        df[f"B{bearing}_Ch{ch1}_Skewness"] +
        df[f"B{bearing}_Ch{ch2}_Skewness"]
    ) / 2

    temp["Peak"] = (
        df[f"B{bearing}_Ch{ch1}_Peak"] +
        df[f"B{bearing}_Ch{ch2}_Peak"]
    ) / 2

    temp["Crest_Factor"] = (
        df[f"B{bearing}_Ch{ch1}_CrestFactor"] +
        df[f"B{bearing}_Ch{ch2}_CrestFactor"]
    ) / 2

    # RUL in hours
    temp["Actual_RUL_hours"] = (
        failure_time - temp["Timestamp"]
    ).dt.total_seconds() / 3600

    datasets.append(temp)

# Combine Bearing 3 and Bearing 4
rul_df = pd.concat(datasets, ignore_index=True)

# Sort chronologically
rul_df = rul_df.sort_values(
    ["Bearing_ID", "Timestamp"]
).reset_index(drop=True)

# Save
rul_df.to_csv(OUTPUT_FILE, index=False)

print("\nRUL dataset created successfully.")
print("Saved to:")
print(OUTPUT_FILE)

print("\nDataset shape:")
print(rul_df.shape)

print("\nFirst 5 rows:")
print(rul_df.head())

print("\nLast 5 rows:")
print(rul_df.tail())

print("\nRUL range:")
print(rul_df["Actual_RUL_hours"].min(),
      "to",
      rul_df["Actual_RUL_hours"].max(),
      "hours")