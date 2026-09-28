import pandas as pd
import matplotlib.pyplot as plt

# Input RUL dataset
INPUT_FILE = r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\IMS_1st_test_RUL_dataset.csv"

# Read dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

print("Dataset shape:", df.shape)

print("\nBearing counts:")
print(df["Bearing_ID"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRUL statistics:")
print(df["Actual_RUL_hours"].describe())

# -----------------------------
# Plot RUL for Bearing 3
# -----------------------------

bearing3 = df[df["Bearing_ID"] == 3].sort_values("Timestamp")

plt.figure(figsize=(10, 5))
plt.plot(
    bearing3["Timestamp"],
    bearing3["Actual_RUL_hours"]
)

plt.xlabel("Time")
plt.ylabel("Actual RUL (hours)")
plt.title("Bearing 3 - Actual RUL vs Time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\Bearing3_RUL.png",
    dpi=300
)

plt.show()

# -----------------------------
# Plot RUL for Bearing 4
# -----------------------------

bearing4 = df[df["Bearing_ID"] == 4].sort_values("Timestamp")

plt.figure(figsize=(10, 5))
plt.plot(
    bearing4["Timestamp"],
    bearing4["Actual_RUL_hours"]
)

plt.xlabel("Time")
plt.ylabel("Actual RUL (hours)")
plt.title("Bearing 4 - Actual RUL vs Time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\Bearing4_RUL.png",
    dpi=300
)

plt.show()

print("\nRUL verification completed.")