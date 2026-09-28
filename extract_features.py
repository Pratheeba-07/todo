import pandas as pd
import numpy as np
import os

# NASA IMS 1st test folder
DATA_FOLDER = r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\IMS\IMS\1st_test\1st_test"

# Output file
OUTPUT_FILE = r"D:\Bearing_RUL_Project raw\NASA_IMS_RAW\IMS_1st_test_features.csv"


def calculate_features(signal):

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    kurtosis = pd.Series(signal).kurtosis()
    skewness = pd.Series(signal).skew()
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms

    return rms, std, kurtosis, skewness, peak, crest_factor


# Get raw files
files = sorted([
    f for f in os.listdir(DATA_FOLDER)
    if os.path.isfile(os.path.join(DATA_FOLDER, f))
])

print("Number of files found:", len(files))

results = []

for i, filename in enumerate(files):

    file_path = os.path.join(DATA_FOLDER, filename)

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    row = {
        "File": filename
    }

    # Bearing 1 → Channels 1 and 2
    # Bearing 2 → Channels 3 and 4
    # Bearing 3 → Channels 5 and 6
    # Bearing 4 → Channels 7 and 8

    bearing_channels = {
        1: [0, 1],
        2: [2, 3],
        3: [4, 5],
        4: [6, 7]
    }

    for bearing, channels in bearing_channels.items():

        for channel in channels:

            signal = data[channel].values

            rms, std, kurtosis, skewness, peak, crest_factor = \
                calculate_features(signal)

            ch_number = channel + 1

            row[f"B{bearing}_Ch{ch_number}_RMS"] = rms
            row[f"B{bearing}_Ch{ch_number}_Std"] = std
            row[f"B{bearing}_Ch{ch_number}_Kurtosis"] = kurtosis
            row[f"B{bearing}_Ch{ch_number}_Skewness"] = skewness
            row[f"B{bearing}_Ch{ch_number}_Peak"] = peak
            row[f"B{bearing}_Ch{ch_number}_CrestFactor"] = crest_factor

    results.append(row)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(files)} files")


# Convert to DataFrame
features_df = pd.DataFrame(results)

# Save
features_df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed.")
print("Output saved to:")
print(OUTPUT_FILE)

print("\nFinal dataset shape:")
print(features_df.shape)

print("\nFirst 5 rows:")
print(features_df.head())