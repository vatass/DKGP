#!/usr/bin/env python3
"""
Preprocessing pipeline for biomarker inference.

This script normalizes imaging, demographic, and clinical features using
precomputed statistics stored in .pkl files. It outputs a CSV file
with preserved column names and order, ready for inference.
"""

import argparse
import pandas as pd
import pickle
import numpy as np
import os

# ------------------- Parser -------------------
parser = argparse.ArgumentParser(description='Preprocess data for biomarker inference')
parser.add_argument("--data_file", required=True, help="Path to the data CSV file")
parser.add_argument("--biomarker", required=True, help="Biomarker to preprocess (MUSE, SPARE_AD, SPARE_BA, MMSE, ADAS)")
args = parser.parse_args()

data_file = args.data_file
biomarker = args.biomarker.upper()

# ------------------- Load Data -------------------
data = pd.read_csv(data_file)
# remove any Unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

print(f"\n✅ Loaded data: {data.shape[0]} subjects × {data.shape[1]} columns")


# ------------------- Helpers -------------------
def normalize_column(col, mean, std):
    return (col - mean) / std

def binarize_sex(sex_col):
    return sex_col.map({'F': 1, 'M': 0})

def encode_diagnosis(dx_col):
    return dx_col.map({'CN': 0, 'MCI': 1, 'AD': 2})

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def unpack_stats(obj, name):
    """Unpack stats whether stored as dict or list."""
    if isinstance(obj, dict):
        return obj['mean'], obj['std']
    elif isinstance(obj, list) and len(obj) == 2:
        return obj[0], obj[1]
    else:
        raise ValueError(f"❌ Unexpected format for {name} stats.")

# ------------------- Biomarker Config -------------------
stats_dir = './statistics'

if biomarker == 'MUSE':
    roi_stats_path = os.path.join(stats_dir, 'dlmuse_rois_mean_std.pkl')
    age_stats_path = os.path.join(stats_dir, 'age_stats_dlmuse.pkl')

elif biomarker in ['SPARE_AD', 'SPARE_BA']:
    roi_stats_path = os.path.join(stats_dir, 'dlmuse_rois_mean_std_spare.pkl')
    age_stats_path = os.path.join(stats_dir, 'age_mean_std_spare.pkl')
    spare_ba_stats_path = os.path.join(stats_dir, 'spare_ba_mean_std.pkl')

elif biomarker == 'MMSE':
    roi_stats_path = os.path.join(stats_dir, 'dlmuse_rois_mean_std_mmse.pkl')
    age_stats_path = os.path.join(stats_dir, 'age_mean_std_mmse.pkl')
    spare_ba_stats_path = os.path.join(stats_dir, 'spare_ba_mean_std_mmse.pkl')
    mmse_stats_path = os.path.join(stats_dir, 'mmse_mean_std.pkl')

elif biomarker == 'ADAS':
    roi_stats_path = os.path.join(stats_dir, 'dlmuse_rois_mean_std_adas.pkl')
    age_stats_path = os.path.join(stats_dir, 'age_mean_std_adas.pkl')
    spare_ba_stats_path = os.path.join(stats_dir, 'spare_ba_mean_std_adas.pkl')
    adas_stats_path = os.path.join(stats_dir, 'adas_mean_std.pkl')

else:
    raise ValueError(f"❌ Unsupported biomarker: {biomarker}")

# ------------------- Load Stats -------------------
roi_stats = load_pickle(roi_stats_path)
roi_mean = roi_stats['mean']
roi_std = roi_stats['std']
print(f"📊 Loaded ROI stats with {len(roi_mean)} entries")

age_stats = load_pickle(age_stats_path)
age_mean, age_std = unpack_stats(age_stats, "Age")

# Track every column that is successfully normalized / encoded
normalized_cols = set()

# ------------------- ROI Normalization -------------------
roi_cols = [col for col in data.columns if col in roi_mean]
print(f"🔍 Found {len(roi_cols)} ROI columns to normalize")

if len(roi_cols) == 0:
    print(f"⚠️  No ROI columns matched the stats keys — check that column names align.")
    print(f"    Data columns (first 10): {list(data.columns[:10])}")
    print(f"    Stats keys (first 10):   {list(roi_mean.keys())[:10]}")
else:
    for col in roi_cols:
        data[col] = (data[col] - roi_mean[col]) / roi_std[col]
    normalized_cols.update(roi_cols)
    print(f"✅ Normalized {len(roi_cols)} ROI columns")

# ------------------- Demographic Normalization -------------------
if 'Diagnosis_nearest_2.0' in data.columns:
    data['Diagnosis_nearest_2.0'] = encode_diagnosis(data['Diagnosis_nearest_2.0'])
    normalized_cols.add('Diagnosis_nearest_2.0')
    print("✅ Encoded Diagnosis (CN=0, MCI=1, AD=2)")

if 'Sex' in data.columns:
    data['Sex'] = binarize_sex(data['Sex'])
    normalized_cols.add('Sex')
    print("✅ Binarized Sex (F=1, M=0)")

if 'Age' in data.columns:
    data['Age'] = normalize_column(data['Age'], age_mean, age_std)
    normalized_cols.add('Age')
    print("✅ Normalized Age")

# ------------------- Biomarker-specific Normalization -------------------
if biomarker in ['SPARE_AD', 'SPARE_BA', 'MMSE', 'ADAS']:
    spare_ba_stats = load_pickle(spare_ba_stats_path)
    spare_ba_mean, spare_ba_std = unpack_stats(spare_ba_stats, "SPARE_BA")
    if 'SPARE_BA' in data.columns:
        data['SPARE_BA'] = normalize_column(data['SPARE_BA'], spare_ba_mean, spare_ba_std)
        normalized_cols.add('SPARE_BA')
        print("✅ Normalized SPARE_BA")
    else:
        print(f"⚠️  SPARE_BA column not found in data — skipping normalization. Available columns: {list(data.columns)}")

if biomarker == 'MMSE':
    if 'MMSE_nearest_2.0' in data.columns:
        mmse_stats = load_pickle(mmse_stats_path)
        mmse_mean, mmse_std = unpack_stats(mmse_stats, "MMSE")
        data['MMSE_nearest_2.0'] = normalize_column(data['MMSE_nearest_2.0'], mmse_mean, mmse_std)
        normalized_cols.add('MMSE_nearest_2.0')
        print("✅ Normalized MMSE_nearest_2.0")
    else:
        print(f"⚠️  MMSE_nearest_2.0 column not found in data — skipping normalization. Available columns: {list(data.columns)}")

if biomarker == 'ADAS':
    if 'ADAS_COG_13' in data.columns:
        adas_stats = load_pickle(adas_stats_path)
        adas_mean, adas_std = unpack_stats(adas_stats, "ADAS")
        data['ADAS_COG_13'] = normalize_column(data['ADAS_COG_13'], adas_mean, adas_std)
        normalized_cols.add('ADAS_COG_13')
        print("✅ Normalized ADAS_COG_13")
    else:
        print(f"⚠️  ADAS_COG_13 column not found in data — skipping normalization. Available columns: {list(data.columns)}")

# ------------------- Pre-save Validation -------------------
# Build the set of columns that must have been normalized for this biomarker.
# Only include a column in the required set if it actually exists in the data,
# so we get a hard failure rather than a confusing "missing column" message.
required_cols = set()
if roi_cols:                              # at least one ROI was found → all must be done
    required_cols.update(roi_cols)
if 'Age' in data.columns:
    required_cols.add('Age')
if biomarker in ['SPARE_AD', 'SPARE_BA', 'MMSE', 'ADAS']:
    if 'SPARE_BA' in data.columns:
        required_cols.add('SPARE_BA')
if biomarker == 'MMSE':
    if 'MMSE_nearest_2.0' in data.columns:
        required_cols.add('MMSE_nearest_2.0')
if biomarker == 'ADAS':
    if 'ADAS_COG_13' in data.columns:
        required_cols.add('ADAS_COG_13')

missing_normalization = required_cols - normalized_cols
assert len(missing_normalization) == 0, (
    f"❌ Pre-save check failed: the following columns were expected to be "
    f"normalized but were not: {sorted(missing_normalization)}"
)
print("✅ Pre-save validation passed — all required columns are normalized.")

# ------------------- Save Output -------------------
output_path = data_file.replace('.csv', f'_preprocessed.csv')
data.to_csv(output_path, index=False)

print(f"\n💾 Saved preprocessed CSV with preserved column order to: {output_path}")
print("✅ Preprocessing complete.\n")


