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

# ------------------- ROI Normalization -------------------
roi_cols = [col for col in data.columns if col in roi_mean]
print(f"🔍 Found {len(roi_cols)} ROI columns to normalize")

for col in roi_cols:
    data[col] = (data[col] - roi_mean[col]) / roi_std[col]

print(f"✅ Normalized {len(roi_cols)} ROI columns")

# ------------------- Demographic Normalization -------------------
if 'Diagnosis_nearest_2.0' in data.columns:
    data['Diagnosis_nearest_2.0'] = encode_diagnosis(data['Diagnosis_nearest_2.0'])
    print("✅ Encoded Diagnosis (CN=0, MCI=1, AD=2)")

if 'Sex' in data.columns:
    data['Sex'] = binarize_sex(data['Sex'])
    print("✅ Binarized Sex (F=1, M=0)")

if 'Age' in data.columns:
    data['Age'] = normalize_column(data['Age'], age_mean, age_std)
    print("✅ Normalized Age")

# ------------------- Biomarker-specific Normalization -------------------
if biomarker in ['SPARE_AD', 'SPARE_BA', 'MMSE', 'ADAS']:
    spare_ba_stats = load_pickle(spare_ba_stats_path)
    spare_ba_mean, spare_ba_std = unpack_stats(spare_ba_stats, "SPARE_BA")
    if 'SPARE_BA' in data.columns:
        data['SPARE_BA'] = normalize_column(data['SPARE_BA'], spare_ba_mean, spare_ba_std)
        print("✅ Normalized SPARE_BA")

if biomarker == 'MMSE' and 'MMSE' in data.columns:
    mmse_stats = load_pickle(mmse_stats_path)
    mmse_mean, mmse_std = unpack_stats(mmse_stats, "MMSE")
    data['MMSE'] = normalize_column(data['MMSE'], mmse_mean, mmse_std)
    print("✅ Normalized MMSE")

if biomarker == 'ADAS' and 'ADAS' in data.columns:
    adas_stats = load_pickle(adas_stats_path)
    adas_mean, adas_std = unpack_stats(adas_stats, "ADAS")
    data['ADAS'] = normalize_column(data['ADAS'], adas_mean, adas_std)
    print("✅ Normalized ADAS")

# ------------------- Save Output -------------------
output_path = data_file.replace('.csv', f'_preprocessed.csv')
data.to_csv(output_path, index=False)

print(f"\n💾 Saved preprocessed CSV with preserved column order to: {output_path}")
print("✅ Preprocessing complete.\n")


