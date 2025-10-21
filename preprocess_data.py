#!/usr/bin/env python3

import argparse
import pandas as pd
import pickle
import numpy as np
import os

# ------------------- Parser Setup -------------------
parser = argparse.ArgumentParser(description='Preprocessing pipeline for the input data. From the csv format to the array format for inference')
parser.add_argument("--data_file", help="Path to the data CSV file", required=True)
parser.add_argument("--biomarker", help="Biomarker to preprocess", required=True)
args = parser.parse_args()

data_file = args.data_file
biomarker = args.biomarker.upper()

# ------------------- Load Data -------------------
data = pd.read_csv(data_file)
print(f"Loaded data with shape: {data.shape}")

# ------------------- Common Preprocessing Functions -------------------
def normalize_column(col, mean, std):
    return (col - mean) / std

def binarize_sex(sex_col):
    return sex_col.map({'F': 1, 'M': 0})

def encode_diagnosis(dx_col):
    return dx_col.map({'CN': 0, 'MCI': 1, 'AD': 2})

# ------------------- Biomarker-Specific Configs -------------------
stats_dir = './statistics'

def load_stats(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if biomarker == 'MUSE':
    roi_stats = load_stats(os.path.join(stats_dir, 'dlmuse_rois_mean_std_dlmuse.pkl'))
    age_stats = load_stats(os.path.join(stats_dir, 'age_stats.pkl'))

elif biomarker == 'SPARE_AD' or biomarker == 'SPARE_BA':
    roi_stats = load_stats(os.path.join(stats_dir, 'dlmuse_rois_mean_std_spare.pkl'))
    age_stats = load_stats(os.path.join(stats_dir, 'age_mean_std_spare.pkl'))
    spare_ba_stats = load_stats(os.path.join(stats_dir, 'spare_ba_mean_std.pkl'))

elif biomarker == 'MMSE':
    roi_stats = load_stats(os.path.join(stats_dir, 'dlmuse_rois_mean_std_mmse.pkl'))
    age_stats = load_stats(os.path.join(stats_dir, 'age_mean_std_mmse.pkl'))
    spare_ba_stats = load_stats(os.path.join(stats_dir, 'spare_ba_mean_std_mmse.pkl'))
    mmse_stats = load_stats(os.path.join(stats_dir, 'mmse_mean_std.pkl'))

elif biomarker == 'ADAS':
    roi_stats = load_stats(os.path.join(stats_dir, 'dlmuse_rois_mean_std_adas.pkl'))
    age_stats = load_stats(os.path.join(stats_dir, 'age_mean_std_adas.pkl'))
    spare_ba_stats = load_stats(os.path.join(stats_dir, 'spare_ba_mean_std_adas.pkl'))
    adas_stats = load_stats(os.path.join(stats_dir, 'adas_mean_std.pkl'))

else:
    raise ValueError(f"Unsupported biomarker: {biomarker}")

# ------------------- Preprocessing Steps -------------------
# Normalize ROIs
roi_cols = [col for col in data.columns if 'DL_MUSE' in col]
data[roi_cols] = (data[roi_cols] - roi_stats['mean']) / roi_stats['std']
print(f"Normalized {len(roi_cols)} ROI columns")

# Encode Diagnosis
if 'Diagnosis' in data.columns:
    data['Diagnosis'] = encode_diagnosis(data['Diagnosis'])
    print("Encoded Diagnosis")

# Normalize Age
if 'Age' in data.columns:
    data['Age'] = normalize_column(data['Age'], age_stats['mean'], age_stats['std'])
    print("Normalized Age")

# Binarize Sex
if 'Sex' in data.columns:
    data['Sex'] = binarize_sex(data['Sex'])
    print("Binarized Sex")

# Normalize biomarker if applicable
if biomarker == 'SPARE_BA' or biomarker == 'SPARE_AD' or biomarker in ['MMSE', 'ADAS']:
    if 'SPARE_BA' in data.columns:
        data['SPARE_BA'] = normalize_column(data['SPARE_BA'], spare_ba_stats['mean'], spare_ba_stats['std'])
        print("Normalized SPARE_BA")

if biomarker == 'MMSE':
    if 'MMSE' in data.columns:
        data['MMSE'] = normalize_column(data['MMSE'], mmse_stats['mean'], mmse_stats['std'])
        print("Normalized MMSE")

if biomarker == 'ADAS':
    if 'ADAS' in data.columns:
        data['ADAS'] = normalize_column(data['ADAS'], adas_stats['mean'], adas_stats['std'])
        print("Normalized ADAS")

# ------------------- Save Preprocessed Output -------------------
output_path = data_file.replace('.csv', '_preprocessed.npy')
np.save(output_path, data.to_numpy(dtype=np.float32))
print(f"Saved preprocessed data to {output_path}")
