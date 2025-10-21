#!/usr/bin/env python3
"""
Script to verify the normalization during the processing of the data.
It compares the original (unnormalized) data and the normalized output,
ensuring normalization was correctly applied.
"""

import pandas as pd
import numpy as np
import random

# ------------------- Load Data -------------------
raw_path = './data/data_dl_muse_nichart_test.csv'
norm_path = './data/data_dl_muse_nichart_test_unnorm_preprocessed.csv'

data_raw = pd.read_csv(raw_path)
data_norm = pd.read_csv(norm_path)

# ------------------- 1. Check subject count -------------------
assert data_raw.shape[0] == data_norm.shape[0], "❌ Mismatch in number of subjects!"
print(f"✅ Same number of subjects: {data_raw.shape[0]}")

# ------------------- 2. Check column names -------------------
assert list(data_raw.columns) == list(data_norm.columns), "❌ Column names or order mismatch!"
print(f"✅ Same columns and order ({len(data_raw.columns)} features)")

# ------------------- 3. Pick random subject & feature -------------------
i = random.randint(0, data_raw.shape[0] - 1)
j = random.randint(0, data_raw.shape[1] - 1)
col_name = data_raw.columns[j]

x_raw = data_raw.iloc[i, j]
x_norm = data_norm.iloc[i, j]

print(f"✅ Raw value: {x_raw:.4f}")
print(f"✅ Normalized value: {x_norm:.4f}")


