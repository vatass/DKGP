#!/usr/bin/env bash
set -euo pipefail

echo "Starting preprocessing..."

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_test_unnorm.csv --biomarker MUSE

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_spare_test_unnorm.csv --biomarker SPARE_AD

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_spare_test_unnorm.csv --biomarker SPARE_BA

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_mmse_test_unnorm.csv --biomarker MMSE

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_adas_test_unnorm.csv --biomarker ADAS

echo "Preprocessing complete."
