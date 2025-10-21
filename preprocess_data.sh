#!/usr/bin/env bash
set -euo pipefail

echo "Starting preprocessing..."

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_test.csv --biomarker MUSE

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_spare_test.csv --biomarker SPARE_AD

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_spare_test.csv --biomarker SPARE_BA

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_mmse_test.csv --biomarker MMSE

python preprocess_data.py --data_file ./data/data_dl_muse_nichart_adas_test.csv --biomarker ADAS

echo "Preprocessing complete."
