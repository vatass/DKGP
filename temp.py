"""
Temporary script to keep only the test subjects from the test_ids
"""

import pandas as pd

# --- DL MUSE ROIs ---
data = pd.read_csv('./data/data_dl_muse_nichart.csv')
test_ids = pd.read_pickle('./data/test_subject_allstudies_ids_dl_hmuse0.pkl')

data = data[data['PTID'].isin(test_ids)]
data.to_csv('./data/data_dl_muse_nichart_test.csv', index=False)

# --- SPARE Scores ---
data = pd.read_csv('./data/data_dl_muse_nichart_spare.csv')
test_ids = pd.read_pickle('./data/test_subject_allstudies_ids_dl_muse_spare0.pkl')

data = data[data['PTID'].isin(test_ids)]
data.to_csv('./data/data_dl_muse_nichart_spare_test.csv', index=False)

# --- MMSE Scores ---
data = pd.read_csv('./data/data_dl_muse_nichart_mmse.csv')
test_ids = pd.read_pickle('./data/test_subject_allstudies_ids_mmse0.pkl')

data = data[data['PTID'].isin(test_ids)]
data.to_csv('./data/data_dl_muse_nichart_mmse_test.csv', index=False)

# --- ADAS Scores ---
data = pd.read_csv('./data/data_dl_muse_nichart_adas.csv')
test_ids = pd.read_pickle('./data/test_subject_adni_ids_adas0.pkl')

data = data[data['PTID'].isin(test_ids)]
data.to_csv('./data/data_dl_muse_nichart_adas_test.csv', index=False)



# --- Reload the produced CSVs and print number of subjects ---
print("\n--- Verification: Number of subjects per test CSV ---")
for label, path in [
    ('DL MUSE', './data/data_dl_muse_nichart_test.csv'),
    ('SPARE', './data/data_dl_muse_nichart_spare_test.csv'),
    ('MMSE', './data/data_dl_muse_nichart_mmse_test.csv'),
    ('ADAS', './data/data_dl_muse_nichart_adas_test.csv')
]:
    df = pd.read_csv(path)
    print(f"{label}: {df['PTID'].nunique()} subjects")



## Load also the data and verify that no repeated measures are present
data = pd.read_csv('./data/data_dl_muse_nichart_test.csv')
print(data.head())
print(data['PTID'].nunique())




## Load this file ADAS_DATA_FILE="./data/subjectsamples_longclean_dlmuse_adas_adni.csv"
# and estimate the input size


