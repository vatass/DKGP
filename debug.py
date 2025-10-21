import pandas as pd
import numpy as np
import seaborn as sns

# Load data
test_data = pd.read_csv('./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv')
test_ids_file = './data/train_subject_allstudies_ids_dl_muse_spare0.pkl'
# Display the first few rows of the data
print(test_data.head())

# Load test IDs
import pickle
with open(test_ids_file, "rb") as openfile:
    test_ids = []
    while True:
        try:
            test_ids.append(pickle.load(openfile))
        except EOFError:
            break
test_ids = test_ids[0]
print(f"Loaded {len(test_ids)} test subject IDs")

ptid = test_ids[0]

subject_data = test_data[test_data['PTID'] == ptid]
if len(subject_data) > 0:
    # Get the first record (baseline) for this subject
    # the baseline record would be total row for this subject. 
    baseline_record = subject_data.iloc[0]
    
    # Parse the X array (features)

    x_str = baseline_record['X']
    x_array = np.array([float(i) for i in x_str.strip('][').split(', ')])
    
    print('Input data', x_array.shape)


