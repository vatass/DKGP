'''
DKGP Inference
Predict biomarker values at future time points (8 years ahead, every 12 months)
'''

import pandas as pd
import numpy as np
import torch
import gpytorch
from utils import *
from models import SingleTaskDeepKernel
import argparse
import json
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(description='Future time point inference with trained DKGP model')
parser.add_argument("--data_file", help="Path to the data CSV file", required=True)
parser.add_argument("--test_ids_file", help="Path to the test IDs pickle file", required=True)
parser.add_argument("--model_file", help="Path to the trained model file", required=True)
parser.add_argument("--roi_idx", help="ROI index for inference", type=int, required=True)
parser.add_argument("--output_file", help="Path to save inference results CSV", required=True)
parser.add_argument("--gpu_id", help="GPU ID to use", type=int, default=0)

args = parser.parse_args()

# Parse arguments
gpu_id = args.gpu_id
roi_idx = args.roi_idx
data_file = args.data_file
test_ids_file = args.test_ids_file
model_file = args.model_file
output_file = args.output_file

# Define future time points (8 years = 96 months, every 12 months)
future_timepoints = [12, 24, 36, 48, 60, 72, 84, 96]

print(f"Starting future time point inference for ROI {roi_idx}")
print(f"Future time points: {future_timepoints}")
print(f"Loading data from: {data_file}")
print(f"Loading test IDs from: {test_ids_file}")
print(f"Loading model from: {model_file}")

# Load data
datasamples = pd.read_csv(data_file)
print(f"Loaded data with {len(datasamples)} samples")

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

# Load model
print("Loading trained model...")
checkpoint = torch.load(model_file, map_location=f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Extract model components
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
likelihood_state_dict = checkpoint['likelihood_state_dict']

# Check if training data is available in checkpoint
if 'train_x' in checkpoint and 'train_y' in checkpoint:
    train_x = checkpoint['train_x']
    train_y = checkpoint['train_y']
    print(f"Model loaded successfully with training data")
    print(f"Training data shape: {train_x.shape}")
    print(f"Training targets shape: {train_y.shape}")
else:
    print("Error: Training data not found in checkpoint. This model was not saved with training data.")
    print("Please retrain the model using the updated training script.")
    exit(1)

# Initialize model components
likelihood = gpytorch.likelihoods.GaussianLikelihood()
deepkernelmodel = SingleTaskDeepKernel(
    input_dim=train_x.shape[1], 
    train_x=train_x, 
    train_y=train_y, 
    likelihood=likelihood, 
    depth=[(train_x.shape[1], int(train_x.shape[1]/2))], 
    dropout=0.2, 
    activation='relu', 
    kernel_choice='RBF', 
    mean='Constant',
    pretrained=False, 
    feature_extractor=None, 
    latent_dim=int(train_x.shape[1]/2), 
    gphyper=None
)

# Load state dicts
deepkernelmodel.load_state_dict(model_state_dict)
likelihood.load_state_dict(likelihood_state_dict)

# Move to GPU if available
if torch.cuda.is_available():
    deepkernelmodel = deepkernelmodel.cuda(gpu_id)
    likelihood = likelihood.cuda(gpu_id)
    train_x = train_x.cuda(gpu_id)
    train_y = train_y.cuda(gpu_id)

# Set to evaluation mode
deepkernelmodel.eval()
likelihood.eval()

# Get baseline data (time = 0) for test subjects
print("Preparing baseline test data...")
test_data = datasamples[datasamples['PTID'].isin(test_ids)]

# Filter for baseline data (assuming time=0 is at baseline)
# We'll extract baseline features and create future time points
baseline_data = []
baseline_ptids = []


for ptid in test_ids:
    subject_data = test_data[test_data['PTID'] == ptid]
    if len(subject_data) > 0:
        # Get the first record (baseline) for this subject
        # the baseline record would be total row for this subject. 
        baseline_record = subject_data.iloc[0]
        
        # Parse the X array (features)
        x_str = baseline_record['X']
        x_array = np.array([float(i) for i in x_str.strip('][').split(', ')])
        
        baseline_data.append(x_array)
        baseline_ptids.append(ptid)

baseline_data = np.array(baseline_data)
print(f"Extracted baseline data for {len(baseline_ptids)} subjects")
print(f"Baseline feature shape: {baseline_data.shape}")


# Create future time point data
all_results = []

for time_point in future_timepoints:
    print(f"\n=== Processing time point: {time_point} months ===")
    
    # Create future data by modifying the time component (last feature)
    future_data = baseline_data.copy()
    future_data[:, -1] = time_point  # Set time to future time point
    
    # Convert to tensor
    future_tensor = torch.Tensor(future_data)
    if torch.cuda.is_available():
        future_tensor = future_tensor.cuda(gpu_id)
    
    print(f"Future data shape: {future_tensor.shape}")
    
    # Make predictions
    print(f"Making predictions for time point {time_point}...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_preds = deepkernelmodel(future_tensor)
        y_preds = likelihood(f_preds)
        
        mean = y_preds.mean
        variance = y_preds.variance
        lower, upper = y_preds.confidence_region()
    
    # Convert to numpy for processing
    mean_np = mean.cpu().detach().numpy()
    variance_np = variance.cpu().detach().numpy()
    lower_np = lower.cpu().detach().numpy()
    upper_np = upper.cpu().detach().numpy()
    
    # Create results for this time point
    for i, ptid in enumerate(baseline_ptids):
        result = {
            'PTID': ptid,
            'time_months': time_point,
            'predicted_value': mean_np[i],
            'variance': variance_np[i],
            'lower_bound': lower_np[i],
            'upper_bound': upper_np[i],
            'interval_width': upper_np[i] - lower_np[i],
            'roi_idx': roi_idx
        }
        all_results.append(result)
    
    print(f"Completed predictions for {len(baseline_ptids)} subjects at {time_point} months")

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Add summary statistics
print(f"\n=== Summary Statistics ===")
print(f"Total predictions: {len(results_df)}")
print(f"Subjects: {len(baseline_ptids)}")
print(f"Time points: {len(future_timepoints)}")
print(f"ROI: {roi_idx}")

for time_point in future_timepoints:
    tp_data = results_df[results_df['time_months'] == time_point]
    mean_pred = tp_data['predicted_value'].mean()
    std_pred = tp_data['predicted_value'].std()
    mean_uncertainty = tp_data['interval_width'].mean()
    print(f"Time {time_point}m: Mean={mean_pred:.4f} ± {std_pred:.4f}, Uncertainty={mean_uncertainty:.4f}")

# Save results
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Save summary metrics
summary_metrics = {
    'roi_idx': roi_idx,
    'n_subjects': len(baseline_ptids),
    'n_timepoints': len(future_timepoints),
    'future_timepoints': future_timepoints,
    'n_total_predictions': len(results_df),
    'mean_prediction_by_timepoint': {
        str(tp): float(results_df[results_df['time_months'] == tp]['predicted_value'].mean())
        for tp in future_timepoints
    },
    'mean_uncertainty_by_timepoint': {
        str(tp): float(results_df[results_df['time_months'] == tp]['interval_width'].mean())
        for tp in future_timepoints
    }
}

summary_file = output_file.replace('.csv', '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary_metrics, f, indent=2)

print(f"Summary metrics saved to: {summary_file}")
print("Future time point inference completed successfully!")
