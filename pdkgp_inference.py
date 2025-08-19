'''
Population DKGP Model Inference
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

parser = argparse.ArgumentParser(description='Inference with trained Deep Kernel Single Task GP model')
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

print(f"Starting inference for ROI {roi_idx}")
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
train_x = checkpoint['train_x']
train_y = checkpoint['train_y']

print(f"Model loaded successfully")
print(f"Training data shape: {train_x.shape}")
print(f"Training targets shape: {train_y.shape}")

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

# Prepare test data
print("Preparing test data...")
test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

# Get corresponding test IDs for tracking
corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()

# Process test data
test_x, test_y, _, _ = process_temporal_singletask_data(
    train_x=test_x, train_y=test_y, test_x=test_x, test_y=test_y, test_ids=test_ids
)

# Move test data to GPU if available
if torch.cuda.is_available():
    test_x = test_x.cuda(gpu_id)
    test_y = test_y.cuda(gpu_id)

print(f"Test data shape: {test_x.shape}")

# Select ROI
test_y_roi = test_y[:, roi_idx]
test_y_roi = test_y_roi.squeeze()

print(f"Test Y ROI shape: {test_y_roi.shape}")

# Make predictions
print("Making predictions...")
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_preds = deepkernelmodel(test_x)
    y_preds = likelihood(f_preds)
    
    mean = y_preds.mean
    variance = y_preds.variance
    lower, upper = y_preds.confidence_region()

# Convert to numpy for processing
mean_np = mean.cpu().detach().numpy()
variance_np = variance.cpu().detach().numpy()
lower_np = lower.cpu().detach().numpy()
upper_np = upper.cpu().detach().numpy()
test_y_np = test_y_roi.cpu().detach().numpy()
test_x_np = test_x.cpu().detach().numpy()

# Calculate metrics
mae = mean_absolute_error(test_y_np, mean_np)
mse = mean_squared_error(test_y_np, mean_np)
rmse = np.sqrt(mse)
r2 = r2_score(test_y_np, mean_np)

# Calculate coverage
coverage, interval_width, mean_coverage, mean_interval_width = calc_coverage(
    predictions=mean_np, 
    groundtruth=test_y_np,
    intervals=[lower_np, upper_np]
)

coverage = coverage.numpy().astype(int)
interval_width = interval_width.numpy()
mean_coverage = mean_coverage.numpy()
mean_interval_width = mean_interval_width.numpy()

print(f"\nInference Results for ROI {roi_idx}:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coverage: {np.mean(coverage):.4f}")
print(f"Interval Width: {mean_interval_width:.4f}")

# Prepare results for CSV
results_data = {
    'subject_id': corresponding_test_ids,
    'time_point': test_x_np[:, -1].tolist(),  # Assuming last column is time
    'true_value': test_y_np.tolist(),
    'predicted_value': mean_np.tolist(),
    'variance': variance_np.tolist(),
    'lower_bound': lower_np.tolist(),
    'upper_bound': upper_np.tolist(),
    'coverage': coverage.tolist(),
    'interval_width': interval_width.tolist(),
    'roi_idx': [roi_idx] * len(corresponding_test_ids)
}

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results_data)
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print(f"Total predictions: {len(results_df)}")

# Save summary metrics
summary_metrics = {
    'roi_idx': roi_idx,
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'r2': r2,
    'coverage': float(np.mean(coverage)),
    'interval_width': float(mean_interval_width),
    'num_predictions': len(results_df),
    'num_subjects': len(set(corresponding_test_ids))
}

summary_file = output_file.replace('.csv', '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary_metrics, f, indent=2)

print(f"Summary metrics saved to: {summary_file}")
print("Inference completed successfully!")