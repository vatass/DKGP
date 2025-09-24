# Deep Kernel Gaussian Process (DKGP) for Population Modeling

This repository contains a production-ready implementation of Deep Kernel Gaussian Process models for population-level temporal data analysis, specifically designed for medical imaging applications.

## Overview

The DKGP model combines deep neural networks with Gaussian Processes to provide:
- **Population-level modeling** of temporal trajectories
- **Uncertainty quantification** with confidence intervals
- **Deep feature learning** for complex temporal patterns
- **Production-ready inference** for new subjects

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd DKGP

# Set up conda environment
./setup_environment.sh

# Activate the environment
conda activate dkgp
```

### 2. Data Preparation

Your data should be in CSV format with the following columns:
- `PTID`: Subject identifier
- `X`: Temporal features (as string representation of arrays)
- `Y`: Target values (as string representation of arrays)

Example data structure:
```csv
PTID,X,Y
subject_001,"[1.0, 2.0, 3.0]","[0.5, 0.6, 0.7]"
subject_002,"[1.1, 2.1, 3.1]","[0.4, 0.5, 0.6]"
```

### 3. Train/Test Split

Create pickle files containing subject IDs for training and testing:
```python
import pickle

# Example: Create train/test split
train_ids = ['subject_001', 'subject_002', ...]
test_ids = ['subject_003', 'subject_004', ...]

with open('train_ids.pkl', 'wb') as f:
    pickle.dump(train_ids, f)

with open('test_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)
```

## Usage

### Training

Train a DKGP model for a specific ROI:

```bash
python pdkgp_training.py \
    --data_file /path/to/data.csv \
    --train_ids_file /path/to/train_ids.pkl \
    --test_ids_file /path/to/test_ids.pkl \
    --roi_idx 17 \
    --output_dir ./models \
    --gpu_id 0
```

**Parameters:**
- `--data_file`: Path to your data CSV file
- `--train_ids_file`: Path to train subject IDs pickle file
- `--test_ids_file`: Path to test subject IDs pickle file
- `--roi_idx`: ROI index to train on (integer)
- `--output_dir`: Directory to save model outputs (default: ./output)
- `--gpu_id`: GPU ID to use (default: 0)

### Inference

Run inference on test subjects:

```bash
python pdkgp_inference.py \
    --data_file /path/to/data.csv \
    --test_ids_file /path/to/test_ids.pkl \
    --model_file ./models/population_deep_kernel_gp_17.pth \
    --roi_idx 17 \
    --output_file ./results/inference_results_roi_17.csv \
    --gpu_id 0
```

**Parameters:**
- `--data_file`: Path to your data CSV file
- `--test_ids_file`: Path to test subject IDs pickle file
- `--model_file`: Path to trained model file
- `--roi_idx`: ROI index for inference
- `--output_file`: Path to save inference results CSV
- `--gpu_id`: GPU ID to use (default: 0)

### Using the Shell Script

For convenience, use the provided shell script for inference:

```bash
./run_inference_single_roi.sh 17
```

This will automatically:
- Load the model for ROI 17
- Run inference on test subjects
- Save results to `./inference_results/`

## Output Files

### Training Outputs
- `population_deep_kernel_gp_{roi_idx}.pth`: Trained model with weights and training data
- `results_roi_{roi_idx}.json`: Training metrics and summary

### Inference Outputs
- `inference_results_roi_{roi_idx}.csv`: Detailed trajectory predictions
  - `subject_id`: Subject identifier
  - `time_point`: Time point for prediction
  - `true_value`: Ground truth value
  - `predicted_value`: Model prediction
  - `variance`: Prediction uncertainty
  - `lower_bound`, `upper_bound`: Confidence intervals
  - `coverage`: Whether true value falls in confidence interval
  - `interval_width`: Width of confidence interval
- `inference_results_roi_{roi_idx}_summary.json`: Summary metrics

## Model Architecture

The DKGP model uses:
- **Feature Extractor**: Deep neural network with configurable depth
- **Kernel**: RBF kernel with automatic relevance determination (ARD)
- **Mean Function**: Linear mean function
- **Likelihood**: Gaussian likelihood for regression

## Dependencies

Core dependencies (automatically installed):
- `torch>=1.10.0`: PyTorch for deep learning
- `gpytorch>=1.6.0`: Gaussian Processes in PyTorch
- `pandas>=1.1.0`: Data manipulation
- `numpy>=1.19.0`: Numerical computing
- `scikit-learn>=0.24.0`: Machine learning utilities
- `scipy>=1.5.0`: Scientific computing

## File Structure

```
DKGP/
├── pdkgp_training.py          # Training script
├── pdkgp_inference.py         # Inference script
├── run_inference_single_roi.sh # Inference shell script
├── setup_environment.sh       # Environment setup script
├── environment.yml            # Conda environment file
├── requirements.txt           # Python dependencies
├── models.py                  # Model definitions
├── utils.py                   # Utility functions
├── README.md                  # This file
├── output/                    # Training outputs (created automatically)
└── inference_results/         # Inference outputs (created automatically)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model file not found**: Ensure training completed successfully
3. **Data format error**: Check CSV format and array string representation

### Testing Installation

```bash
conda activate dkgp
python -c "import torch; import gpytorch; import pandas; import numpy; import sklearn; print('All dependencies installed successfully!')"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dkgp2024,
  title={Deep Kernel Gaussian Processes for Population Modeling},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Add your license information here]

## Contact

[Add your contact information here] 