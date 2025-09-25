# Deep Kernel Gaussian Process (DKGP) for Population Modeling

This repository contains a production-ready implementation of Deep Kernel Gaussian Process models for population-level temporal data analysis, specifically designed for medical imaging applications and biomarker trajectory prediction.

## Overview

The DKGP model combines deep neural networks with Gaussian Processes to provide:
- **Population-level modeling** of temporal trajectories
- **Uncertainty quantification** with confidence intervals that correspond to the 95% percentile of the posterior predictive distribution
- **Deep feature learning** for complex temporal patterns
- **Production-ready inference** for new subjects
- **8-year trajectory forecasting** with 12-month intervals

## Quick Start

### 1. Environment Setup

**Option A: Automated Setup (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd DKGP

# Run the automated setup script
./setup_minimal_environment.sh

# Activate the environment
source dkgp-venv/bin/activate
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv dkgp-venv

# Activate environment
source dkgp-venv/bin/activate

# Install dependencies
pip install -r requirements_minimal.txt
```

### 2. System Requirements

**Hardware:**
- GPU: NVIDIA RTX A6000 or similar (recommended)
- CPU: Multi-core processor (tested on 2x Intel Xeon Gold 6248R)
- RAM: 16GB+ (tested on 754GB)
- Storage: High-performance SSD recommended

**Software:**
- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 2.4.1+
- GPyTorch 1.13+

### 3. Test Installation

```bash
# Activate environment
source dkgp-venv/bin/activate

# Test all dependencies
python -c "import torch; import gpytorch; import pandas; import numpy; import sklearn; print('✅ All dependencies installed successfully!')"

# Test inference
./run_inference.sh hippocampus_right
```

## Inference Usage

### Main Inference Script

The comprehensive inference script supports all biomarker types with a single command:

```bash
# Activate environment
source dkgp-venv/bin/activate

# Run inference for specific biomarkers
./run_inference.sh hippocampus_right
./run_inference.sh spare_ad
./run_inference.sh mmse

# Run inference for all biomarkers
./run_inference.sh all

# Run inference for all 145 Volume ROIs (creates single CSV)
./run_inference.sh volume_rois
```

### Supported Biomarkers

**Volume ROIs:**
- `hippocampus_right` (ROI 14) - Right Hippocampus
- `hippocampus_left` (ROI 15) - Left Hippocampus  
- `ventricle_right` (ROI 16) - Right Lateral Ventricle
- `ventricle_left` (ROI 17) - Left Lateral Ventricle
- `volume_rois` - All 145 Volume ROIs (single CSV output)

**SPARE Scores:**
- `spare_ad` (ROI 0) - SPARE-AD Score
- `spare_ba` (ROI 1) - SPARE-BA Score

**Cognitive Scores:**
- `mmse` (ROI 0) - MMSE Cognitive Score
- `adas` (ROI 0) - ADAS Cognitive Score

**Combined Options:**
- `all` - Run inference for all individual biomarkers

### Output Structure

Results are saved to `./output/` directory:

```
output/
├── hippocampus_right_output.csv          # Individual biomarker files
├── hippocampus_left_output.csv
├── lateral_ventricle_right_output.csv
├── lateral_ventricle_left_output.csv
├── spare_ad_output.csv
├── spare_ba_output.csv
├── mmse_output.csv
├── adas_output.csv
└── volume_rois_output.csv                # All 145 ROIs in single file
```

### CSV Output Format

Each CSV file contains:
- `PTID`: Subject identifier
- `time_months`: Time point (12, 24, 36, 48, 60, 72, 84, 96 months)
- `predicted_value`: DKGP prediction
- `variance`: Model uncertainty
- `lower_bound`, `upper_bound`: 95% confidence intervals
- `interval_width`: Width of confidence interval
- `roi_idx`: ROI index

For `volume_rois_output.csv`:
- `PTID`, `Time`: Subject and time columns
- `DL_MUSE_0` to `DL_MUSE_144`: All 145 ROI predictions

## Visualization and Validation

### Comprehensive Validation Script

```bash
# Activate environment
source dkgp-venv/bin/activate

# Validate inference quality and create publication-ready plots
python visualize_trajectories.py --csv_file output/hippocampus_right_output.csv

# Create specific plot types
python visualize_trajectories.py --csv_file output/spare_ad_output.csv --plot_type trajectory
python visualize_trajectories.py --csv_file output/mmse_output.csv --plot_type uncertainty
python visualize_trajectories.py --csv_file output/hippocampus_left_output.csv --plot_type summary
```

**Plot Types:**
- `trajectory` - Population mean trajectory with individual subject trajectories
- `uncertainty` - Model uncertainty validation plots
- `diversity` - Trajectory slope distribution analysis
- `summary` - Comprehensive validation summary (6-panel plot)
- `all` - All plot types (default)

### Single Subject Visualization

```bash
# Plot random subject trajectory
python plot_single_subject.py --csv_file output/hippocampus_right_output.csv

# Plot specific subject
python plot_single_subject.py --csv_file output/spare_ad_output.csv --subject_id 002_S_1155

# Custom output directory
python plot_single_subject.py --csv_file output/mmse_output.csv --output_dir ./my_plots
```

## Performance Metrics

**Inference Speed:**
- Single biomarker: ~2-3 minutes (617 subjects, 8 time points)
- All 145 ROIs: ~6-8 hours (617 subjects, 8 time points each)
- GPU acceleration: 3-5x speedup vs CPU

**Model Performance:**
- Mean Absolute Error: 0.15-0.25 (normalized units)
- Uncertainty Calibration: 95% confidence intervals
- Trajectory Forecasting: Up to 8 years (96 months)

## File Structure

```
DKGP/
├── run_inference.sh                    # Main inference script
├── pdkgp_future_inference.py          # Core inference logic
├── visualize_trajectories.py          # Validation and plotting
├── plot_single_subject.py             # Single subject visualization
├── setup_minimal_environment.sh       # Environment setup script
├── requirements_minimal.txt           # Minimal dependencies
├── environment_minimal.yml            # Conda environment (alternative)
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── data/                              # Input data (not tracked)
├── models/                            # Trained models (not tracked)
└── output/                            # Inference results (not tracked)
```

## Troubleshooting

### Common Issues

**1. CUDA/GPU Issues:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA not available, models will run on CPU (slower)
```

**2. Memory Issues:**
```bash
# For large datasets, reduce batch size in pdkgp_future_inference.py
# Or process fewer subjects at once
```

**3. Environment Issues:**
```bash
# Recreate environment
rm -rf dkgp-venv
./setup_minimal_environment.sh
```

**4. Missing Dependencies:**
```bash
# Reinstall requirements
source dkgp-venv/bin/activate
pip install -r requirements_minimal.txt --force-reinstall
```

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your system meets the requirements
3. Ensure all dependencies are installed correctly
4. Check that data and model files are in the correct locations

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dkgp2024,
  title={Deep Kernel Gaussian Process for Population Modeling},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
