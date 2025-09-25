# Deep Kernel Gaussian Process (DKGP) for Population Modeling

This repository contains a production-ready implementation of Deep Kernel Gaussian Process models for population-level temporal data analysis, specifically designed for medical imaging applications and biomarker trajectory prediction.

## Overview

The DKGP model combines deep neural networks with Gaussian Processes to provide:
- **Population-level modeling** of temporal trajectories
- **Uncertainty quantification** with confidence intervals
- **Deep feature learning** for complex temporal patterns
- **Production-ready inference** for new subjects
- **8-year trajectory forecasting** with 12-month intervals

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd DKGP

# Set up conda environment
./setup_environment.sh

# Activate the environment
conda activate dk
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
- PyTorch 1.13.1+
- GPyTorch 1.6.0+

## Inference Usage

### Main Inference Script

The comprehensive inference script supports all biomarker types with a single command:

```bash
# Activate environment
conda activate dk

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

**Output:** Files saved as `{subject_id}_{biomarker_name}_forecast.png`

## Performance Metrics

**Inference Performance (RTX A6000):**
- **Individual biomarkers**: ~4.9 seconds per biomarker
- **Time per subject**: 7.9-10.0 ms
- **Time per prediction**: 1.0-1.3 ms
- **Throughput**: ~1,000 predictions per second

**Memory Usage:**
- **GPU Memory**: 13-120 MB per inference
- **System RAM**: Minimal usage due to batch processing
- **Storage**: 300-400 KB per CSV file

**Scalability:**
- **Single biomarker analysis**: < 5 seconds
- **Multi-biomarker study**: < 1 minute
- **Comprehensive analysis (all ROIs)**: ~12 minutes
- **Large cohort study (1000+ subjects)**: 8-10 minutes per biomarker

## Population Demographics Analysis

```bash
# Extract population statistics and create demographic reports
python population_demographics_analysis.py
```

This script provides:
- Population demographics summary
- Training/test split statistics
- Biomarker distribution analysis
- Publication-ready demographic tables

## File Structure

```
DKGP/
├── run_inference.sh                    # Main comprehensive inference script
├── visualize_trajectories.py           # Validation and visualization script
├── plot_single_subject.py              # Single subject trajectory visualization
├── pdkgp_future_inference.py           # Core inference engine
├── population_demographics_analysis.py # Demographics analysis
├── setup_environment.sh                # Environment setup
├── models.py                           # Model definitions
├── utils.py                            # Utility functions
├── environment.yml                     # Conda environment file
├── requirements.txt                    # Python dependencies
├── data/                               # Data directory (not tracked)
├── models/                             # Trained models (not tracked)
├── models_spare/                       # SPARE models (not tracked)
├── models_cognitive/                   # Cognitive models (not tracked)
├── output/                             # Inference results (not tracked)
└── README.md                           # This file
```

## Data Requirements

### Input Data Format

Your data should be in CSV format with the following structure:

```csv
PTID,X,Y
002_S_0295,"[1.0, 2.0, 3.0, ..., 151.0]","[0.5]"
002_S_1155,"[1.1, 2.1, 3.1, ..., 151.1]","[0.6]"
```

**Required Columns:**
- `PTID`: Subject identifier (string)
- `X`: Feature array as string representation (e.g., "[1.0, 2.0, 3.0]")
- `Y`: Target value as string representation (e.g., "[0.5]")

### Test Subject Files

Create pickle files containing test subject IDs:

```python
import pickle

# Example: Create test subject IDs
test_ids = ['002_S_0295', '002_S_1155', '002_S_1261', ...]

with open('test_subject_allstudies_ids_dl_hmuse0.pkl', 'wb') as f:
    pickle.dump(test_ids, f)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: 
   - Use CPU inference: `export CUDA_VISIBLE_DEVICES=""`

2. **Model file not found**: 
   - Ensure trained models are available in the models directories
   - Check model file paths in scripts

3. **Data format error**: 
   - Verify CSV format and array string representation
   - Check pickle file format for test IDs

4. **Environment issues**:
   - Ensure conda environment is activated: `conda activate dk`
   - Verify all dependencies are installed

### Testing Installation

```bash
conda activate dk
python -c "import torch; import gpytorch; import pandas; import numpy; import sklearn; print('All dependencies installed successfully!')"
```

### Performance Optimization

1. **GPU Utilization**: Ensure CUDA is available and GPU memory is sufficient
2. **Batch Processing**: All subjects processed simultaneously per time point
3. **Memory Efficiency**: Low memory footprint allows large-scale processing

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
