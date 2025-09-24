#!/bin/bash

echo "=========================================="
echo "Setting up DKGP (Deep Kernel Gaussian Process) Environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "dkgp"; then
    echo "Environment 'dkgp' already exists. Removing it..."
    conda env remove -n dkgp -y
fi

echo "Creating conda environment 'dkgp'..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Environment created successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the environment, run:"
    echo "conda activate dkgp"
    echo ""
    echo "To test the installation, run:"
    echo "python -c \"import torch; import gpytorch; import pandas; import numpy; import sklearn; print('All dependencies installed successfully!')\""
    echo ""
    echo "To run training:"
    echo "python pdkgp_training.py --help"
    echo ""
    echo "To run inference:"
    echo "python pdkgp_inference.py --help"
    echo "=========================================="
else
    echo "=========================================="
    echo "Error: Failed to create environment"
    echo "=========================================="
    exit 1
fi 