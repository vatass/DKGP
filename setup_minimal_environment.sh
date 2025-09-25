#!/bin/bash

# DKGP Minimal Environment Setup Script
# Creates a clean environment with only essential packages for inference

echo "🔧 Setting up DKGP minimal environment..."

# Method 1: Try pip virtual environment (recommended)
echo "🐍 Creating virtual environment with pip..."
python -m venv dkgp-venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully!"
    echo "📦 Installing packages..."
    source dkgp-venv/bin/activate
    pip install -r requirements_minimal.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ All packages installed successfully!"
        echo ""
        echo "🚀 To activate the environment, run:"
        echo "   source dkgp-venv/bin/activate"
        echo ""
        echo "🧪 To test the installation, run:"
        echo "   source dkgp-venv/bin/activate"
        echo "   python -c \"import torch; import gpytorch; import pandas; import numpy; import sklearn; print('✅ All dependencies installed successfully!')\""
        echo ""
        echo "📊 To run inference, use:"
        echo "   source dkgp-venv/bin/activate"
        echo "   ./run_inference.sh hippocampus_right"
        echo ""
        echo "🎉 Setup complete! You can now run DKGP inference."
        exit 0
    else
        echo "❌ Package installation failed."
        exit 1
    fi
else
    echo "❌ Virtual environment creation failed."
    echo "   Please check your Python installation and try again."
    exit 1
fi
