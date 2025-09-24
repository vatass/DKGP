#!/bin/bash

# Complete evaluation script - runs all inference and evaluation
# Usage: ./run_complete_evaluation.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

echo "=========================================="
echo "Complete Biomarker Evaluation Pipeline"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "=========================================="

# Step 1: Run inference for all ROIs
echo "Step 1: Running inference for all ROIs..."
./run_inference_all_rois.sh $HMUSE_VERSION $GPU_ID

if [ $? -ne 0 ]; then
    echo "Error: ROI inference failed!"
    exit 1
fi

# Step 2: Run inference for spare scores
echo "Step 2: Running inference for spare scores..."
./run_inference_spare_scores.sh $HMUSE_VERSION $GPU_ID

if [ $? -ne 0 ]; then
    echo "Error: Spare score inference failed!"
    exit 1
fi

# Step 3: Run inference for cognitive scores
echo "Step 3: Running inference for cognitive scores..."
./run_inference_cognitive_scores.sh $HMUSE_VERSION $GPU_ID

if [ $? -ne 0 ]; then
    echo "Error: Cognitive score inference failed!"
    exit 1
fi

# Step 4: Run comprehensive evaluation
echo "Step 4: Running comprehensive evaluation..."
python evaluate_all_biomarkers.py $HMUSE_VERSION

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo "=========================================="
echo "Complete evaluation pipeline finished!"
echo "=========================================="
echo "Results available in:"
echo "- ROI inference: ./inference_results_rois/"
echo "- Spare score inference: ./inference_results_spare/"
echo "- Cognitive score inference: ./inference_results_cognitive/"
echo "- Evaluation results: ./evaluation_results_hmuse${HMUSE_VERSION}/"
echo "==========================================" 