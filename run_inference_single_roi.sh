#!/bin/bash

# Inference script for single ROI
# Usage: ./run_inference_single_roi.sh <roi_idx>

# Check if ROI index is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <roi_idx>"
    echo "Example: $0 17"
    exit 1
fi

ROI_IDX=$1

# Configuration - Update these paths as needed
DATA_FILE="/home/cbica/Desktop/LongGPClustering/data1/subjectsamples_longclean_hmuse_allstudies.csv"
TEST_IDS_FILE="/home/cbica/Desktop/LongGPClustering/data1/test_subject_allstudies_ids_dl_hmuse0.pkl"
MODEL_FILE="./models/population_deep_kernel_gp_MUSE_${ROI_IDX}.pth"
OUTPUT_FILE="./inference_results/inference_results_roi_${ROI_IDX}.csv"
GPU_ID=0

# Create output directory if it doesn't exist
mkdir -p ./inference_results

echo "=========================================="
echo "Running inference for ROI ${ROI_IDX}"
echo "=========================================="
echo "Data file: ${DATA_FILE}"
echo "Test IDs file: ${TEST_IDS_FILE}"
echo "Model file: ${MODEL_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "GPU ID: ${GPU_ID}"
echo "=========================================="

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Please ensure the model has been trained for ROI ${ROI_IDX}"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

# Check if test IDs file exists
if [ ! -f "$TEST_IDS_FILE" ]; then
    echo "Error: Test IDs file not found: $TEST_IDS_FILE"
    exit 1
fi

# Run inference
echo "Starting inference..."
python pdkgp_inference.py \
    --data_file "$DATA_FILE" \
    --test_ids_file "$TEST_IDS_FILE" \
    --model_file "$MODEL_FILE" \
    --roi_idx "$ROI_IDX" \
    --output_file "$OUTPUT_FILE" \
    --gpu_id "$GPU_ID"

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "Summary saved to: ${OUTPUT_FILE%.csv}_summary.json"
    echo "=========================================="
else
    echo "=========================================="
    echo "Inference failed!"
    echo "=========================================="
    exit 1
fi 