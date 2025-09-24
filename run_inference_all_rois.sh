#!/bin/bash

# Inference script for all 145 ROIs
# Usage: ./run_inference_all_rois.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

# Configuration
DATA_FILE="./data/subjectsamples_longclean_dl_hmuse_allstudies.csv"
TEST_IDS_FILE="./data/test_subject_allstudies_ids_dl_hmuse${HMUSE_VERSION}.pkl"
MODELS_DIR="./models"
OUTPUT_DIR="./inference_results_rois"
LOG_DIR="./logs_inference_rois"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Total number of ROIs
TOTAL_ROIS=145
START_ROI=0
END_ROI=144

echo "=========================================="
echo "DKGP Inference for All ROIs"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "Data file: ${DATA_FILE}"
echo "Test IDs file: ${TEST_IDS_FILE}"
echo "Models directory: ${MODELS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "ROI range: ${START_ROI} to ${END_ROI}"
echo "=========================================="

# Check if required files exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

if [ ! -f "$TEST_IDS_FILE" ]; then
    echo "Error: Test IDs file not found: $TEST_IDS_FILE"
    exit 1
fi

# Function to run inference for all ROIs
run_inference_all_rois() {
    local completed=0
    local failed=0
    
    echo "Starting inference for all ROIs..."
    
    for roi_idx in $(seq $START_ROI $END_ROI); do
        echo "=========================================="
        echo "Running inference for ROI ${roi_idx}"
        echo "=========================================="
        
        # Check if model exists
        local model_file="$MODELS_DIR/population_deep_kernel_gp_MUSE_${roi_idx}.pth"
        local output_file="$OUTPUT_DIR/inference_results_roi_${roi_idx}.csv"
        local log_file="$LOG_DIR/inference_roi_${roi_idx}.log"
        
        if [ ! -f "$model_file" ]; then
            echo "ROI ${roi_idx}: Model not found, skipping..."
            failed=$((failed + 1))
            continue
        fi
        
        if [ -f "$output_file" ]; then
            echo "ROI ${roi_idx}: Results already exist, skipping..."
            completed=$((completed + 1))
            continue
        fi
        
        echo "Starting inference for ROI ${roi_idx} at $(date)"
        
        # Run inference
        CUDA_VISIBLE_DEVICES=$GPU_ID python pdkgp_inference.py \
            --data_file "$DATA_FILE" \
            --test_ids_file "$TEST_IDS_FILE" \
            --model_file "$model_file" \
            --roi_idx "$roi_idx" \
            --output_file "$output_file" \
            --gpu_id "$GPU_ID" \
            > "$log_file" 2>&1
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "ROI ${roi_idx}: Inference completed successfully"
            completed=$((completed + 1))
            
            # Extract key metrics from summary file
            local summary_file="$OUTPUT_DIR/inference_results_roi_${roi_idx}_summary.json"
            if [ -f "$summary_file" ]; then
                local mae=$(grep -o '"mae": [0-9.]*' "$summary_file" | cut -d' ' -f2)
                local r2=$(grep -o '"r2": [0-9.-]*' "$summary_file" | cut -d' ' -f2)
                echo "ROI ${roi_idx}: MAE=${mae}, R²=${r2}"
            fi
        else
            echo "ROI ${roi_idx}: Inference failed (exit code: $exit_code)"
            echo "Check log file: $log_file"
            failed=$((failed + 1))
        fi
        
        echo "Progress: $((roi_idx + 1))/145 completed"
        echo ""
    done
    
    echo "Inference completed: $completed successful, $failed failed"
}

# Main execution
echo "Starting inference at $(date)"
run_inference_all_rois

# Generate summary report
echo "=========================================="
echo "Inference Summary Report"
echo "=========================================="
echo "Completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"

# Count successful and failed inferences
successful=0
failed=0

for roi_idx in $(seq $START_ROI $END_ROI); do
    output_file="$OUTPUT_DIR/inference_results_roi_${roi_idx}.csv"
    summary_file="$OUTPUT_DIR/inference_results_roi_${roi_idx}_summary.json"
    
    if [ -f "$output_file" ] && [ -f "$summary_file" ]; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
        echo "Failed ROI: $roi_idx"
    fi
done

echo "Total ROIs: $TOTAL_ROIS"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Success rate: $((successful * 100 / TOTAL_ROIS))%"

# Create summary JSON file
summary_file="$OUTPUT_DIR/inference_summary.json"
cat > "$summary_file" << EOF
{
  "hmuse_version": $HMUSE_VERSION,
  "gpu_id": $GPU_ID,
  "data_file": "$DATA_FILE",
  "test_ids_file": "$TEST_IDS_FILE",
  "models_dir": "$MODELS_DIR",
  "output_dir": "$OUTPUT_DIR",
  "log_dir": "$LOG_DIR",
  "total_rois": $TOTAL_ROIS,
  "successful": $successful,
  "failed": $failed,
  "success_rate": $((successful * 100 / TOTAL_ROIS)),
  "start_time": "$(date -d @$(stat -c %Y "$0"))",
  "end_time": "$(date)"
}
EOF

echo "Summary saved to: $summary_file"
echo "=========================================="
echo "Inference completed!"
echo "==========================================" 