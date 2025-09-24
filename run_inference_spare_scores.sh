#!/bin/bash

# Inference script for spare scores
# Usage: ./run_inference_spare_scores.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

# Configuration
DATA_FILE="./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv"
TEST_IDS_FILE="./data/test_subject_allstudies_ids_dl_muse_spare${HMUSE_VERSION}.pkl"
MODELS_DIR="./models_spare"
OUTPUT_DIR="./inference_results_spare"
LOG_DIR="./logs_inference_spare"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Spare scores configuration
declare -A SPARE_TO_ROI
SPARE_TO_ROI["spare_ad"]=0
SPARE_TO_ROI["spare_ef"]=1
SPARE_SCORES=("spare_ad" "spare_ef")
TOTAL_SPARE_SCORES=2

echo "=========================================="
echo "DKGP Inference for Spare Scores"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "Data file: ${DATA_FILE}"
echo "Test IDs file: ${TEST_IDS_FILE}"
echo "Models directory: ${MODELS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Spare scores: ${SPARE_SCORES[*]}"
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

# Function to run inference for spare scores
run_inference_spare_scores() {
    local completed=0
    local failed=0
    
    echo "Starting inference for spare scores..."
    
    for spare_idx in "${!SPARE_SCORES[@]}"; do
        local spare_score="${SPARE_SCORES[$spare_idx]}"
        local roi_idx="${SPARE_TO_ROI[$spare_score]}"
        
        echo "=========================================="
        echo "Running inference for ${spare_score} (ROI ${roi_idx})"
        echo "=========================================="
        
        # Check if model exists
        local model_file="$MODELS_DIR/population_deep_kernel_gp_${roi_idx}.pth"
        local output_file="$OUTPUT_DIR/inference_results_${spare_score}.csv"
        local log_file="$LOG_DIR/inference_${spare_score}.log"
        
        if [ ! -f "$model_file" ]; then
            echo "${spare_score} (ROI ${roi_idx}): Model not found, skipping..."
            failed=$((failed + 1))
            continue
        fi
        
        if [ -f "$output_file" ]; then
            echo "${spare_score} (ROI ${roi_idx}): Results already exist, skipping..."
            completed=$((completed + 1))
            continue
        fi
        
        echo "Starting inference for ${spare_score} (ROI ${roi_idx}) at $(date)"
        
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
            echo "${spare_score} (ROI ${roi_idx}): Inference completed successfully"
            completed=$((completed + 1))
            
            # Extract key metrics from summary file
            local summary_file="$OUTPUT_DIR/inference_results_${spare_score}_summary.json"
            if [ -f "$summary_file" ]; then
                local mae=$(grep -o '"mae": [0-9.]*' "$summary_file" | cut -d' ' -f2)
                local r2=$(grep -o '"r2": [0-9.-]*' "$summary_file" | cut -d' ' -f2)
                echo "${spare_score} (ROI ${roi_idx}): MAE=${mae}, R²=${r2}"
            fi
        else
            echo "${spare_score} (ROI ${roi_idx}): Inference failed (exit code: $exit_code)"
            echo "Check log file: $log_file"
            failed=$((failed + 1))
        fi
        
        echo "Progress: $((spare_idx + 1))/${TOTAL_SPARE_SCORES} completed"
        echo ""
    done
    
    echo "Inference completed: $completed successful, $failed failed"
}

# Main execution
echo "Starting inference at $(date)"
run_inference_spare_scores

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

for spare_score in "${SPARE_SCORES[@]}"; do
    roi_idx="${SPARE_TO_ROI[$spare_score]}"
    output_file="$OUTPUT_DIR/inference_results_${spare_score}.csv"
    summary_file="$OUTPUT_DIR/inference_results_${spare_score}_summary.json"
    
    if [ -f "$output_file" ] && [ -f "$summary_file" ]; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
        echo "Failed spare score: $spare_score (ROI ${roi_idx})"
    fi
done

echo "Total spare scores: $TOTAL_SPARE_SCORES"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Success rate: $((successful * 100 / TOTAL_SPARE_SCORES))%"

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
  "spare_scores": ["${SPARE_SCORES[0]}", "${SPARE_SCORES[1]}"],
  "spare_to_roi_mapping": {"${SPARE_SCORES[0]}": ${SPARE_TO_ROI[${SPARE_SCORES[0]}]}, "${SPARE_SCORES[1]}": ${SPARE_TO_ROI[${SPARE_SCORES[1]}]}},
  "total_spare_scores": $TOTAL_SPARE_SCORES,
  "successful": $successful,
  "failed": $failed,
  "success_rate": $((successful * 100 / TOTAL_SPARE_SCORES)),
  "start_time": "$(date -d @$(stat -c %Y "$0"))",
  "end_time": "$(date)"
}
EOF

echo "Summary saved to: $summary_file"
echo "=========================================="
echo "Inference completed!"
echo "==========================================" 