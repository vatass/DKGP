#!/bin/bash

# Training script for the two spare scores
# Usage: ./run_training_spare_scores.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

# Configuration - Update these paths as needed
DATA_FILE="./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv"
TRAIN_IDS_FILE="./data/train_subject_allstudies_ids_dl_muse_spare${HMUSE_VERSION}.pkl"
TEST_IDS_FILE="./data/test_subject_allstudies_ids_dl_muse_spare${HMUSE_VERSION}.pkl"
OUTPUT_DIR="./models_spare"
LOG_DIR="./logs_spare_hmuse${HMUSE_VERSION}"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Spare scores configuration - map to ROI indices
declare -A SPARE_TO_ROI
SPARE_TO_ROI["spare_ad"]=0
SPARE_TO_ROI["spare_ef"]=1
SPARE_SCORES=("spare_ad" "spare_ef")  # The two spare scores
TOTAL_SPARE_SCORES=2

echo "=========================================="
echo "DKGP Training for Spare Scores"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "Data file: ${DATA_FILE}"
echo "Train IDs file: ${TRAIN_IDS_FILE}"
echo "Test IDs file: ${TEST_IDS_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Spare scores: ${SPARE_SCORES[*]}"
echo "=========================================="

# Check if required files exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_IDS_FILE" ]; then
    echo "Error: Train IDs file not found: $TRAIN_IDS_FILE"
    exit 1
fi

if [ ! -f "$TEST_IDS_FILE" ]; then
    echo "Error: Test IDs file not found: $TEST_IDS_FILE"
    exit 1
fi

# Function to run training for spare scores
run_spare_training() {
    local completed=0
    local failed=0
    
    echo "Starting training for spare scores..."
    
    for spare_idx in "${!SPARE_SCORES[@]}"; do
        local spare_score="${SPARE_SCORES[$spare_idx]}"
        echo "=========================================="
        echo "Training Spare Score: ${spare_score}"
        echo "=========================================="
        
        # Get ROI index for this spare score
        local roi_idx="${SPARE_TO_ROI[$spare_score]}"
        
        # Check if model already exists (but allow retraining)
        local model_file="$OUTPUT_DIR/population_deep_kernel_gp_${roi_idx}.pth"
        local results_file="$OUTPUT_DIR/results_roi_${roi_idx}.json"
        
        if [ -f "$model_file" ] && [ -f "$results_file" ]; then
            echo "${spare_score} (ROI ${roi_idx}): Model exists, but proceeding with retraining..."
        fi
        
        # Run training with CUDA_VISIBLE_DEVICES
        local log_file="$LOG_DIR/training_${spare_score}_roi${roi_idx}.log"
        
        echo "Starting training for ${spare_score} (ROI ${roi_idx}) at $(date)"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python pdkgp_training.py \
            --data_file "$DATA_FILE" \
            --train_ids_file "$TRAIN_IDS_FILE" \
            --test_ids_file "$TEST_IDS_FILE" \
            --roi_idx "$roi_idx" \
            --output_dir "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            > "$log_file" 2>&1
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "${spare_score} (ROI ${roi_idx}): Training completed successfully"
            completed=$((completed + 1))
            
            # Extract key metrics from results file
            if [ -f "$results_file" ]; then
                local mae=$(grep -o '"mae": [0-9.]*' "$results_file" | cut -d' ' -f2)
                local r2=$(grep -o '"r2": [0-9.-]*' "$results_file" | cut -d' ' -f2)
                echo "${spare_score} (ROI ${roi_idx}): MAE=${mae}, R²=${r2}"
            fi
        else
            echo "${spare_score} (ROI ${roi_idx}): Training failed (exit code: $exit_code)"
            echo "Check log file: $log_file"
            failed=$((failed + 1))
        fi
        
        echo "Progress: $((spare_idx + 1))/${TOTAL_SPARE_SCORES} completed"
        echo ""
    done
    
    echo "Spare score training completed: $completed successful, $failed failed"
}

# Main execution
echo "Starting training at $(date)"
run_spare_training

# Generate summary report
echo "=========================================="
echo "Training Summary Report"
echo "=========================================="
echo "Completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"

# Count successful and failed trainings
successful=0
failed=0

for spare_score in "${SPARE_SCORES[@]}"; do
    roi_idx="${SPARE_TO_ROI[$spare_score]}"
    model_file="$OUTPUT_DIR/population_deep_kernel_gp_${roi_idx}.pth"
    results_file="$OUTPUT_DIR/results_roi_${roi_idx}.json"
    
    if [ -f "$model_file" ] && [ -f "$results_file" ]; then
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
summary_file="$OUTPUT_DIR/spare_training_summary.json"
cat > "$summary_file" << EOF
{
  "hmuse_version": $HMUSE_VERSION,
  "gpu_id": $GPU_ID,
  "data_file": "$DATA_FILE",
  "train_ids_file": "$TRAIN_IDS_FILE",
  "test_ids_file": "$TEST_IDS_FILE",
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
echo "Spare score training completed!"
echo "==========================================" 