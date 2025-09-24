#!/bin/bash

# Training script for the cognitive scores (ADAS and MMSE)
# Usage: ./run_training_cognitive_scores.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

# Configuration for cognitive scores
ADAS_DATA_FILE="./data/subjectsamples_longclean_dlmuse_adas_adni.csv"
ADAS_TRAIN_IDS_FILE="./data/train_subject_adni_ids_adas${HMUSE_VERSION}.pkl"
ADAS_TEST_IDS_FILE="./data/test_subject_adni_ids_adas${HMUSE_VERSION}.pkl"

MMSE_DATA_FILE="./data/subjectsamples_longclean_mmse_dlmuse_allstudies.csv"
MMSE_TRAIN_IDS_FILE="./data/train_subject_allstudies_ids_mmse${HMUSE_VERSION}.pkl"
MMSE_TEST_IDS_FILE="./data/test_subject_allstudies_ids_mmse${HMUSE_VERSION}.pkl"

OUTPUT_DIR="./models_cognitive"
LOG_DIR="./logs_cognitive_hmuse${HMUSE_VERSION}"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Cognitive scores configuration - each has its own file, so ROI index is always 0
declare -A COGNITIVE_TO_ROI
COGNITIVE_TO_ROI["adas"]=0
COGNITIVE_TO_ROI["mmse"]=0
COGNITIVE_SCORES=("adas" "mmse")  # The two cognitive scores
TOTAL_COGNITIVE_SCORES=2

echo "=========================================="
echo "DKGP Training for Cognitive Scores"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "ADAS data file: ${ADAS_DATA_FILE}"
echo "ADAS train IDs file: ${ADAS_TRAIN_IDS_FILE}"
echo "ADAS test IDs file: ${ADAS_TEST_IDS_FILE}"
echo "MMSE data file: ${MMSE_DATA_FILE}"
echo "MMSE train IDs file: ${MMSE_TRAIN_IDS_FILE}"
echo "MMSE test IDs file: ${MMSE_TEST_IDS_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Cognitive scores: ${COGNITIVE_SCORES[*]}"
echo "=========================================="

# Check if required files exist
if [ ! -f "$ADAS_DATA_FILE" ]; then
    echo "Error: ADAS data file not found: $ADAS_DATA_FILE"
    exit 1
fi

if [ ! -f "$ADAS_TRAIN_IDS_FILE" ]; then
    echo "Error: ADAS train IDs file not found: $ADAS_TRAIN_IDS_FILE"
    exit 1
fi

if [ ! -f "$ADAS_TEST_IDS_FILE" ]; then
    echo "Error: ADAS test IDs file not found: $ADAS_TEST_IDS_FILE"
    exit 1
fi

if [ ! -f "$MMSE_DATA_FILE" ]; then
    echo "Error: MMSE data file not found: $MMSE_DATA_FILE"
    exit 1
fi

if [ ! -f "$MMSE_TRAIN_IDS_FILE" ]; then
    echo "Error: MMSE train IDs file not found: $MMSE_TRAIN_IDS_FILE"
    exit 1
fi

if [ ! -f "$MMSE_TEST_IDS_FILE" ]; then
    echo "Error: MMSE test IDs file not found: $MMSE_TEST_IDS_FILE"
    exit 1
fi

# Function to run training for cognitive scores
run_cognitive_training() {
    local completed=0
    local failed=0
    
    echo "Starting training for cognitive scores..."
    
    for cognitive_idx in "${!COGNITIVE_SCORES[@]}"; do
        local cognitive_score="${COGNITIVE_SCORES[$cognitive_idx]}"
        echo "=========================================="
        echo "Training Cognitive Score: ${cognitive_score}"
        echo "=========================================="
        
        # Get ROI index for this cognitive score
        local roi_idx="${COGNITIVE_TO_ROI[$cognitive_score]}"
        
        # Set data files based on cognitive score
        if [ "$cognitive_score" = "adas" ]; then
            local data_file="$ADAS_DATA_FILE"
            local train_ids_file="$ADAS_TRAIN_IDS_FILE"
            local test_ids_file="$ADAS_TEST_IDS_FILE"
        elif [ "$cognitive_score" = "mmse" ]; then
            local data_file="$MMSE_DATA_FILE"
            local train_ids_file="$MMSE_TRAIN_IDS_FILE"
            local test_ids_file="$MMSE_TEST_IDS_FILE"
        fi
        
        # Create separate output directory for each cognitive score
        local cognitive_output_dir="$OUTPUT_DIR/${cognitive_score}"
        mkdir -p "$cognitive_output_dir"
        
        # Check if model already exists (but allow retraining)
        local model_file="$cognitive_output_dir/population_deep_kernel_gp_${roi_idx}.pth"
        local results_file="$cognitive_output_dir/results_roi_${roi_idx}.json"
        
        if [ -f "$model_file" ] && [ -f "$results_file" ]; then
            echo "${cognitive_score} (ROI ${roi_idx}): Model exists, but proceeding with retraining..."
        fi
        
        # Run training with CUDA_VISIBLE_DEVICES
        local log_file="$LOG_DIR/training_${cognitive_score}_roi${roi_idx}.log"
        
        echo "Starting training for ${cognitive_score} (ROI ${roi_idx}) at $(date)"
        echo "Using data file: $data_file"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python pdkgp_training.py \
            --data_file "$data_file" \
            --train_ids_file "$train_ids_file" \
            --test_ids_file "$test_ids_file" \
            --roi_idx "$roi_idx" \
            --output_dir "$cognitive_output_dir" \
            --gpu_id "$GPU_ID" \
            > "$log_file" 2>&1
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "${cognitive_score} (ROI ${roi_idx}): Training completed successfully"
            completed=$((completed + 1))
            
            # Extract key metrics from results file
            if [ -f "$results_file" ]; then
                local mae=$(grep -o '"mae": [0-9.]*' "$results_file" | cut -d' ' -f2)
                local r2=$(grep -o '"r2": [0-9.-]*' "$results_file" | cut -d' ' -f2)
                echo "${cognitive_score} (ROI ${roi_idx}): MAE=${mae}, R²=${r2}"
            fi
        else
            echo "${cognitive_score} (ROI ${roi_idx}): Training failed (exit code: $exit_code)"
            echo "Check log file: $log_file"
            failed=$((failed + 1))
        fi
        
        echo "Progress: $((cognitive_idx + 1))/${TOTAL_COGNITIVE_SCORES} completed"
        echo ""
    done
    
    echo "Cognitive score training completed: $completed successful, $failed failed"
}

# Main execution
echo "Starting training at $(date)"
run_cognitive_training

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

for cognitive_score in "${COGNITIVE_SCORES[@]}"; do
    roi_idx="${COGNITIVE_TO_ROI[$cognitive_score]}"
    cognitive_output_dir="$OUTPUT_DIR/${cognitive_score}"
    model_file="$cognitive_output_dir/population_deep_kernel_gp_${roi_idx}.pth"
    results_file="$cognitive_output_dir/results_roi_${roi_idx}.json"
    
    if [ -f "$model_file" ] && [ -f "$results_file" ]; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
        echo "Failed cognitive score: $cognitive_score (ROI ${roi_idx})"
    fi
done

echo "Total cognitive scores: $TOTAL_COGNITIVE_SCORES"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Success rate: $((successful * 100 / TOTAL_COGNITIVE_SCORES))%"

# Create summary JSON file
summary_file="$OUTPUT_DIR/cognitive_training_summary.json"
cat > "$summary_file" << EOF
{
  "hmuse_version": $HMUSE_VERSION,
  "gpu_id": $GPU_ID,
  "adas_data_file": "$ADAS_DATA_FILE",
  "adas_train_ids_file": "$ADAS_TRAIN_IDS_FILE",
  "adas_test_ids_file": "$ADAS_TEST_IDS_FILE",
  "mmse_data_file": "$MMSE_DATA_FILE",
  "mmse_train_ids_file": "$MMSE_TRAIN_IDS_FILE",
  "mmse_test_ids_file": "$MMSE_TEST_IDS_FILE",
  "output_dir": "$OUTPUT_DIR",
  "log_dir": "$LOG_DIR",
  "cognitive_scores": ["${COGNITIVE_SCORES[0]}", "${COGNITIVE_SCORES[1]}"],
  "cognitive_to_roi_mapping": {"${COGNITIVE_SCORES[0]}": ${COGNITIVE_TO_ROI[${COGNITIVE_SCORES[0]}]}, "${COGNITIVE_SCORES[1]}": ${COGNITIVE_TO_ROI[${COGNITIVE_SCORES[1]}]}},
  "total_cognitive_scores": $TOTAL_COGNITIVE_SCORES,
  "successful": $successful,
  "failed": $failed,
  "success_rate": $((successful * 100 / TOTAL_COGNITIVE_SCORES)),
  "start_time": "$(date -d @$(stat -c %Y "$0"))",
  "end_time": "$(date)"
}
EOF

echo "Summary saved to: $summary_file"
echo "=========================================="
echo "Cognitive score training completed!"
echo "==========================================" 