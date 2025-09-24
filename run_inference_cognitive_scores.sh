#!/bin/bash

# Inference script for cognitive scores (ADAS and MMSE)
# Usage: ./run_inference_cognitive_scores.sh [hmuse_version] [gpu_id]

# Default values
HMUSE_VERSION=${1:-0}  # Default to hmuse0
GPU_ID=${2:-0}         # Default GPU ID

# Configuration for cognitive scores
ADAS_DATA_FILE="./data/subjectsamples_longclean_dlmuse_adas_adni.csv"
ADAS_TEST_IDS_FILE="./data/test_subject_adni_ids_adas${HMUSE_VERSION}.pkl"

MMSE_DATA_FILE="./data/subjectsamples_longclean_mmse_dlmuse_allstudies.csv"
MMSE_TEST_IDS_FILE="./data/test_subject_allstudies_ids_mmse${HMUSE_VERSION}.pkl"

MODELS_DIR="./models_cognitive"
OUTPUT_DIR="./inference_results_cognitive"
LOG_DIR="./logs_inference_cognitive"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Cognitive scores configuration
declare -A COGNITIVE_TO_ROI
COGNITIVE_TO_ROI["adas"]=0
COGNITIVE_TO_ROI["mmse"]=0
COGNITIVE_SCORES=("adas" "mmse")
TOTAL_COGNITIVE_SCORES=2

echo "=========================================="
echo "DKGP Inference for Cognitive Scores"
echo "=========================================="
echo "HMUSE Version: ${HMUSE_VERSION}"
echo "GPU ID: ${GPU_ID}"
echo "ADAS data file: ${ADAS_DATA_FILE}"
echo "ADAS test IDs file: ${ADAS_TEST_IDS_FILE}"
echo "MMSE data file: ${MMSE_DATA_FILE}"
echo "MMSE test IDs file: ${MMSE_TEST_IDS_FILE}"
echo "Models directory: ${MODELS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Cognitive scores: ${COGNITIVE_SCORES[*]}"
echo "=========================================="

# Check if required files exist
if [ ! -f "$ADAS_DATA_FILE" ]; then
    echo "Error: ADAS data file not found: $ADAS_DATA_FILE"
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

if [ ! -f "$MMSE_TEST_IDS_FILE" ]; then
    echo "Error: MMSE test IDs file not found: $MMSE_TEST_IDS_FILE"
    exit 1
fi

# Function to run inference for cognitive scores
run_inference_cognitive_scores() {
    local completed=0
    local failed=0
    
    echo "Starting inference for cognitive scores..."
    
    for cognitive_idx in "${!COGNITIVE_SCORES[@]}"; do
        local cognitive_score="${COGNITIVE_SCORES[$cognitive_idx]}"
        local roi_idx="${COGNITIVE_TO_ROI[$cognitive_score]}"
        
        echo "=========================================="
        echo "Running inference for ${cognitive_score} (ROI ${roi_idx})"
        echo "=========================================="
        
        # Set data files based on cognitive score
        if [ "$cognitive_score" = "adas" ]; then
            local data_file="$ADAS_DATA_FILE"
            local test_ids_file="$ADAS_TEST_IDS_FILE"
        elif [ "$cognitive_score" = "mmse" ]; then
            local data_file="$MMSE_DATA_FILE"
            local test_ids_file="$MMSE_TEST_IDS_FILE"
        fi
        
        # Check if model exists in the cognitive score specific directory
        local cognitive_models_dir="$MODELS_DIR/${cognitive_score}"
        local model_file="$cognitive_models_dir/population_deep_kernel_gp_${roi_idx}.pth"
        local output_file="$OUTPUT_DIR/inference_results_${cognitive_score}.csv"
        local log_file="$LOG_DIR/inference_${cognitive_score}.log"
        
        if [ ! -f "$model_file" ]; then
            echo "${cognitive_score} (ROI ${roi_idx}): Model not found, skipping..."
            failed=$((failed + 1))
            continue
        fi
        
        if [ -f "$output_file" ]; then
            echo "${cognitive_score} (ROI ${roi_idx}): Results already exist, skipping..."
            completed=$((completed + 1))
            continue
        fi
        
        echo "Starting inference for ${cognitive_score} (ROI ${roi_idx}) at $(date)"
        echo "Using data file: $data_file"
        
        # Run inference
        CUDA_VISIBLE_DEVICES=$GPU_ID python pdkgp_inference.py \
            --data_file "$data_file" \
            --test_ids_file "$test_ids_file" \
            --model_file "$model_file" \
            --roi_idx "$roi_idx" \
            --output_file "$output_file" \
            --gpu_id "$GPU_ID" \
            > "$log_file" 2>&1
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "${cognitive_score} (ROI ${roi_idx}): Inference completed successfully"
            completed=$((completed + 1))
            
            # Extract key metrics from summary file
            local summary_file="$OUTPUT_DIR/inference_results_${cognitive_score}_summary.json"
            if [ -f "$summary_file" ]; then
                local mae=$(grep -o '"mae": [0-9.]*' "$summary_file" | cut -d' ' -f2)
                local r2=$(grep -o '"r2": [0-9.-]*' "$summary_file" | cut -d' ' -f2)
                echo "${cognitive_score} (ROI ${roi_idx}): MAE=${mae}, R²=${r2}"
            fi
        else
            echo "${cognitive_score} (ROI ${roi_idx}): Inference failed (exit code: $exit_code)"
            echo "Check log file: $log_file"
            failed=$((failed + 1))
        fi
        
        echo "Progress: $((cognitive_idx + 1))/${TOTAL_COGNITIVE_SCORES} completed"
        echo ""
    done
    
    echo "Inference completed: $completed successful, $failed failed"
}

# Main execution
echo "Starting inference at $(date)"
run_inference_cognitive_scores

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

for cognitive_score in "${COGNITIVE_SCORES[@]}"; do
    roi_idx="${COGNITIVE_TO_ROI[$cognitive_score]}"
    output_file="$OUTPUT_DIR/inference_results_${cognitive_score}.csv"
    summary_file="$OUTPUT_DIR/inference_results_${cognitive_score}_summary.json"
    
    if [ -f "$output_file" ] && [ -f "$summary_file" ]; then
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
summary_file="$OUTPUT_DIR/inference_summary.json"
cat > "$summary_file" << EOF
{
  "hmuse_version": $HMUSE_VERSION,
  "gpu_id": $GPU_ID,
  "adas_data_file": "$ADAS_DATA_FILE",
  "adas_test_ids_file": "$ADAS_TEST_IDS_FILE",
  "mmse_data_file": "$MMSE_DATA_FILE",
  "mmse_test_ids_file": "$MMSE_TEST_IDS_FILE",
  "models_dir": "$MODELS_DIR",
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
echo "Inference completed!"
echo "==========================================" 