#!/bin/bash

# Future Time Point Inference Script
# Predict biomarker values 8 years ahead (every 12 months)

# Set default parameters
GPU_ID=0
OUTPUT_DIR="./future_inference_results"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting Trajectory Prediction for the Selected Biomarker..."
echo "Output directory: $OUTPUT_DIR"

# Function to run inference for a specific model
run_inference() {
    local model_type=$1
    local model_file=$2
    local data_file=$3
    local test_ids_file=$4
    local roi_idx=$5
    local output_prefix=$6
    
    echo ""
    echo "=== Running inference for $model_type (ROI $roi_idx) ==="
    echo "Model: $model_file"
    echo "Data: $data_file"
    echo "Test IDs: $test_ids_file"
    
    output_file="$OUTPUT_DIR/${output_prefix}_future_inference.csv"
    
    python pdkgp_future_inference.py \
        --data_file "$data_file" \
        --test_ids_file "$test_ids_file" \
        --model_file "$model_file" \
        --roi_idx $roi_idx \
        --output_file "$output_file" \
        --gpu_id $GPU_ID
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $model_type inference completed"
        echo "Results saved to: $output_file"
    else
        echo "❌ Error: $model_type inference failed"
    fi
}

# Example usage for different model types:

echo ""
echo "Available model types:"
echo "1. Hippocampus (ROI 17) - Best performing model"
echo "2. SPARE-AD Score"
echo "3. SPARE-EF Score" 
echo "4. MMSE Cognitive Score"
echo "5. ADAS Cognitive Score"
echo ""

# Example: Run inference for Hippocampus (ROI 17)
if [ "$1" == "hippocampus" ] || [ "$1" == "17" ]; then
    run_inference \
        "Hippocampus" \
        "./models/population_deep_kernel_gp_MUSE_17.pth" \
        "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
        17 \
        "hippocampus_roi17"

# Example: Run inference for SPARE-AD
elif [ "$1" == "spare_ad" ] || [ "$1" == "spare-ad" ]; then
    run_inference \
        "SPARE-AD" \
        "./models_spare/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
        0 \
        "spare_ad"

# Example: Run inference for SPARE-EF
elif [ "$1" == "spare_ef" ] || [ "$1" == "spare-ef" ]; then
    run_inference \
        "SPARE-EF" \
        "./models_spare/population_deep_kernel_gp_1.pth" \
        "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
        1 \
        "spare_ef"

# Example: Run inference for MMSE
elif [ "$1" == "mmse" ]; then
    run_inference \
        "MMSE" \
        "./models_cognitive/mmse/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_mmse_dlmuse_allstudies.csv" \
        "./data/test_subject_allstudies_ids_mmse0.pkl" \
        0 \
        "mmse"

# Example: Run inference for ADAS
elif [ "$1" == "adas" ]; then
    run_inference \
        "ADAS" \
        "./models_cognitive/adas/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_dlmuse_adas_adni.csv" \
        "./data/test_subject_adni_ids_adas0.pkl" \
        0 \
        "adas"

# Run all models
elif [ "$1" == "all" ]; then
    echo "Running inference for all models..."
    
    # Hippocampus
    run_inference \
        "Hippocampus" \
        "./models/population_deep_kernel_gp_MUSE_17.pth" \
        "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
        17 \
        "hippocampus_roi17"
    
    # SPARE-AD
    run_inference \
        "SPARE-AD" \
        "./models_spare/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
        0 \
        "spare_ad"
    
    # SPARE-EF
    run_inference \
        "SPARE-EF" \
        "./models_spare/population_deep_kernel_gp_1.pth" \
        "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
        "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
        1 \
        "spare_ef"
    
    # MMSE
    run_inference \
        "MMSE" \
        "./models_cognitive/mmse/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_mmse_dlmuse_allstudies.csv" \
        "./data/test_subject_allstudies_ids_mmse0.pkl" \
        0 \
        "mmse"
    
    # ADAS
    run_inference \
        "ADAS" \
        "./models_cognitive/adas/population_deep_kernel_gp_0.pth" \
        "./data/subjectsamples_longclean_dlmuse_adas_adni.csv" \
        "./data/test_subject_adni_ids_adas0.pkl" \
        0 \
        "adas"

else
    echo "Usage: $0 [model_type]"
    echo ""
    echo "Available model types:"
    echo "  hippocampus (or 17) - Hippocampus ROI 17 model"
    echo "  spare_ad           - SPARE-AD score model"
    echo "  spare_ef           - SPARE-EF score model"
    echo "  mmse               - MMSE cognitive score model"
    echo "  adas               - ADAS cognitive score model"
    echo "  all                - Run inference for all models"
    echo ""
    echo "Examples:"
    echo "  $0 hippocampus     # Run for hippocampus only"
    echo "  $0 spare_ad        # Run for SPARE-AD only"
    echo "  $0 all             # Run for all models"
fi

echo ""
echo "Future inference script completed!"
