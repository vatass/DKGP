#!/bin/bash

# Comprehensive Biomarker Inference Script
# Predict biomarker values 8 years ahead (every 12 months)
# Supports all Volume ROIs, Hippocampus R/L, Lateral Ventricle R/L, SPARE-AD, SPARE-BA, MMSE, ADAS

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
    
    # Check if model file exists
    if [ ! -f "$model_file" ]; then
        echo "❌ Error: Model file not found: $model_file"
        return 1
    fi
    
    # Check if data file exists
    if [ ! -f "$data_file" ]; then
        echo "❌ Error: Data file not found: $data_file"
        return 1
    fi
    
    # Check if test IDs file exists
    if [ ! -f "$test_ids_file" ]; then
        echo "❌ Error: Test IDs file not found: $test_ids_file"
        return 1
    fi
    
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

# Display available biomarkers
echo ""
echo "Available biomarker types:"
echo "=== Volume ROIs ==="
echo "  hippocampus_right (ROI 14) - Right Hippocampus"
echo "  hippocampus_left (ROI 15)  - Left Hippocampus"
echo "  ventricle_right (ROI 16)   - Right Lateral Ventricle"
echo "  ventricle_left (ROI 17)    - Left Lateral Ventricle"
echo "  volume_rois                - All 145 Volume ROIs"
echo ""
echo "=== SPARE Scores ==="
echo "  spare_ad (ROI 0)           - SPARE-AD Score"
echo "  spare_ba (ROI 1)           - SPARE-BA Score"
echo ""
echo "=== Cognitive Scores ==="
echo "  mmse (ROI 0)               - MMSE Cognitive Score"
echo "  adas (ROI 0)               - ADAS Cognitive Score"
echo ""
echo "=== Combined Options ==="
echo "  all                        - Run inference for all biomarkers"
echo ""

# Parse biomarker argument and run appropriate inference
case "$1" in

    # Volume ROIs - Hippocampus
    "hippocampus_right"|"hippo_right"|"14")
        run_inference \
            "Right Hippocampus" \
            "./models/population_deep_kernel_gp_MUSE_14.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            14 \
            "hippocampus_right_roi14"
        ;;

    "hippocampus_left"|"hippo_left"|"15")
        run_inference \
            "Left Hippocampus" \
            "./models/population_deep_kernel_gp_MUSE_15.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            15 \
            "hippocampus_left_roi15"
        ;;

    # Volume ROIs - Lateral Ventricles
    "ventricle_right"|"lateral_ventricle_right"|"16")
        run_inference \
            "Right Lateral Ventricle" \
            "./models/population_deep_kernel_gp_MUSE_16.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            16 \
            "lateral_ventricle_right_roi16"
        ;;

    "ventricle_left"|"lateral_ventricle_left"|"17")
        run_inference \
            "Left Lateral Ventricle" \
            "./models/population_deep_kernel_gp_MUSE_17.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            17 \
            "lateral_ventricle_left_roi17"
        ;;

    # SPARE Scores
    "spare_ad"|"spare-ad")
        run_inference \
            "SPARE-AD" \
            "./models_spare/population_deep_kernel_gp_0.pth" \
            "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
            0 \
            "spare_ad"
        ;;

    "spare_ba"|"spare-ba")
        run_inference \
            "SPARE-BA" \
            "./models_spare/population_deep_kernel_gp_1.pth" \
            "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
            1 \
            "spare_ba"
        ;;

    # Cognitive Scores
    "mmse")
        run_inference \
            "MMSE" \
            "./models_cognitive/mmse/population_deep_kernel_gp_0.pth" \
            "./data/subjectsamples_longclean_mmse_dlmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_mmse0.pkl" \
            0 \
            "mmse"
        ;;

    "adas")
        run_inference \
            "ADAS" \
            "./models_cognitive/adas/population_deep_kernel_gp_0.pth" \
            "./data/subjectsamples_longclean_dlmuse_adas_adni.csv" \
            "./data/test_subject_adni_ids_adas0.pkl" \
            0 \
            "adas"
        ;;

    # Volume ROIs - Run all 145 ROIs
    "volume_rois"|"all_rois")
        echo "Running inference for all 145 Volume ROIs..."
        for roi_idx in {0..144}; do
            echo "Processing ROI $roi_idx..."
            run_inference \
                "Volume ROI $roi_idx" \
                "./models/population_deep_kernel_gp_MUSE_${roi_idx}.pth" \
                "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
                "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
                $roi_idx \
                "volume_roi_${roi_idx}"
        done
        ;;

    # Run all biomarkers
    "all")
        echo "Running inference for all biomarkers..."
        
        # Right Hippocampus
        run_inference \
            "Right Hippocampus" \
            "./models/population_deep_kernel_gp_MUSE_14.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            14 \
            "hippocampus_right_roi14"
        
        # Left Hippocampus
        run_inference \
            "Left Hippocampus" \
            "./models/population_deep_kernel_gp_MUSE_15.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            15 \
            "hippocampus_left_roi15"
        
        # Right Lateral Ventricle
        run_inference \
            "Right Lateral Ventricle" \
            "./models/population_deep_kernel_gp_MUSE_16.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            16 \
            "lateral_ventricle_right_roi16"
        
        # Left Lateral Ventricle
        run_inference \
            "Left Lateral Ventricle" \
            "./models/population_deep_kernel_gp_MUSE_17.pth" \
            "./data/subjectsamples_longclean_dl_hmuse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            17 \
            "lateral_ventricle_left_roi17"
        
        # SPARE-AD
        run_inference \
            "SPARE-AD" \
            "./models_spare/population_deep_kernel_gp_0.pth" \
            "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
            0 \
            "spare_ad"
        
        # SPARE-BA
        run_inference \
            "SPARE-BA" \
            "./models_spare/population_deep_kernel_gp_1.pth" \
            "./data/subjectsamples_longclean_dl_muse_spare_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_muse_spare0.pkl" \
            1 \
            "spare_ba"
        
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
        ;;

    *)
        echo "Usage: $0 [biomarker_name]"
        echo ""
        echo "=== Volume ROIs ==="
        echo "  hippocampus_right      - Right Hippocampus (ROI 14)"
        echo "  hippocampus_left       - Left Hippocampus (ROI 15)"
        echo "  ventricle_right        - Right Lateral Ventricle (ROI 16)"
        echo "  ventricle_left         - Left Lateral Ventricle (ROI 17)"
        echo "  volume_rois            - All 145 Volume ROIs"
        echo ""
        echo "=== SPARE Scores ==="
        echo "  spare_ad               - SPARE-AD Score"
        echo "  spare_ba               - SPARE-BA Score"
        echo ""
        echo "=== Cognitive Scores ==="
        echo "  mmse                   - MMSE Cognitive Score"
        echo "  adas                   - ADAS Cognitive Score"
        echo ""
        echo "=== Combined Options ==="
        echo "  all                    - Run inference for all biomarkers"
        echo ""
        echo "Examples:"
        echo "  $0 hippocampus_right   # Run for right hippocampus only"
        echo "  $0 spare_ad            # Run for SPARE-AD only"
        echo "  $0 mmse                # Run for MMSE only"
        echo "  $0 all                 # Run for all biomarkers"
        ;;
esac

echo ""
echo "Biomarker inference script completed!"
