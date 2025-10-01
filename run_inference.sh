#!/bin/bash

# Comprehensive Biomarker Inference Script
# Predict biomarker values 8 years ahead (every 12 months)
# Supports all Volume ROIs, Hippocampus R/L, Lateral Ventricle R/L, SPARE-AD, SPARE-BA, MMSE, ADAS

# Set default parameters
GPU_ID=0
OUTPUT_DIR="./output"

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
    
    output_file="$OUTPUT_DIR/${output_prefix}_output.csv"
    
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

# Function to run inference for all ROIs and combine into single CSV
run_inference_all_rois() {
    echo "Running inference for all 145 Volume ROIs..."
    echo "This will create individual CSV files and then combine them into volume_rois_output.csv"
    
    local temp_dir="$OUTPUT_DIR/temp_rois"
    mkdir -p "$temp_dir"
    
    local completed=0
    local failed=0
    
    # Run inference for each ROI
    for roi_idx in {0..144}; do
        echo "Processing ROI $roi_idx..."
        local temp_output="$temp_dir/roi_${roi_idx}_output.csv"
        
        run_inference \
            "Volume ROI $roi_idx" \
            "./models/population_deep_kernel_gp_MUSE_${roi_idx}.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            $roi_idx \
            "temp_rois/roi_${roi_idx}"
        
        if [ $? -eq 0 ]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    echo "Completed: $completed ROIs, Failed: $failed ROIs"
    
    # Combine all ROI results into single CSV
    echo "Combining all ROI results into volume_rois_output.csv..."
    python -c "
import pandas as pd
import os
import glob

temp_dir = '$temp_dir'
output_file = '$OUTPUT_DIR/volume_rois_output.csv'

# Get all CSV files
csv_files = glob.glob(os.path.join(temp_dir, 'roi_*_output.csv'))
csv_files.sort()

if not csv_files:
    print('No CSV files found to combine')
    exit(1)

print(f'Found {len(csv_files)} CSV files to combine')

# Read first file to get structure
first_df = pd.read_csv(csv_files[0])
print(f'First file columns: {list(first_df.columns)}')

# Initialize combined dataframe with PTID and Time from first file
combined_df = first_df[['PTID', 'Time']].copy()

# Add each ROI column
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    roi_idx = os.path.basename(csv_file).split('_')[1]
    
    # Get the biomarker column (should be the last column)
    biomarker_col = [col for col in df.columns if col not in ['PTID', 'Time']][0]
    
    # Rename to DL_MUSE_{roi_idx}
    new_col_name = f'DL_MUSE_{roi_idx}'
    combined_df[new_col_name] = df[biomarker_col]
    
    if i % 20 == 0:
        print(f'Processed {i+1}/{len(csv_files)} ROIs...')

# Save combined results
combined_df.to_csv(output_file, index=False)
print(f'Combined results saved to: {output_file}')
print(f'Final dataframe shape: {combined_df.shape}')
print(f'Columns: {list(combined_df.columns)[:10]}... (showing first 10)')
"

    # Clean up temporary files
    echo "Cleaning up temporary files..."
    rm -rf "$temp_dir"
    
    echo "✅ All ROI inference completed and combined into volume_rois_output.csv"
}

# Display available biomarkers
echo ""
echo "Available biomarker types:"
echo "=== Volume ROIs ==="
echo "  hippocampus_right (ROI 14) - Right Hippocampus"
echo "  hippocampus_left (ROI 15)  - Left Hippocampus"
echo "  ventricle_right (ROI 16)   - Right Lateral Ventricle"
echo "  ventricle_left (ROI 17)    - Left Lateral Ventricle"
echo "  volume_rois                - All 145 Volume ROIs (single CSV)"
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
            "./models/population_deep_kernel_gp_14.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            14 \
            "hippocampus_right"
        ;;

    "hippocampus_left"|"hippo_left"|"15")
        run_inference \
            "Left Hippocampus" \
            "./models/population_deep_kernel_gp_15.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            15 \
            "hippocampus_left"
        ;;

    # Volume ROIs - Lateral Ventricles
    "ventricle_right"|"lateral_ventricle_right"|"16")
        run_inference \
            "Right Lateral Ventricle" \
            "./models/population_deep_kernel_gp_16.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            16 \
            "lateral_ventricle_right"
        ;;

    "ventricle_left"|"lateral_ventricle_left"|"17")
        run_inference \
            "Left Lateral Ventricle" \
            "./models/population_deep_kernel_gp_17.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            17 \
            "lateral_ventricle_left"
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

    # Volume ROIs - Run all 145 ROIs and combine into single CSV
    "volume_rois"|"all_rois")
        run_inference_all_rois
        ;;

    # Run all biomarkers
    "all")
        echo "Running inference for all biomarkers..."
        
        # Right Hippocampus
        run_inference \
            "Right Hippocampus" \
            "./models/population_deep_kernel_gp_14.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            14 \
            "hippocampus_right"
        
        # Left Hippocampus
        run_inference \
            "Left Hippocampus" \
            "./models/population_deep_kernel_gp_15.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            15 \
            "hippocampus_left"
        
        # Right Lateral Ventricle
        run_inference \
            "Right Lateral Ventricle" \
            "./models/population_deep_kernel_gp_16.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            16 \
            "lateral_ventricle_right"
        
        # Left Lateral Ventricle
        run_inference \
            "Left Lateral Ventricle" \
            "./models/population_deep_kernel_gp_17.pth" \
            "./data/subjectsamples_longclean_dl_muse_allstudies.csv" \
            "./data/test_subject_allstudies_ids_dl_hmuse0.pkl" \
            17 \
            "lateral_ventricle_left"
        
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
        echo "  volume_rois            - All 145 Volume ROIs (single CSV)"
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
        echo "  $0 volume_rois         # Run for all 145 ROIs (single CSV)"
        echo "  $0 all                 # Run for all biomarkers"
        echo ""
        echo "Output:"
        echo "  Results will be saved to ./output/{biomarker_name}_output.csv"
        echo "  CSV format: PTID, Time, biomarker_columns"
        echo "  For volume_rois: PTID, Time, DL_MUSE_0, DL_MUSE_1, ..., DL_MUSE_144"
        ;;
esac

echo ""
echo "Biomarker inference script completed!"
