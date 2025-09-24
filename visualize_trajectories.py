#!/usr/bin/env python3
"""
Visualize Future Trajectory Predictions
Plot predicted biomarker trajectories over 8 years (every 12 months)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir, model_prefix):
    """Load inference results for a specific model"""
    csv_file = Path(results_dir) / f"{model_prefix}_future_inference.csv"
    json_file = Path(results_dir) / f"{model_prefix}_future_inference_summary.json"
    
    if not csv_file.exists():
        print(f"Warning: Results file not found: {csv_file}")
        return None, None
        
    df = pd.read_csv(csv_file)
    
    summary = None
    if json_file.exists():
        with open(json_file, 'r') as f:
            summary = json.load(f)
    
    return df, summary

def plot_individual_trajectories(df, model_name, output_dir, n_subjects=50):
    """Plot individual subject trajectories"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample subjects for visualization (to avoid overcrowding)
    unique_subjects = df['PTID'].unique()
    if len(unique_subjects) > n_subjects:
        sampled_subjects = np.random.choice(unique_subjects, n_subjects, replace=False)
    else:
        sampled_subjects = unique_subjects
    
    # Plot individual trajectories
    for subject in sampled_subjects:
        subject_data = df[df['PTID'] == subject].sort_values('time_months')
        ax.plot(subject_data['time_months'], subject_data['predicted_value'], 
               alpha=0.3, linewidth=0.8, color='blue')
    
    # Plot mean trajectory with confidence interval
    mean_trajectory = df.groupby('time_months').agg({
        'predicted_value': ['mean', 'std'],
        'lower_bound': 'mean',
        'upper_bound': 'mean'
    }).reset_index()
    
    mean_trajectory.columns = ['time_months', 'mean_pred', 'std_pred', 'mean_lower', 'mean_upper']
    
    # Plot mean trajectory
    ax.plot(mean_trajectory['time_months'], mean_trajectory['mean_pred'], 
           linewidth=3, color='red', label='Population Mean')
    
    # Plot confidence interval
    ax.fill_between(mean_trajectory['time_months'], 
                   mean_trajectory['mean_lower'], 
                   mean_trajectory['mean_upper'],
                   alpha=0.2, color='red', label='95% Confidence Interval')
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel(f'{model_name}', fontsize=12)
    ax.set_title(f'Predicted Trajectories: {model_name}\n8-Year Future Predictions', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add time point markers
    time_points = df['time_months'].unique()
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{int(t/12)}y' for t in time_points])
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / f"{model_name.replace(' ', '_').lower()}_trajectories.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved trajectory plot: {output_file}")

def plot_population_summary(df, model_name, output_dir):
    """Plot population-level summary statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate summary statistics by time point
    summary_stats = df.groupby('time_months').agg({
        'predicted_value': ['mean', 'std', 'min', 'max'],
        'interval_width': 'mean'
    }).reset_index()
    
    summary_stats.columns = ['time_months', 'mean_pred', 'std_pred', 'min_pred', 'max_pred', 'mean_uncertainty']
    
    # Plot 1: Mean trajectory with error bars
    ax1.errorbar(summary_stats['time_months'], summary_stats['mean_pred'], 
                yerr=summary_stats['std_pred'], 
                marker='o', linewidth=2, markersize=8,
                color='blue', capsize=5)
    
    ax1.set_xlabel('Time (months)', fontsize=12)
    ax1.set_ylabel(f'{model_name}', fontsize=12)
    ax1.set_title(f'Population Mean Trajectory: {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(summary_stats['time_months'])
    ax1.set_xticklabels([f'{int(t/12)}y' for t in summary_stats['time_months']])
    
    # Plot 2: Uncertainty over time
    ax2.plot(summary_stats['time_months'], summary_stats['mean_uncertainty'], 
            marker='s', linewidth=2, markersize=8, color='red')
    
    ax2.set_xlabel('Time (months)', fontsize=12)
    ax2.set_ylabel('Mean Prediction Uncertainty', fontsize=12)
    ax2.set_title(f'Prediction Uncertainty Over Time: {model_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(summary_stats['time_months'])
    ax2.set_xticklabels([f'{int(t/12)}y' for t in summary_stats['time_months']])
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / f"{model_name.replace(' ', '_').lower()}_summary.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved summary plot: {output_file}")

def main():
    results_dir = "./future_inference_results"
    output_dir = "./trajectory_plots"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Generating trajectory visualizations...")
    
    # Find all available result files
    result_files = list(Path(results_dir).glob("*_future_inference.csv"))
    available_models = [f.stem.replace('_future_inference', '') for f in result_files]
    
    print(f"Found results for: {available_models}")
    
    # Model name mapping
    model_names = {
        'hippocampus_roi17': 'Hippocampus (ROI 17)',
        'spare_ad': 'SPARE-AD Score',
        'spare_ef': 'SPARE-EF Score',
        'mmse': 'MMSE Score',
        'adas': 'ADAS Score'
    }
    
    # Generate plots for each model
    for model_prefix in available_models:
        print(f"\nProcessing {model_prefix}...")
        
        df, summary = load_results(results_dir, model_prefix)
        if df is None:
            continue
            
        model_name = model_names.get(model_prefix, model_prefix.replace('_', ' ').title())
        
        # Individual trajectories plot
        plot_individual_trajectories(df, model_name, output_dir)
        
        # Population summary plot
        plot_population_summary(df, model_name, output_dir)
        
        print(f"✅ Generated plots for {model_prefix}")
    
    print(f"\n🎉 All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
