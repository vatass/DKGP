#!/usr/bin/env python3
"""
Single Subject Trajectory Visualization
Creates publication-ready single subject trajectory plots with uncertainty bounds
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')

def load_inference_data(csv_file):
    """Load inference results from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading {csv_file}: {e}")
        return None

def plot_single_subject_trajectory(df, subject_id, biomarker_name, output_dir, figsize=(10, 6)):
    """Plot single subject trajectory with prediction and uncertainty bounds"""
    
    # Filter data for the specific subject
    subject_data = df[df['PTID'] == subject_id].sort_values('time_months')
    
    if len(subject_data) == 0:
        print(f"✗ No data found for subject: {subject_id}")
        return None
    
    print(f"✓ Plotting trajectory for subject: {subject_id}")
    print(f"  Data points: {len(subject_data)}")
    print(f"  Time range: {subject_data['time_months'].min()}-{subject_data['time_months'].max()} months")
    
    # Create figure with publication quality
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    time_points = subject_data['time_months']
    predictions = subject_data['predicted_value']
    lower_bounds = subject_data['lower_bound']
    upper_bounds = subject_data['upper_bound']
    
    # Plot uncertainty band (confidence interval)
    ax.fill_between(time_points, lower_bounds, upper_bounds, 
                   color='lightblue', alpha=0.3, 
                   label='95% Confidence Interval')
    
    # Plot prediction line
    ax.plot(time_points, predictions, 'o-', 
           color='darkblue', linewidth=3, markersize=8,
           label='DKGP Prediction', zorder=10)
    
    # Customize plot
    ax.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{biomarker_name} Prediction', fontsize=14, fontweight='bold')
    ax.set_title(f'DKGP Trajectory Forecast: {biomarker_name}\n'
                f'Subject: {subject_id}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # Calculate and display trajectory statistics
    if len(subject_data) > 1:
        slope, intercept = np.polyfit(time_points, predictions, 1)
        r_squared = np.corrcoef(time_points, predictions)[0, 1]**2
        
        # Add statistics text box
        stats_text = f'Trajectory Statistics:\n'
        stats_text += f'Slope: {slope:.4f} units/month\n'
        stats_text += f'R²: {r_squared:.3f}\n'
        stats_text += f'Time points: {len(time_points)}\n'
        stats_text += f'Prediction range: {predictions.min():.3f} to {predictions.max():.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Add uncertainty statistics
    if 'variance' in subject_data.columns:
        mean_uncertainty = subject_data['variance'].mean()
        uncertainty_text = f'Mean Uncertainty: {mean_uncertainty:.4f}'
        ax.text(0.98, 0.02, uncertainty_text, transform=ax.transAxes, 
               horizontalalignment='right', verticalalignment='bottom', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with specific naming convention: subject_id_biomarker_name_forecast.png
    output_filename = f"{subject_id}_{biomarker_name.replace(' ', '_')}_forecast.png"
    output_file = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved single subject plot: {output_file}")
    
    plt.show()
    return output_file

def get_random_subject(df):
    """Get a random subject ID from the data"""
    subjects = df['PTID'].unique()
    if len(subjects) == 0:
        return None
    return np.random.choice(subjects)

def main():
    parser = argparse.ArgumentParser(description='Single Subject Trajectory Visualization')
    parser.add_argument('--csv_file', required=True, help='Path to inference results CSV file')
    parser.add_argument('--output_dir', default='./single_subject_plots', help='Output directory for plots')
    parser.add_argument('--biomarker_name', help='Name of biomarker (auto-detected from filename if not provided)')
    parser.add_argument('--subject_id', help='Specific subject ID to plot (random if not provided)')
    parser.add_argument('--random', action='store_true', help='Force random subject selection')
    
    args = parser.parse_args()
    
    # Auto-detect biomarker name from filename
    if args.biomarker_name is None:
        filename = Path(args.csv_file).stem
        args.biomarker_name = filename.replace('_output', '').replace('_', ' ').title()
    
    print(f"🔬 Single Subject Trajectory Visualization")
    print(f"📊 Biomarker: {args.biomarker_name}")
    print(f"📁 Input file: {args.csv_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Load data
    df = load_inference_data(args.csv_file)
    if df is None:
        return
    
    # Determine subject ID
    if args.subject_id:
        subject_id = args.subject_id
        print(f"🎯 Using specified subject: {subject_id}")
    else:
        subject_id = get_random_subject(df)
        print(f"🎲 Using random subject: {subject_id}")
    
    if subject_id is None:
        print("✗ No subjects found in data")
        return
    
    # Plot single subject trajectory
    output_file = plot_single_subject_trajectory(df, subject_id, args.biomarker_name, args.output_dir)
    
    if output_file:
        print(f"\n✅ Single subject visualization complete!")
        print(f"📈 Plot saved: {output_file}")
        print(f"👤 Subject: {subject_id}")
        print(f"🧬 Biomarker: {args.biomarker_name}")
    else:
        print(f"\n❌ Failed to create visualization for subject: {subject_id}")

if __name__ == "__main__":
    main()
