#!/usr/bin/env python3
"""
DKGP Trajectory Visualization and Validation Script
Creates publication-ready visualizations for inference validation and nichart
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_inference_data(csv_file):
    """Load inference results from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error loading {csv_file}: {e}")
        return None

def validate_inference_quality(df, biomarker_name):
    """Validate inference quality and print summary statistics"""
    print(f"\n=== Inference Quality Validation for {biomarker_name} ===")
    
    # Basic statistics
    n_subjects = df['PTID'].nunique()
    n_timepoints = df['time_months'].nunique()
    n_predictions = len(df)
    
    print(f"Subjects: {n_subjects}")
    print(f"Time points: {n_timepoints}")
    print(f"Total predictions: {n_predictions}")
    
    # Prediction statistics
    pred_stats = df['predicted_value'].describe()
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {pred_stats['mean']:.4f}")
    print(f"  Std: {pred_stats['std']:.4f}")
    print(f"  Min: {pred_stats['min']:.4f}")
    print(f"  Max: {pred_stats['max']:.4f}")
    
    # Uncertainty statistics
    if 'variance' in df.columns:
        uncertainty_stats = df['variance'].describe()
        print(f"\nUncertainty Statistics:")
        print(f"  Mean variance: {uncertainty_stats['mean']:.4f}")
        print(f"  Std variance: {uncertainty_stats['std']:.4f}")
    
    # Trajectory consistency check
    slopes = []
    for subject in df['PTID'].unique():
        subject_data = df[df['PTID'] == subject].sort_values('time_months')
        if len(subject_data) > 1:
            slope = np.polyfit(subject_data['time_months'], subject_data['predicted_value'], 1)[0]
            slopes.append(slope)
    
    if slopes:
        print(f"\nTrajectory Analysis:")
        print(f"  Mean slope: {np.mean(slopes):.4f}")
        print(f"  Slope std: {np.std(slopes):.4f}")
        print(f"  Positive slopes: {sum(1 for s in slopes if s > 0)}/{len(slopes)} ({100*sum(1 for s in slopes if s > 0)/len(slopes):.1f}%)")
    
    return {
        'n_subjects': n_subjects,
        'n_timepoints': n_timepoints,
        'n_predictions': n_predictions,
        'pred_mean': pred_stats['mean'],
        'pred_std': pred_stats['std'],
        'slopes': slopes
    }

def create_publication_trajectory_plot(df, output_dir, biomarker_name, figsize=(10, 6)):
    """Create publication-ready trajectory visualization"""
    
    # Create figure with publication quality
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique subjects and time points
    subjects = df['PTID'].unique()
    time_points = sorted(df['time_months'].unique())
    
    # Calculate population mean trajectory
    mean_traj = df.groupby('time_months')['predicted_value'].agg(['mean', 'std', 'count'])
    mean_traj['ci'] = 1.96 * mean_traj['std'] / np.sqrt(mean_traj['count'])
    
    # Plot individual trajectories (light, transparent)
    for subject in subjects[:100]:  # Limit for clarity
        subject_data = df[df['PTID'] == subject].sort_values('time_months')
        if len(subject_data) > 1:
            ax.plot(subject_data['time_months'], subject_data['predicted_value'], 
                   color='lightblue', alpha=0.3, linewidth=0.5)
    
    # Plot population mean trajectory (bold, prominent)
    ax.plot(mean_traj.index, mean_traj['mean'], 'o-', 
           color='darkblue', linewidth=3, markersize=8, 
           label='Population Mean', zorder=10)
    
    # Add confidence interval
    ax.fill_between(mean_traj.index, 
                   mean_traj['mean'] - mean_traj['ci'],
                   mean_traj['mean'] + mean_traj['ci'],
                   color='darkblue', alpha=0.2, zorder=5)
    
    # Customize plot
    ax.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{biomarker_name} Prediction', fontsize=14, fontweight='bold')
    ax.set_title(f'DKGP Trajectory Predictions: {biomarker_name}\n'
                f'({len(subjects)} subjects, {len(time_points)} time points)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add statistics text box
    stats_text = f'Subjects: {len(subjects)}\nTime points: {len(time_points)}\n'
    stats_text += f'Predictions: {len(df)}\nMean slope: {np.mean([np.polyfit(df[df["PTID"]==s].sort_values("time_months")["time_months"], df[df["PTID"]==s].sort_values("time_months")["predicted_value"], 1)[0] for s in subjects[:10]]) if len(subjects) > 0 else 0:.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save high-quality plot
    output_file = os.path.join(output_dir, f'{biomarker_name}_trajectory_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved trajectory plot: {output_file}")
    
    plt.show()
    return output_file

def create_uncertainty_validation_plot(df, output_dir, biomarker_name, figsize=(12, 5)):
    """Create uncertainty validation visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Uncertainty over time
    if 'variance' in df.columns:
        uncertainty_by_time = df.groupby('time_months')['variance'].agg(['mean', 'std'])
        
        ax1.plot(uncertainty_by_time.index, uncertainty_by_time['mean'], 'o-', 
                linewidth=2, markersize=6, color='red')
        ax1.fill_between(uncertainty_by_time.index,
                        uncertainty_by_time['mean'] - uncertainty_by_time['std'],
                        uncertainty_by_time['mean'] + uncertainty_by_time['std'],
                        alpha=0.3, color='red')
        
        ax1.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Model Uncertainty (Variance)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Uncertainty Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction vs Uncertainty scatter
    if 'variance' in df.columns:
        scatter = ax2.scatter(df['predicted_value'], df['variance'], 
                            alpha=0.6, s=20, c=df['time_months'], 
                            cmap='viridis')
        ax2.set_xlabel('Predicted Value', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Model Uncertainty (Variance)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction vs Uncertainty', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (months)', fontsize=10)
    
    plt.suptitle(f'Uncertainty Validation: {biomarker_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{biomarker_name}_uncertainty_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved uncertainty plot: {output_file}")
    
    plt.show()
    return output_file

def create_trajectory_diversity_plot(df, output_dir, biomarker_name, figsize=(10, 6)):
    """Create visualization showing trajectory diversity"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate trajectory slopes
    slopes = []
    intercepts = []
    subject_ids = []
    
    for subject in df['PTID'].unique():
        subject_data = df[df['PTID'] == subject].sort_values('time_months')
        if len(subject_data) > 1:
            slope, intercept = np.polyfit(subject_data['time_months'], subject_data['predicted_value'], 1)
            slopes.append(slope)
            intercepts.append(intercept)
            subject_ids.append(subject)
    
    # Create slope distribution plot
    ax.hist(slopes, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    
    ax.axvline(mean_slope, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_slope:.3f}')
    ax.axvline(mean_slope + std_slope, color='orange', linestyle=':', linewidth=2,
              label=f'+1σ: {mean_slope + std_slope:.3f}')
    ax.axvline(mean_slope - std_slope, color='orange', linestyle=':', linewidth=2,
              label=f'-1σ: {mean_slope - std_slope:.3f}')
    
    # Customize plot
    ax.set_xlabel('Trajectory Slope (change per month)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=14, fontweight='bold')
    ax.set_title(f'Trajectory Diversity: {biomarker_name}\n'
                f'Distribution of Individual Trajectory Slopes', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add statistics text
    stats_text = f'Total subjects: {len(slopes)}\n'
    stats_text += f'Mean slope: {mean_slope:.4f}\n'
    stats_text += f'Std slope: {std_slope:.4f}\n'
    stats_text += f'Positive slopes: {sum(1 for s in slopes if s > 0)} ({100*sum(1 for s in slopes if s > 0)/len(slopes):.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{biomarker_name}_trajectory_diversity.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved diversity plot: {output_file}")
    
    plt.show()
    return output_file

def create_validation_summary_plot(df, output_dir, biomarker_name, figsize=(15, 10)):
    """Create comprehensive validation summary"""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'DKGP Inference Validation Summary: {biomarker_name}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Prediction distribution by time point
    ax1 = axes[0, 0]
    time_points = sorted(df['time_months'].unique())
    for i, tp in enumerate(time_points):
        tp_data = df[df['time_months'] == tp]['predicted_value']
        ax1.hist(tp_data, alpha=0.6, bins=20, label=f'{tp}m', density=True)
    ax1.set_xlabel('Predicted Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Prediction Distributions by Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean trajectory with confidence intervals
    ax2 = axes[0, 1]
    mean_traj = df.groupby('time_months')['predicted_value'].agg(['mean', 'std', 'count'])
    mean_traj['ci'] = 1.96 * mean_traj['std'] / np.sqrt(mean_traj['count'])
    
    ax2.plot(mean_traj.index, mean_traj['mean'], 'o-', linewidth=2, markersize=6, color='darkblue')
    ax2.fill_between(mean_traj.index, 
                    mean_traj['mean'] - mean_traj['ci'],
                    mean_traj['mean'] + mean_traj['ci'],
                    alpha=0.3, color='darkblue')
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Mean Predicted Value')
    ax2.set_title('Population Mean Trajectory')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trajectory slopes distribution
    ax3 = axes[0, 2]
    slopes = []
    for subject in df['PTID'].unique():
        subject_data = df[df['PTID'] == subject].sort_values('time_months')
        if len(subject_data) > 1:
            slope = np.polyfit(subject_data['time_months'], subject_data['predicted_value'], 1)[0]
            slopes.append(slope)
    
    ax3.hist(slopes, bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(slopes), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(slopes):.3f}')
    ax3.set_xlabel('Trajectory Slope')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Trajectory Slope Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty over time
    ax4 = axes[1, 0]
    if 'variance' in df.columns:
        uncertainty = df.groupby('time_months')['variance'].mean()
        ax4.plot(uncertainty.index, uncertainty.values, 'o-', linewidth=2, markersize=6, color='red')
        ax4.set_xlabel('Time (months)')
        ax4.set_ylabel('Mean Variance')
        ax4.set_title('Model Uncertainty Over Time')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Variance data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Uncertainty Over Time')
    
    # 5. Subject count by time point
    ax5 = axes[1, 1]
    subject_counts = df.groupby('time_months')['PTID'].nunique()
    bars = ax5.bar(subject_counts.index, subject_counts.values, alpha=0.7, color='orange')
    ax5.set_xlabel('Time (months)')
    ax5.set_ylabel('Number of Subjects')
    ax5.set_title('Subject Count by Time Point')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # 6. Time point correlations
    ax6 = axes[1, 2]
    pivot_data = df.pivot_table(index='PTID', columns='time_months', values='predicted_value')
    correlation_matrix = pivot_data.corr()
    
    im = ax6.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax6.set_xticks(range(len(correlation_matrix.columns)))
    ax6.set_yticks(range(len(correlation_matrix.index)))
    ax6.set_xticklabels(correlation_matrix.columns)
    ax6.set_yticklabels(correlation_matrix.index)
    ax6.set_title('Time Point Correlations')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Correlation')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{biomarker_name}_validation_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved validation summary: {output_file}")
    
    plt.show()
    return output_file

def main():
    parser = argparse.ArgumentParser(description='DKGP Trajectory Visualization and Validation')
    parser.add_argument('--csv_file', required=True, help='Path to inference results CSV file')
    parser.add_argument('--output_dir', default='./validation_plots', help='Output directory for plots')
    parser.add_argument('--biomarker_name', help='Name of biomarker (auto-detected from filename if not provided)')
    parser.add_argument('--plot_type', choices=['all', 'trajectory', 'uncertainty', 'diversity', 'summary'], 
                       default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    # Auto-detect biomarker name from filename
    if args.biomarker_name is None:
        filename = Path(args.csv_file).stem
        args.biomarker_name = filename.replace('_output', '').replace('_', ' ').title()
    
    print(f"🔬 DKGP Trajectory Validation and Visualization")
    print(f"📊 Biomarker: {args.biomarker_name}")
    print(f"📁 Input file: {args.csv_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Load data
    df = load_inference_data(args.csv_file)
    if df is None:
        return
    
    # Validate inference quality
    validation_stats = validate_inference_quality(df, args.biomarker_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots based on type
    plots_created = []
    
    if args.plot_type in ['all', 'trajectory']:
        plots_created.append(create_publication_trajectory_plot(df, args.output_dir, args.biomarker_name))
    
    if args.plot_type in ['all', 'uncertainty']:
        plots_created.append(create_uncertainty_validation_plot(df, args.output_dir, args.biomarker_name))
    
    if args.plot_type in ['all', 'diversity']:
        plots_created.append(create_trajectory_diversity_plot(df, args.output_dir, args.biomarker_name))
    
    if args.plot_type in ['all', 'summary']:
        plots_created.append(create_validation_summary_plot(df, args.output_dir, args.biomarker_name))
    
    # Print summary
    print(f"\n✅ Validation and visualization complete!")
    print(f"📈 Plots created: {len(plots_created)}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Validation summary:")
    print(f"   - Subjects: {validation_stats['n_subjects']}")
    print(f"   - Time points: {validation_stats['n_timepoints']}")
    print(f"   - Total predictions: {validation_stats['n_predictions']}")
    print(f"   - Mean prediction: {validation_stats['pred_mean']:.4f}")
    print(f"   - Prediction std: {validation_stats['pred_std']:.4f}")
    
    if validation_stats['slopes']:
        print(f"   - Mean trajectory slope: {np.mean(validation_stats['slopes']):.4f}")
        print(f"   - Slope std: {np.std(validation_stats['slopes']):.4f}")

if __name__ == "__main__":
    main()
