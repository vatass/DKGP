#!/usr/bin/env python3
"""
Comprehensive Hippocampus Trajectory Analysis
Analyzes and visualizes hippocampus biomarker predictions with confidence bounds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class HippocampusAnalyzer:
    def __init__(self, results_file="./future_inference_results/hippocampus_roi17_future_inference.csv"):
        self.results_file = results_file
        self.output_dir = Path("./hippocampus_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(results_file)
        print(f"Loaded {len(self.df)} predictions for {self.df['PTID'].nunique()} subjects")
        
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        
        # 1. Summary statistics
        self.print_summary_statistics()
        
        # 2. Individual trajectory plots
        self.plot_individual_trajectories()
        
        # 3. Population summary
        self.plot_population_summary()
        
        # 4. Uncertainty analysis
        self.plot_uncertainty_analysis()
        
        # 5. Distribution analysis
        self.plot_distribution_analysis()
        
        # 6. Save detailed CSV with analysis
        self.save_analysis_csv()
        
        print(f"\n🎉 All analyses saved to: {self.output_dir}")
        
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        print("\n" + "="*60)
        print("HIPPOCAMPUS TRAJECTORY PREDICTION SUMMARY")
        print("="*60)
        
        n_subjects = self.df['PTID'].nunique()
        n_predictions = len(self.df)
        timepoints = sorted(self.df['time_months'].unique())
        
        print(f"Number of subjects: {n_subjects}")
        print(f"Number of predictions: {n_predictions}")
        print(f"Time points: {timepoints} months ({[f'{t//12}y' for t in timepoints]})")
        print(f"ROI Index: {self.df['roi_idx'].iloc[0]}")
        
        print("\nPrediction Statistics by Time Point:")
        print("-" * 50)
        
        for timepoint in timepoints:
            tp_data = self.df[self.df['time_months'] == timepoint]
            mean_pred = tp_data['predicted_value'].mean()
            std_pred = tp_data['predicted_value'].std()
            mean_uncertainty = tp_data['interval_width'].mean()
            
            print(f"{timepoint:3d} months ({timepoint//12:1d}y): "
                  f"Mean={mean_pred:7.4f} ± {std_pred:6.4f}, "
                  f"Uncertainty={mean_uncertainty:6.4f}")
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"Mean prediction: {self.df['predicted_value'].mean():.4f}")
        print(f"Std prediction: {self.df['predicted_value'].std():.4f}")
        print(f"Mean uncertainty: {self.df['interval_width'].mean():.4f}")
        print(f"Min uncertainty: {self.df['interval_width'].min():.4f}")
        print(f"Max uncertainty: {self.df['interval_width'].max():.4f}")
        
    def plot_individual_trajectories(self, n_subjects=20):
        """Plot individual subject trajectories with confidence bounds"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sample subjects for visualization
        unique_subjects = self.df['PTID'].unique()
        if len(unique_subjects) > n_subjects:
            sampled_subjects = np.random.choice(unique_subjects, n_subjects, replace=False)
        else:
            sampled_subjects = unique_subjects
        
        # Plot individual trajectories
        for i, subject in enumerate(sampled_subjects):
            subject_data = self.df[self.df['PTID'] == subject].sort_values('time_months')
            
            # Plot prediction line
            ax.plot(subject_data['time_months'], subject_data['predicted_value'], 
                   alpha=0.6, linewidth=1.5, color=f'C{i%10}')
            
            # Plot confidence bounds
            ax.fill_between(subject_data['time_months'], 
                           subject_data['lower_bound'], 
                           subject_data['upper_bound'],
                           alpha=0.2, color=f'C{i%10}')
        
        # Plot population mean trajectory
        mean_trajectory = self.df.groupby('time_months').agg({
            'predicted_value': 'mean',
            'lower_bound': 'mean',
            'upper_bound': 'mean'
        }).reset_index()
        
        ax.plot(mean_trajectory['time_months'], mean_trajectory['predicted_value'], 
               linewidth=4, color='red', label='Population Mean', zorder=10)
        
        ax.fill_between(mean_trajectory['time_months'], 
                       mean_trajectory['lower_bound'], 
                       mean_trajectory['upper_bound'],
                       alpha=0.3, color='red', label='Population Bounds', zorder=5)
        
        ax.set_xlabel('Time from Baseline (months)', fontsize=14)
        ax.set_ylabel('Hippocampus Volume (Normalized)', fontsize=14)
        ax.set_title(f'Hippocampus Trajectory Predictions\n{len(sampled_subjects)} Individual Subjects + Population Mean', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        timepoints = sorted(self.df['time_months'].unique())
        ax.set_xticks(timepoints)
        ax.set_xticklabels([f'{int(t/12)}y' for t in timepoints])
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "hippocampus_individual_trajectories.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved individual trajectories: {output_file}")

def main():
    """Main function to run hippocampus analysis"""
    print("Starting Hippocampus Trajectory Analysis...")
    
    analyzer = HippocampusAnalyzer()
    analyzer.generate_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
