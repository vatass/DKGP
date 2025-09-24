'''
Visualize Future Trajectory Predictions
Plot predicted biomarker trajectories over 8 years (every 12 months)
'''

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

class TrajectoryVisualizer:
    def __init__(self, results_dir="./future_inference_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("./trajectory_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define model names and colors
        self.model_info = {
            'hippocampus_roi17': {
                'name': 'Hippocampus (ROI 17)',
                'color': '#2E86AB',
                'unit': 'Normalized Volume'
            },
            'spare_ad': {
                'name': 'SPARE-AD Score',
                'color': '#A23B72',
                'unit': 'SPARE Score'
            },
            'spare_ef': {
                'name': 'SPARE-EF Score', 
                'color': '#F18F01',
                'unit': 'SPARE Score'
            },
            'mmse': {
                'name': 'MMSE Score',
                'color': '#C73E1D',
                'unit': 'MMSE Score'
            },
            'adas': {
                'name': 'ADAS Score',
                'color': '#6A994E',
                'unit': 'ADAS Score'
            }
        }
    
    def load_results(self, model_prefix):
        """Load inference results for a specific model"""
        csv_file = self.results_dir / f"{model_prefix}_future_inference.csv"
        json_file = self.results_dir / f"{model_prefix}_future_inference_summary.json"
        
        if not csv_file.exists():
            print(f"Warning: Results file not found: {csv_file}")
            return None, None
            
        df = pd.read_csv(csv_file)
        
        summary = None
        if json_file.exists():
            with open(json_file, 'r') as f:
                summary = json.load(f)
        
        return df, summary
    
    def plot_individual_trajectories(self, df, model_info, n_subjects=50):
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
                   alpha=0.3, linewidth=0.8, color=model_info['color'])
        
        # Plot mean trajectory with confidence interval
        mean_trajectory = df.groupby('time_months').agg({
            'predicted_value': ['mean', 'std'],
            'lower_bound': 'mean',
            'upper_bound': 'mean'
        }).reset_index()
        
        mean_trajectory.columns = ['time_months', 'mean_pred', 'std_pred', 'mean_lower', 'mean_upper']
        
        # Plot mean trajectory
        ax.plot(mean_trajectory['time_months'], mean_trajectory['mean_pred'], 
               linewidth=3, color=model_info['color'], label='Population Mean')
        
        # Plot confidence interval
        ax.fill_between(mean_trajectory['time_months'], 
                       mean_trajectory['mean_lower'], 
                       mean_trajectory['mean_upper'],
                       alpha=0.2, color=model_info['color'], label='95% Confidence Interval')
        
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel(f'{model_info["name"]} ({model_info["unit"]})', fontsize=12)
        ax.set_title(f'Predicted Trajectories: {model_info["name"]}\n8-Year Future Predictions', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add time point markers
        time_points = df['time_months'].unique()
        ax.set_xticks(time_points)
        ax.set_xticklabels([f'{int(t/12)}y' for t in time_points])
        
        plt.tight_layout()
        return fig
    
    def plot_population_summary(self, df, model_info):
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
                    color=model_info['color'], capsize=5)
        
        ax1.set_xlabel('Time (months)', fontsize=12)
        ax1.set_ylabel(f'{model_info["name"]} ({model_info["unit"]})', fontsize=12)
        ax1.set_title(f'Population Mean Trajectory: {model_info["name"]}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(summary_stats['time_months'])
        ax1.set_xticklabels([f'{int(t/12)}y' for t in summary_stats['time_months']])
        
        # Plot 2: Uncertainty over time
        ax2.plot(summary_stats['time_months'], summary_stats['mean_uncertainty'], 
                marker='s', linewidth=2, markersize=8, color=model_info['color'])
        
        ax2.set_xlabel('Time (months)', fontsize=12)
        ax2.set_ylabel('Mean Prediction Uncertainty', fontsize=12)
        ax2.set_title(f'Prediction Uncertainty Over Time: {model_info["name"]}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(summary_stats['time_months'])
        ax2.set_xticklabels([f'{int(t/12)}y' for t in summary_stats['time_months']])
        
        plt.tight_layout()
        return fig
    
    def plot_distribution_by_timepoint(self, df, model_info):
        """Plot distribution of predictions at each time point"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        time_points = sorted(df['time_months'].unique())
        
        for i, time_point in enumerate(time_points):
            if i >= len(axes):
                break
                
            time_data = df[df['time_months'] == time_point]['predicted_value']
            
            axes[i].hist(time_data, bins=30, alpha=0.7, color=model_info['color'], edgecolor='black')
            axes[i].axvline(time_data.mean(), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {time_data.mean():.3f}')
            axes[i].set_title(f'{int(time_point/12)} Years ({time_point} months)', fontweight='bold')
            axes[i].set_xlabel(f'{model_info["name"]} ({model_info["unit"]})')
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(time_points), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Distribution of Predictions by Time Point: {model_info["name"]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_combined_plot(self, all_results):
        """Create a combined plot showing all models"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for model_prefix, (df, summary) in all_results.items():
            if df is None:
                continue
                
            model_info = self.model_info.get(model_prefix, {
                'name': model_prefix.replace('_', ' ').title(),
                'color': '#666666',
                'unit': 'Score'
            })
            
            # Calculate mean trajectory
            mean_trajectory = df.groupby('time_months')['predicted_value'].mean().reset_index()
            
            ax.plot(mean_trajectory['time_months'], mean_trajectory['predicted_value'], 
                   marker='o', linewidth=2, markersize=6, label=model_info['name'],
                   color=model_info['color'])
        
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Predicted Value (Normalized)', fontsize=12)
        ax.set_title('Combined Future Trajectory Predictions\n8-Year Projections for All Biomarkers', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        time_points = [12, 24, 36, 48, 60, 72, 84, 96]
        ax.set_xticks(time_points)
        ax.set_xticklabels([f'{int(t/12)}y' for t in time_points])
        
        plt.tight_layout()
        return fig
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating trajectory visualizations...")
        
        # Find all available result files
        result_files = list(self.results_dir.glob("*_future_inference.csv"))
        available_models = [f.stem.replace('_future_inference', '') for f in result_files]
        
        print(f"Found results for: {available_models}")
        
        all_results = {}
        
        # Generate individual plots for each model
        for model_prefix in available_models:
            print(f"\nProcessing {model_prefix}...")
            
            df, summary = self.load_results(model_prefix)
            if df is None:
                continue
                
            all_results[model_prefix] = (df, summary)
            model_info = self.model_info.get(model_prefix, {
                'name': model_prefix.replace('_', ' ').title(),
                'color': '#666666',
                'unit': 'Score'
            })
            
            # Individual trajectories plot
            fig1 = self.plot_individual_trajectories(df, model_info)
            fig1.savefig(self.output_dir / f"{model_prefix}_individual_trajectories.png", 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Population summary plot
            fig2 = self.plot_population_summary(df, model_info)
            fig2.savefig(self.output_dir / f"{model_prefix}_population_summary.png", 
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # Distribution plot
            fig3 = self.plot_distribution_by_timepoint(df, model_info)
            fig3.savefig(self.output_dir / f"{model_prefix}_distributions.png", 
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            print(f"✅ Generated plots for {model_prefix}")
        
        # Combined plot
        if len(all_results) > 1:
            print("\nGenerating combined plot...")
            fig_combined = self.create_combined_plot(all_results)
            fig_combined.savefig(self.output_dir / "combined_trajectories.png", 
                               dpi=300, bbox_inches='tight')
            plt.close(fig_combined)
            print("✅ Generated combined plot")
        
        print(f"\n🎉 All visualizations saved to: {self.output_dir}")
        print(f"Generated {len(available_models) * 3 + (1 if len(all_results) > 1 else 0)} plots total")

def main():
    visualizer = TrajectoryVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()
