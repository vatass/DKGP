#!/usr/bin/env python3
'''
Comprehensive Evaluation of All Biomarkers
- Quantitative evaluation of all ROIs, spare scores, and cognitive scores
- Special focus on hippocampus (ROI 17)
- Trajectory plots for hippocampus and other biomarkers
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress matplotlib font warnings specifically
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BiomarkerEvaluator:
    def __init__(self, hmuse_version=0):
        self.hmuse_version = hmuse_version
        self.results_dir = Path(".")
        self.output_dir = Path(f"evaluation_results_hmuse{hmuse_version}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define directories
        self.roi_results_dir = Path(f"inference_results_rois")
        self.spare_results_dir = Path(f"inference_results_spare")
        self.cognitive_results_dir = Path(f"inference_results_cognitive")
        
        # Load scanner-stable subjects
        self.scanner_stable_subjects = self.load_scanner_stable_subjects()
        
        # ROI names for hippocampus and other important regions
        self.roi_names = {
            17: "Hippocampus",
            0: "ROI_0",
            1: "ROI_1",
            # Add more ROI names as needed
        }
        
        # Biomarker categories
        self.biomarker_categories = {
            'rois': {'name': 'ROIs', 'count': 145},
            'spare_scores': {'name': 'Spare Scores', 'count': 2},
            'cognitive_scores': {'name': 'Cognitive Scores', 'count': 2}
        }
        
        # Colors for different biomarker types
        self.colors = {
            'rois': '#1f77b4',
            'spare_scores': '#ff7f0e', 
            'cognitive_scores': '#2ca02c',
            'hippocampus': '#d62728'
        }
    
    def load_scanner_stable_subjects(self):
        """Load scanner-stable subjects from longitudinal covariates file"""
        print("Loading scanner-stable subjects...")
        try:
            import pandas as pd
            df = pd.read_csv('../LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
            scanner_counts = df.groupby('PTID')['MRI_Scanner_Model'].nunique()
            stable_subjects = scanner_counts[scanner_counts == 1].index.tolist()
            print(f"Loaded {len(stable_subjects)} scanner-stable subjects out of {len(scanner_counts)} total subjects")
            return set(stable_subjects)
        except Exception as e:
            print(f"Warning: Could not load scanner-stable subjects: {e}")
            print("Proceeding with all subjects...")
            return set()
        
    def load_all_results(self):
        """Load all inference results"""
        print("Loading all inference results...")
        
        self.roi_results = {}
        self.spare_results = {}
        self.cognitive_results = {}
        
        # Load ROI results
        if self.roi_results_dir.exists():
            for roi_idx in range(145):
                result_file = self.roi_results_dir / f"inference_results_roi_{roi_idx}.csv"
                if result_file.exists():
                    df = pd.read_csv(result_file)
                    # Filter for scanner-stable subjects if available
                    if self.scanner_stable_subjects:
                        df = df[df['PTID'].isin(self.scanner_stable_subjects)]
                        print(f"Loaded ROI {roi_idx}: {len(df)} samples (scanner-stable)")
                    else:
                        print(f"Loaded ROI {roi_idx}: {len(df)} samples")
                    self.roi_results[roi_idx] = df
        
        # Load spare score results
        if self.spare_results_dir.exists():
            for score in ['spare_ad', 'spare_ef']:
                result_file = self.spare_results_dir / f"inference_results_{score}.csv"
                if result_file.exists():
                    df = pd.read_csv(result_file)
                    # Filter for scanner-stable subjects if available
                    if self.scanner_stable_subjects:
                        df = df[df['PTID'].isin(self.scanner_stable_subjects)]
                        print(f"Loaded {score}: {len(df)} samples (scanner-stable)")
                    else:
                        print(f"Loaded {score}: {len(df)} samples")
                    self.spare_results[score] = df
        
        # Load cognitive score results
        if self.cognitive_results_dir.exists():
            for score in ['adas', 'mmse']:
                result_file = self.cognitive_results_dir / f"inference_results_{score}.csv"
                if result_file.exists():
                    df = pd.read_csv(result_file)
                    # Filter for scanner-stable subjects if available
                    if self.scanner_stable_subjects:
                        df = df[df['PTID'].isin(self.scanner_stable_subjects)]
                        print(f"Loaded {score}: {len(df)} samples (scanner-stable)")
                    else:
                        print(f"Loaded {score}: {len(df)} samples")
                    self.cognitive_results[score] = df
        
        print(f"Loaded {len(self.roi_results)} ROIs, {len(self.spare_results)} spare scores, {len(self.cognitive_results)} cognitive scores")
    
    def calculate_overall_metrics(self):
        """Calculate overall performance metrics"""
        print("Calculating overall metrics...")
        
        self.overall_metrics = {}
        
        # ROI metrics
        roi_metrics = []
        for roi_idx, df in self.roi_results.items():
            metrics = {
                'roi_idx': roi_idx,
                'mae': df['mae'].iloc[0],
                'mse': df['mse'].iloc[0],
                'rmse': df['rmse'].iloc[0],
                'r2': df['r2'].iloc[0],
                'coverage': df['mean_coverage'].iloc[0],
                'interval_width': df['mean_interval_width'].iloc[0],
                'n_samples': len(df)
            }
            roi_metrics.append(metrics)
        
        self.overall_metrics['rois'] = pd.DataFrame(roi_metrics)
        
        # Spare score metrics
        spare_metrics = []
        for score, df in self.spare_results.items():
            metrics = {
                'score': score,
                'mae': df['mae'].iloc[0],
                'mse': df['mse'].iloc[0],
                'rmse': df['rmse'].iloc[0],
                'r2': df['r2'].iloc[0],
                'coverage': df['mean_coverage'].iloc[0],
                'interval_width': df['mean_interval_width'].iloc[0],
                'n_samples': len(df)
            }
            spare_metrics.append(metrics)
        
        self.overall_metrics['spare_scores'] = pd.DataFrame(spare_metrics)
        
        # Cognitive score metrics
        cognitive_metrics = []
        for score, df in self.cognitive_results.items():
            metrics = {
                'score': score,
                'mae': df['mae'].iloc[0],
                'mse': df['mse'].iloc[0],
                'rmse': df['rmse'].iloc[0],
                'r2': df['r2'].iloc[0],
                'coverage': df['mean_coverage'].iloc[0],
                'interval_width': df['mean_interval_width'].iloc[0],
                'n_samples': len(df)
            }
            cognitive_metrics.append(metrics)
        
        self.overall_metrics['cognitive_scores'] = pd.DataFrame(cognitive_metrics)
        
        # Save overall metrics
        for category, df in self.overall_metrics.items():
            df.to_csv(self.output_dir / f"{category}_overall_metrics.csv", index=False)
    
    def create_performance_summary(self):
        """Create performance summary plots - Nature Aging publication quality"""
        print("Creating performance summary plots...")
        
        # Set up publication-quality plotting parameters for Nature Aging
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.linewidth': 0.8,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'Biomarker Performance Summary (HMUSE {self.hmuse_version})', 
                    fontsize=12, y=0.95)
        
        # R² scores
        ax = axes[0, 0]
        categories = ['rois', 'spare_scores', 'cognitive_scores']
        r2_data = []
        labels = []
        
        for cat in categories:
            if cat in self.overall_metrics and not self.overall_metrics[cat].empty:
                r2_values = self.overall_metrics[cat]['r2'].values
                r2_data.extend(r2_values)
                labels.extend([self.biomarker_categories[cat]['name']] * len(r2_values))
        
        if r2_data:
            df_r2 = pd.DataFrame({'R²': r2_data, 'Category': labels})
            sns.boxplot(data=df_r2, x='Category', y='R²', ax=ax)
            ax.set_title('R² Score Distribution', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # MAE scores
        ax = axes[0, 1]
        mae_data = []
        labels = []
        
        for cat in categories:
            if cat in self.overall_metrics and not self.overall_metrics[cat].empty:
                mae_values = self.overall_metrics[cat]['mae'].values
                mae_data.extend(mae_values)
                labels.extend([self.biomarker_categories[cat]['name']] * len(mae_values))
        
        if mae_data:
            df_mae = pd.DataFrame({'MAE': mae_data, 'Category': labels})
            sns.boxplot(data=df_mae, x='Category', y='MAE', ax=ax)
            ax.set_title('MAE Distribution', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Coverage
        ax = axes[0, 2]
        coverage_data = []
        labels = []
        
        for cat in categories:
            if cat in self.overall_metrics and not self.overall_metrics[cat].empty:
                coverage_values = self.overall_metrics[cat]['coverage'].values
                coverage_data.extend(coverage_values)
                labels.extend([self.biomarker_categories[cat]['name']] * len(coverage_values))
        
        if coverage_data:
            df_coverage = pd.DataFrame({'Coverage': coverage_data, 'Category': labels})
            sns.boxplot(data=df_coverage, x='Category', y='Coverage', ax=ax)
            ax.set_title('Coverage Distribution', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # ROI R² distribution
        ax = axes[1, 0]
        if 'rois' in self.overall_metrics and not self.overall_metrics['rois'].empty:
            roi_df = self.overall_metrics['rois']
            ax.hist(roi_df['r2'], bins=20, alpha=0.7, color=self.colors['rois'])
            ax.axvline(roi_df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {roi_df["r2"].mean():.3f}')
            ax.set_title('ROI R² Distribution')
            ax.set_xlabel('R²')
            ax.set_ylabel('Count')
            ax.legend()
        
        # Top performing ROIs
        ax = axes[1, 1]
        if 'rois' in self.overall_metrics and not self.overall_metrics['rois'].empty:
            top_rois = self.overall_metrics['rois'].nlargest(10, 'r2')
            ax.barh(range(len(top_rois)), top_rois['r2'], color=self.colors['rois'])
            ax.set_yticks(range(len(top_rois)))
            ax.set_yticklabels([f'ROI {roi}' for roi in top_rois['roi_idx']])
            ax.set_title('Top 10 ROIs by R²')
            ax.set_xlabel('R²')
        
        # Biomarker comparison
        ax = axes[1, 2]
        comparison_data = []
        for cat in categories:
            if cat in self.overall_metrics and not self.overall_metrics[cat].empty:
                df = self.overall_metrics[cat]
                comparison_data.append({
                    'Category': self.biomarker_categories[cat]['name'],
                    'Mean R²': df['r2'].mean(),
                    'Mean MAE': df['mae'].mean(),
                    'Mean Coverage': df['coverage'].mean()
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            comp_df.plot(x='Category', y=['Mean R²', 'Mean MAE', 'Mean Coverage'], 
                        kind='bar', ax=ax, width=0.8)
            ax.set_title('Biomarker Category Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_hippocampus(self):
        """Special analysis for hippocampus (ROI 17)"""
        print("Analyzing hippocampus (ROI 17)...")
        
        if 17 not in self.roi_results:
            print("Hippocampus (ROI 17) results not found!")
            return
        
        hippocampus_df = self.roi_results[17]
        
        # Create hippocampus-specific plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hippocampus (ROI 17) Analysis', fontsize=16)
        
        # True vs Predicted
        ax = axes[0, 0]
        ax.scatter(hippocampus_df['true_value'], hippocampus_df['predicted_value'], 
                  alpha=0.6, color=self.colors['hippocampus'])
        ax.plot([hippocampus_df['true_value'].min(), hippocampus_df['true_value'].max()], 
                [hippocampus_df['true_value'].min(), hippocampus_df['true_value'].max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Hippocampus: True vs Predicted')
        ax.legend()
        
        # Residuals
        ax = axes[0, 1]
        residuals = hippocampus_df['predicted_value'] - hippocampus_df['true_value']
        ax.scatter(hippocampus_df['predicted_value'], residuals, alpha=0.6, color=self.colors['hippocampus'])
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residuals')
        ax.set_title('Hippocampus: Residuals')
        
        # Prediction intervals
        ax = axes[1, 0]
        sorted_idx = np.argsort(hippocampus_df['true_value'])
        true_sorted = hippocampus_df['true_value'].iloc[sorted_idx]
        pred_sorted = hippocampus_df['predicted_value'].iloc[sorted_idx]
        lower_sorted = hippocampus_df['lower_bound'].iloc[sorted_idx]
        upper_sorted = hippocampus_df['upper_bound'].iloc[sorted_idx]
        
        ax.fill_between(range(len(true_sorted)), lower_sorted, upper_sorted, 
                       alpha=0.3, color=self.colors['hippocampus'], label='95% CI')
        ax.plot(range(len(true_sorted)), true_sorted, 'o-', label='True', alpha=0.7)
        ax.plot(range(len(true_sorted)), pred_sorted, 's-', label='Predicted', alpha=0.7)
        ax.set_xlabel('Sample Index (sorted by true value)')
        ax.set_ylabel('Value')
        ax.set_title('Hippocampus: Predictions with Confidence Intervals')
        ax.legend()
        
        # Metrics summary
        ax = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'R²', 'Coverage']
        values = [
            hippocampus_df['mae'].iloc[0],
            hippocampus_df['rmse'].iloc[0],
            hippocampus_df['r2'].iloc[0],
            hippocampus_df['mean_coverage'].iloc[0]
        ]
        
        bars = ax.bar(metrics, values, color=self.colors['hippocampus'])
        ax.set_title('Hippocampus Performance Metrics')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hippocampus_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save hippocampus metrics
        hippocampus_metrics = {
            'roi_idx': 17,
            'roi_name': 'Hippocampus',
            'mae': float(hippocampus_df['mae'].iloc[0]),
            'mse': float(hippocampus_df['mse'].iloc[0]),
            'rmse': float(hippocampus_df['rmse'].iloc[0]),
            'r2': float(hippocampus_df['r2'].iloc[0]),
            'coverage': float(hippocampus_df['mean_coverage'].iloc[0]),
            'interval_width': float(hippocampus_df['mean_interval_width'].iloc[0]),
            'n_samples': len(hippocampus_df)
        }
        
        with open(self.output_dir / 'hippocampus_metrics.json', 'w') as f:
            json.dump(hippocampus_metrics, f, indent=2)
    
    def create_trajectory_plots(self, n_subjects=5):
        """Create trajectory plots for selected subjects focusing on baseline predictors (population model)"""
        print(f"Creating trajectory plots for {n_subjects} subjects...")
        
        # Set publication-quality figure parameters for Nature Aging
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.linewidth': 0.8,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        # Define biomarkers to plot
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)'),
            ('spare_ad', 'SPARE-AD'),
            ('spare_ef', 'SPARE-EF'),
            ('adas', 'ADAS'),
            ('mmse', 'MMSE')
        ]
        
        # Get subjects from hippocampus data (ROI 17)
        if 17 not in self.roi_results:
            print("Hippocampus data not available for trajectory plots!")
            return
        
        hippocampus_df = self.roi_results[17]
        unique_subjects = hippocampus_df['PTID'].unique()[:n_subjects]
        
        # Create trajectory plots for each biomarker
        for biomarker_key, biomarker_name in biomarkers:
            print(f"Creating trajectory plot for {biomarker_name}...")
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = hippocampus_df
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                print(f"Data not available for {biomarker_name}, skipping...")
                continue
            
            # Create figure for this biomarker
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot trajectories for each subject
            for i, subject in enumerate(unique_subjects):
                subject_data = df[df['PTID'] == subject]
                
                if len(subject_data) <= 1:  # Skip if only one timepoint
                    continue
                
                # Sort by time (months from baseline)
                subject_data = subject_data.sort_values('time')
                
                # Plot ground truth as scatter points
                ax.scatter(subject_data['time'], subject_data['true_value'], 
                          color='black', s=50, zorder=5, 
                          marker='o', edgecolor='black', linewidth=1, alpha=0.7,
                          label='Ground Truth' if i == 0 else "")
                
                # Plot population trajectory (mean prediction)
                ax.plot(subject_data['time'], subject_data['predicted_value'], 
                       linestyle='--', color='black', linewidth=2, alpha=0.7,
                       label='Population Trajectory' if i == 0 else "")
                
                # Plot confidence bounds
                ax.fill_between(subject_data['time'], 
                              subject_data['lower_bound'], 
                              subject_data['upper_bound'],
                              color='black', alpha=0.1,
                              label='Population Bounds' if i == 0 else "")
            
            # Customize the plot for Nature Aging publication
            ax.set_title(f'{biomarker_name} Population Trajectories\n{len(unique_subjects)} Test Subjects', 
                        fontsize=11, pad=15)
            ax.set_xlabel('Time from baseline (months)', fontsize=12, labelpad=8)
            ax.set_ylabel(f'{biomarker_name}', fontsize=12, labelpad=8)
            
            # Add legend with improved styling
            ax.legend(['Ground Truth', 'Population Trajectory', 'Population Bounds'],
                     frameon=True, framealpha=0.95, 
                     edgecolor='black', fancybox=False,
                     loc='upper right', fontsize=8)
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add subtle grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'trajectory_plot_{biomarker_key}.png'
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved trajectory plot: {plot_filename}")
        
        # Create a combined summary plot
        self.create_combined_trajectory_summary(unique_subjects)
    
    def create_combined_trajectory_summary(self, subjects):
        """Create a combined summary of trajectory plots"""
        print("Creating combined trajectory summary...")
        
        # Create a single figure with subplots for each biomarker
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Biomarker Population Trajectories Summary', fontsize=12, y=0.95)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Define biomarkers
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 0),
            ('spare_ad', 'SPARE-AD', 1),
            ('spare_ef', 'SPARE-EF', 2),
            ('adas', 'ADAS', 3),
            ('mmse', 'MMSE', 4)
        ]
        
        for biomarker_key, biomarker_name, idx in biomarkers:
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results[17]
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            # Plot trajectories for each subject
            for subject in subjects:
                subject_data = df[df['PTID'] == subject]
                
                if len(subject_data) <= 1:
                    continue
                
                subject_data = subject_data.sort_values('time')
                
                # Plot ground truth and predictions
                ax.scatter(subject_data['time'], subject_data['true_value'], 
                          color='black', s=30, zorder=5, alpha=0.6)
                ax.plot(subject_data['time'], subject_data['predicted_value'], 
                       linestyle='--', color='black', linewidth=1.5, alpha=0.6)
                ax.fill_between(subject_data['time'], 
                              subject_data['lower_bound'], 
                              subject_data['upper_bound'],
                              color='black', alpha=0.05)
            
            ax.set_title(f'{biomarker_name}', fontsize=12, pad=8)
            ax.set_xlabel('Time from baseline (months)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add subtle grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Hide unused subplots
        for idx in range(len(biomarkers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectory_plots_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved combined trajectory summary plot")
    
    def create_error_plots_with_time(self):
        """Create error plots with time for each biomarker - Nature Aging publication quality"""
        print("Creating error plots with time for each biomarker...")
        
        # Set up publication-quality plotting parameters for Nature Aging
        mpl.rcParams.update({
            'font.size': 9,
            'font.family': 'DejaVu Sans',
            'font.sans-serif': ['DejaVu Sans', 'Arial'],
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        # Define biomarkers to plot
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 'ROI'),
            ('spare_ad', 'SPARE-AD', 'Spare Score'),
            ('spare_ef', 'SPARE-EF', 'Spare Score'),
            ('adas', 'ADAS', 'Cognitive Score'),
            ('mmse', 'MMSE', 'Cognitive Score')
        ]
        
        for biomarker_key, biomarker_name, biomarker_type in biomarkers:
            print(f"Creating error plot for {biomarker_name}...")
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results[17]
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            # Calculate absolute error
            df['absolute_error'] = np.abs(df['true_value'] - df['predicted_value'])
            
            # Print interpretable statistics
            print(f"\n=== {biomarker_name} Time Analysis ===")
            print(f"Total samples: {len(df)}")
            print(f"Time range: {df['time'].min():.1f} - {df['time'].max():.1f} months")
            
            # Time correlation
            correlation = np.corrcoef(df['time'], df['absolute_error'])[0, 1]
            print(f"Time-Error Correlation: {correlation:.4f}")
            
            # Time bin statistics
            time_bins = pd.cut(df['time'], bins=5, labels=False)
            df['time_bin'] = time_bins
            time_bin_stats = df.groupby('time_bin')['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nTime Bin Error Statistics:")
            for time_bin, stats in time_bin_stats.iterrows():
                time_range = df[df['time_bin'] == time_bin]['time']
                print(f"  Time Bin {time_bin}: {time_range.min():.1f}-{time_range.max():.1f} months: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Create figure with publication-quality settings for Nature Aging
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f'{biomarker_name} Error Analysis Over Time', 
                        fontsize=11, y=0.95)
            
            # Plot 1: Absolute Error vs Time
            ax1.scatter(df['time'], df['absolute_error'], 
                       alpha=0.6, s=50, color='#1f77b4', edgecolor='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(df['time'], df['absolute_error'], 1)
            p = np.poly1d(z)
            ax1.plot(df['time'], p(df['time']), "r--", linewidth=2, alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(df['time'], df['absolute_error'])[0, 1]
            
            ax1.set_xlabel('Time from baseline (months)', fontsize=12)
            ax1.set_ylabel('Absolute Error', fontsize=12)
            ax1.set_title(f'Absolute Error vs Time\nCorrelation: {correlation:.3f}', 
                         fontsize=12)
            
            # Add grid
            ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Plot 2: Error Distribution by Time Bins
            # Create time bins
            time_bins = pd.cut(df['time'], bins=5, labels=False)
            df['time_bin'] = time_bins
            
            # Calculate mean error for each bin
            error_by_bin = df.groupby('time_bin')['absolute_error'].agg(['mean', 'std', 'count']).reset_index()
            
            # Create bar plot
            bars = ax2.bar(error_by_bin['time_bin'], error_by_bin['mean'], 
                          yerr=error_by_bin['std'], capsize=5,
                          color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add sample size annotations
            for i, (bar, count) in enumerate(zip(bars, error_by_bin['count'])):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + error_by_bin['std'].iloc[i] + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=12)
            
            ax2.set_xlabel('Time Bin', fontsize=12)
            ax2.set_ylabel('Mean Absolute Error', fontsize=12)
            ax2.set_title('Error Distribution by Time Period', fontsize=12)
            
            # Set x-axis labels
            bin_centers = df.groupby('time_bin')['time'].mean()
            ax2.set_xticks(range(len(bin_centers)))
            ax2.set_xticklabels([f'{center:.1f}±{df[df["time_bin"]==i]["time"].std():.1f}' 
                                for i, center in enumerate(bin_centers)], rotation=45)
            
            # Add grid
            ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add statistics text box
            stats_text = f'Total Samples: {len(df)}\nMean Error: {df["absolute_error"].mean():.3f}\nStd Error: {df["absolute_error"].std():.3f}\nMax Error: {df["absolute_error"].max():.3f}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'error_plot_{biomarker_key}.png'
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved error plot: {plot_filename}")
        
        # Create combined error summary plot
        self.create_combined_error_summary()
    
    def create_combined_error_summary(self):
        """Create a combined summary of error plots"""
        print("Creating combined error summary...")
        
        # Create a single figure with subplots for each biomarker
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Biomarker Error Analysis Summary', fontsize=12, y=0.95)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Define biomarkers
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 0),
            ('spare_ad', 'SPARE-AD', 1),
            ('spare_ef', 'SPARE-EF', 2),
            ('adas', 'ADAS', 3),
            ('mmse', 'MMSE', 4)
        ]
        
        for biomarker_key, biomarker_name, idx in biomarkers:
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results[17]
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            # Calculate absolute error
            df['absolute_error'] = np.abs(df['true_value'] - df['predicted_value'])
            
            # Create scatter plot
            ax.scatter(df['time'], df['absolute_error'], 
                      alpha=0.6, s=40, color='#1f77b4', edgecolor='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(df['time'], df['absolute_error'], 1)
            p = np.poly1d(z)
            ax.plot(df['time'], p(df['time']), "r--", linewidth=2, alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(df['time'], df['absolute_error'])[0, 1]
            
            ax.set_xlabel('Time (months)', fontsize=9, fontweight='bold')
            ax.set_ylabel('Absolute Error', fontsize=9, fontweight='bold')
            ax.set_title(f'{biomarker_name}\nCorrelation: {correlation:.3f}', 
                        fontsize=9, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add statistics
            mean_error = df['absolute_error'].mean()
            ax.text(0.02, 0.98, f'Mean: {mean_error:.3f}', transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Hide unused subplots
        for idx in range(len(biomarkers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_plots_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved combined error summary plot")
    
    def create_error_plots_by_progression(self):
        """Create error plots stratified by progression status (diagnosis)"""
        print("Creating error plots by progression status...")
        
        # Load diagnosis information
        diagnosis_mapping, diagnosis_labels = self.load_diagnosis_data()
        
        if not diagnosis_mapping:
            print("No diagnosis information available, skipping progression analysis")
            return
        
        # Set publication-quality plotting parameters
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.linewidth': 0.8,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        # Define biomarkers to analyze
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', self.roi_results.get(17)),
            ('spare_ad', 'SPARE-AD', self.spare_results.get('spare_ad')),
            ('spare_ef', 'SPARE-EF', self.spare_results.get('spare_ef')),
            ('adas', 'ADAS', self.cognitive_results.get('adas')),
            ('mmse', 'MMSE', self.cognitive_results.get('mmse'))
        ]
        
        # Colors for different diagnosis groups
        diagnosis_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#d62728'}  # Blue, Orange, Red
        
        for biomarker_key, biomarker_name, df in biomarkers:
            if df is None or df.empty:
                print(f"No data available for {biomarker_name}, skipping...")
                continue
            
            print(f"Creating progression error plot for {biomarker_name}...")
            
            # Add diagnosis information to the dataframe
            df_with_diagnosis = df.copy()
            df_with_diagnosis['diagnosis'] = df_with_diagnosis['PTID'].map(diagnosis_mapping)
            df_with_diagnosis['diagnosis_label'] = df_with_diagnosis['diagnosis'].map(diagnosis_labels)
            
            # Remove rows with missing diagnosis
            df_with_diagnosis = df_with_diagnosis.dropna(subset=['diagnosis'])
            
            if df_with_diagnosis.empty:
                print(f"No data with diagnosis for {biomarker_name}, skipping...")
                continue
            
            # Calculate absolute error
            df_with_diagnosis['absolute_error'] = np.abs(df_with_diagnosis['true_value'] - df_with_diagnosis['predicted_value'])
            
            # Print interpretable statistics
            print(f"\n=== {biomarker_name} Progression Analysis ===")
            print(f"Total samples: {len(df_with_diagnosis)}")
            
            # Diagnosis statistics
            diagnosis_stats = df_with_diagnosis.groupby('diagnosis_label')['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nDiagnosis Error Statistics:")
            for diagnosis_group, stats in diagnosis_stats.iterrows():
                print(f"  {diagnosis_group}: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Error distribution by diagnosis
            diagnosis_groups = df_with_diagnosis.groupby('diagnosis_label')
            error_data = [group['absolute_error'].values for name, group in diagnosis_groups]
            labels = [name for name, group in diagnosis_groups]
            colors = [diagnosis_colors.get(df_with_diagnosis[df_with_diagnosis['diagnosis_label'] == label]['diagnosis'].iloc[0], '#1f77b4') 
                     for label in labels]
            
            # Create box plot
            bp = ax1.boxplot(error_data, labels=labels, patch_artist=True, 
                           boxprops=dict(facecolor='white', alpha=0.8),
                           medianprops=dict(color='black', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='gray', markersize=3))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_xlabel('Diagnosis Group', fontsize=12)
            ax1.set_ylabel('Absolute Error', fontsize=12)
            ax1.set_title(f'Error Distribution by Diagnosis\n{biomarker_name}', fontsize=12)
            ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add statistics text box
            stats_text = "Error Statistics by Group:\n"
            for name, group in diagnosis_groups:
                mean_error = group['absolute_error'].mean()
                std_error = group['absolute_error'].std()
                count = len(group)
                stats_text += f"{name}: {mean_error:.3f}±{std_error:.3f} (n={count})\n"
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Plot 2: Error vs Time by diagnosis
            for diagnosis_code, diagnosis_label in diagnosis_labels.items():
                if diagnosis_code in df_with_diagnosis['diagnosis'].values:
                    group_data = df_with_diagnosis[df_with_diagnosis['diagnosis'] == diagnosis_code]
                    color = diagnosis_colors.get(diagnosis_code, '#1f77b4')
                    
                    ax2.scatter(group_data['time'], group_data['absolute_error'], 
                              alpha=0.6, s=20, color=color, label=diagnosis_label)
                    
                    # Add trend line for this group
                    if len(group_data) > 1:
                        z = np.polyfit(group_data['time'], group_data['absolute_error'], 1)
                        p = np.poly1d(z)
                        ax2.plot(group_data['time'], p(group_data['time']), 
                               color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax2.set_xlabel('Time from baseline (months)', fontsize=12)
            ax2.set_ylabel('Absolute Error', fontsize=12)
            ax2.set_title(f'Error vs Time by Diagnosis\n{biomarker_name}', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add overall statistics
            overall_stats = f"Overall Statistics:\n"
            overall_stats += f"Total samples: {len(df_with_diagnosis)}\n"
            overall_stats += f"Mean error: {df_with_diagnosis['absolute_error'].mean():.3f}\n"
            overall_stats += f"Std error: {df_with_diagnosis['absolute_error'].std():.3f}"
            
            ax2.text(0.02, 0.98, overall_stats, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'error_by_progression_{biomarker_key}.png'
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved progression error plot: {plot_filename}")
        
        # Create combined progression summary plot
        self.create_combined_progression_summary(diagnosis_mapping, diagnosis_labels, diagnosis_colors)
    
    def create_combined_progression_summary(self, diagnosis_mapping, diagnosis_labels, diagnosis_colors):
        """Create a combined summary plot for all biomarkers by progression status"""
        print("Creating combined progression summary...")
        
        # Create a single figure with subplots for each biomarker
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Biomarker Error Analysis by Progression Status', fontsize=12, y=0.95)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Define biomarkers
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 0),
            ('spare_ad', 'SPARE-AD', 1),
            ('spare_ef', 'SPARE-EF', 2),
            ('adas', 'ADAS', 3),
            ('mmse', 'MMSE', 4)
        ]
        
        for biomarker_key, biomarker_name, idx in biomarkers:
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results.get(17)
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            if df is None or df.empty:
                continue
            
            # Add diagnosis information
            df_with_diagnosis = df.copy()
            df_with_diagnosis['diagnosis'] = df_with_diagnosis['PTID'].map(diagnosis_mapping)
            df_with_diagnosis['diagnosis_label'] = df_with_diagnosis['diagnosis'].map(diagnosis_labels)
            df_with_diagnosis = df_with_diagnosis.dropna(subset=['diagnosis'])
            
            if df_with_diagnosis.empty:
                continue
            
            # Calculate absolute error
            df_with_diagnosis['absolute_error'] = np.abs(df_with_diagnosis['true_value'] - df_with_diagnosis['predicted_value'])
            
            # Create box plot
            diagnosis_groups = df_with_diagnosis.groupby('diagnosis_label')
            error_data = [group['absolute_error'].values for name, group in diagnosis_groups]
            labels = [name for name, group in diagnosis_groups]
            colors = [diagnosis_colors.get(df_with_diagnosis[df_with_diagnosis['diagnosis_label'] == label]['diagnosis'].iloc[0], '#1f77b4') 
                     for label in labels]
            
            bp = ax.boxplot(error_data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='white', alpha=0.8),
                          medianprops=dict(color='black', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='gray', markersize=2))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{biomarker_name}', fontsize=10)
            ax.set_ylabel('Absolute Error', fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add sample size annotations
            for i, (name, group) in enumerate(diagnosis_groups):
                ax.text(i+1, group['absolute_error'].median(), f'n={len(group)}', 
                       ha='center', va='bottom', fontsize=8)
        
        # Remove the last unused subplot
        if len(biomarkers) < len(axes_flat):
            fig.delaxes(axes_flat[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'combined_progression_summary.png'
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined progression summary: {plot_filename}")
    
    def load_baseline_age_data(self):
        """Load baseline age information from longitudinal covariates file"""
        print("Loading baseline age information...")
        try:
            df = pd.read_csv('../LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
            # Get baseline age (age at time 0) for each subject
            baseline_data = df[df['Time'] == 0][['PTID', 'Age']].copy()
            baseline_data = baseline_data.rename(columns={'Age': 'baseline_age'})
            
            # Create a mapping of PTID to baseline age
            baseline_age_mapping = baseline_data.set_index('PTID')['baseline_age'].to_dict()
            
            print(f"Loaded baseline age information for {len(baseline_age_mapping)} subjects")
            print(f"Baseline age range: {min(baseline_age_mapping.values()):.1f} - {max(baseline_age_mapping.values()):.1f} years")
            print(f"Mean baseline age: {np.mean(list(baseline_age_mapping.values())):.1f} ± {np.std(list(baseline_age_mapping.values())):.1f} years")
            
            return baseline_age_mapping
        except Exception as e:
            print(f"Warning: Could not load baseline age information: {e}")
            return {}
    
    def create_error_plots_by_baseline_age(self):
        """Create error plots stratified by baseline age"""
        print("Creating error plots by baseline age...")
        
        # Load baseline age information
        baseline_age_mapping = self.load_baseline_age_data()
        
        if not baseline_age_mapping:
            print("No baseline age information available, skipping age analysis")
            return
        
        # Set publication-quality plotting parameters
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.linewidth': 0.8,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        # Define biomarkers to analyze
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', self.roi_results.get(17)),
            ('spare_ad', 'SPARE-AD', self.spare_results.get('spare_ad')),
            ('spare_ef', 'SPARE-EF', self.spare_results.get('spare_ef')),
            ('adas', 'ADAS', self.cognitive_results.get('adas')),
            ('mmse', 'MMSE', self.cognitive_results.get('mmse'))
        ]
        
        for biomarker_key, biomarker_name, df in biomarkers:
            if df is None or df.empty:
                print(f"No data available for {biomarker_name}, skipping...")
                continue
            
            print(f"Creating baseline age error plot for {biomarker_name}...")
            
            # Add baseline age information to the dataframe
            df_with_age = df.copy()
            df_with_age['baseline_age'] = df_with_age['PTID'].map(baseline_age_mapping)
            
            # Remove rows with missing baseline age
            df_with_age = df_with_age.dropna(subset=['baseline_age'])
            
            if df_with_age.empty:
                print(f"No data with baseline age for {biomarker_name}, skipping...")
                continue
            
            # Calculate absolute error
            df_with_age['absolute_error'] = np.abs(df_with_age['true_value'] - df_with_age['predicted_value'])
            
            # Print interpretable statistics
            print(f"\n=== {biomarker_name} Baseline Age Analysis ===")
            print(f"Total samples: {len(df_with_age)}")
            print(f"Age range: {df_with_age['baseline_age'].min():.1f} - {df_with_age['baseline_age'].max():.1f} years")
            
            # Age correlation
            age_correlation = np.corrcoef(df_with_age['baseline_age'], df_with_age['absolute_error'])[0, 1]
            print(f"Age-Error Correlation: {age_correlation:.4f}")
            
            # Age bin statistics
            age_bins = pd.cut(df_with_age['baseline_age'], bins=5, labels=False)
            df_with_age['age_bin'] = age_bins
            age_bin_stats = df_with_age.groupby('age_bin')['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nAge Bin Error Statistics:")
            for age_bin, stats in age_bin_stats.iterrows():
                age_range = df_with_age[df_with_age['age_bin'] == age_bin]['baseline_age']
                print(f"  Age Bin {age_bin}: {age_range.min():.1f}-{age_range.max():.1f} years: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Error vs Baseline Age (scatter plot)
            ax1.scatter(df_with_age['baseline_age'], df_with_age['absolute_error'], 
                       alpha=0.6, s=20, color='#1f77b4')
            
            # Add trend line
            if len(df_with_age) > 1:
                z = np.polyfit(df_with_age['baseline_age'], df_with_age['absolute_error'], 1)
                p = np.poly1d(z)
                ax1.plot(df_with_age['baseline_age'], p(df_with_age['baseline_age']), 
                        color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Calculate correlation
                correlation = np.corrcoef(df_with_age['baseline_age'], df_with_age['absolute_error'])[0, 1]
                ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax1.set_xlabel('Baseline Age (years)', fontsize=12)
            ax1.set_ylabel('Absolute Error', fontsize=12)
            ax1.set_title(f'Error vs Baseline Age\n{biomarker_name}', fontsize=12)
            ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Plot 2: Error distribution by age bins
            # Create age bins
            age_bins = pd.cut(df_with_age['baseline_age'], bins=5, labels=False)
            df_with_age['age_bin'] = age_bins
            
            # Calculate mean error for each bin
            error_by_age_bin = df_with_age.groupby('age_bin')['absolute_error'].agg(['mean', 'std', 'count']).reset_index()
            
            # Get age bin centers for x-axis labels
            age_bin_centers = df_with_age.groupby('age_bin')['baseline_age'].mean()
            
            # Create bar plot
            bars = ax2.bar(range(len(error_by_age_bin)), error_by_age_bin['mean'], 
                          yerr=error_by_age_bin['std'], capsize=5,
                          color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add sample size annotations
            for i, (bar, count) in enumerate(zip(bars, error_by_age_bin['count'])):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + error_by_age_bin['std'].iloc[i] + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=12)
            
            ax2.set_xlabel('Age Group', fontsize=12)
            ax2.set_ylabel('Mean Absolute Error', fontsize=12)
            ax2.set_title('Error Distribution by Age Group', fontsize=12)
            
            # Set x-axis labels with age ranges
            ax2.set_xticks(range(len(age_bin_centers)))
            age_labels = []
            for i, center in enumerate(age_bin_centers):
                age_std = df_with_age[df_with_age['age_bin'] == i]['baseline_age'].std()
                age_labels.append(f'{center:.1f}±{age_std:.1f}')
            ax2.set_xticklabels(age_labels, rotation=45)
            
            ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add statistics text box
            stats_text = f"Overall Statistics:\n"
            stats_text += f"Total samples: {len(df_with_age)}\n"
            stats_text += f"Mean error: {df_with_age['absolute_error'].mean():.3f}\n"
            stats_text += f"Std error: {df_with_age['absolute_error'].std():.3f}\n"
            stats_text += f"Age range: {df_with_age['baseline_age'].min():.1f}-{df_with_age['baseline_age'].max():.1f} years"
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'error_by_baseline_age_{biomarker_key}.png'
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved baseline age error plot: {plot_filename}")
        
        # Create combined baseline age summary plot
        self.create_combined_baseline_age_summary(baseline_age_mapping)
    
    def create_combined_baseline_age_summary(self, baseline_age_mapping):
        """Create a combined summary plot for all biomarkers by baseline age"""
        print("Creating combined baseline age summary...")
        
        # Create a single figure with subplots for each biomarker
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Biomarker Error Analysis by Baseline Age', fontsize=12, y=0.95)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Define biomarkers
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 0),
            ('spare_ad', 'SPARE-AD', 1),
            ('spare_ef', 'SPARE-EF', 2),
            ('adas', 'ADAS', 3),
            ('mmse', 'MMSE', 4)
        ]
        
        for biomarker_key, biomarker_name, idx in biomarkers:
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results.get(17)
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            if df is None or df.empty:
                continue
            
            # Add baseline age information
            df_with_age = df.copy()
            df_with_age['baseline_age'] = df_with_age['PTID'].map(baseline_age_mapping)
            df_with_age = df_with_age.dropna(subset=['baseline_age'])
            
            if df_with_age.empty:
                continue
            
            # Calculate absolute error
            df_with_age['absolute_error'] = np.abs(df_with_age['true_value'] - df_with_age['predicted_value'])
            
            # Create scatter plot
            ax.scatter(df_with_age['baseline_age'], df_with_age['absolute_error'], 
                      alpha=0.6, s=15, color='#1f77b4')
            
            # Add trend line
            if len(df_with_age) > 1:
                z = np.polyfit(df_with_age['baseline_age'], df_with_age['absolute_error'], 1)
                p = np.poly1d(z)
                ax.plot(df_with_age['baseline_age'], p(df_with_age['baseline_age']), 
                       color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                
                # Calculate and display correlation
                correlation = np.corrcoef(df_with_age['baseline_age'], df_with_age['absolute_error'])[0, 1]
                ax.text(0.05, 0.95, f'r={correlation:.3f}', 
                       transform=ax.transAxes, fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_title(f'{biomarker_name}', fontsize=10)
            ax.set_xlabel('Baseline Age (years)', fontsize=9)
            ax.set_ylabel('Absolute Error', fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Remove the last unused subplot
        if len(biomarkers) < len(axes_flat):
            fig.delaxes(axes_flat[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'combined_baseline_age_summary.png'
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined baseline age summary: {plot_filename}")
    
    def load_apoe4_data(self):
        """Load APOE4 alleles information from longitudinal covariates file"""
        print("Loading APOE4 alleles information...")
        try:
            df = pd.read_csv('../LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
            # Get baseline APOE4 alleles (at time 0) for each subject
            baseline_data = df[df['Time'] == 0][['PTID', 'APOE4_Alleles']].copy()
            baseline_data = baseline_data.rename(columns={'APOE4_Alleles': 'baseline_apoe4'})
            
            # Create a mapping of PTID to baseline APOE4 alleles
            apoe4_mapping = baseline_data.set_index('PTID')['baseline_apoe4'].to_dict()
            
            # Create APOE4 labels
            apoe4_labels = {-1: 'Unknown', 0: '0 alleles', 1: '1 allele', 2: '2 alleles'}
            
            print(f"Loaded APOE4 information for {len(apoe4_mapping)} subjects")
            print("APOE4 distribution:")
            for code, label in apoe4_labels.items():
                count = sum(1 for a in apoe4_mapping.values() if a == code)
                print(f"  {label} (Code {code}): {count} subjects")
            
            return apoe4_mapping, apoe4_labels
        except Exception as e:
            print(f"Warning: Could not load APOE4 information: {e}")
            return {}, {}
    
    def create_error_plots_by_apoe4_and_diagnosis(self):
        """Create error plots stratified by APOE4 alleles and diagnosis status"""
        print("Creating error plots by APOE4 alleles and diagnosis status...")
        
        # Load APOE4 and diagnosis information
        apoe4_mapping, apoe4_labels = self.load_apoe4_data()
        diagnosis_mapping, diagnosis_labels = self.load_diagnosis_data()
        
        if not apoe4_mapping or not diagnosis_mapping:
            print("No APOE4 or diagnosis information available, skipping analysis")
            return
        
        # Set publication-quality plotting parameters with larger fonts for APOE4 and diagnosis plots
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 13,
            'axes.linewidth': 0.8,
            'axes.labelsize': 14,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.5
        })
        
        # Define biomarkers to analyze
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', self.roi_results.get(17)),
            ('spare_ad', 'SPARE-AD', self.spare_results.get('spare_ad')),
            ('spare_ef', 'SPARE-EF', self.spare_results.get('spare_ef')),
            ('adas', 'ADAS', self.cognitive_results.get('adas')),
            ('mmse', 'MMSE', self.cognitive_results.get('mmse'))
        ]
        
        # Colors for different groups
        diagnosis_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#d62728'}  # Blue, Orange, Red
        apoe4_colors = {0: '#2ca02c', 1: '#d62728', 2: '#9467bd'}  # Green, Red, Purple
        
        for biomarker_key, biomarker_name, df in biomarkers:
            if df is None or df.empty:
                print(f"No data available for {biomarker_name}, skipping...")
                continue
            
            print(f"Creating APOE4 and diagnosis error plot for {biomarker_name}...")
            
            # Add APOE4 and diagnosis information to the dataframe
            df_with_info = df.copy()
            df_with_info['apoe4'] = df_with_info['PTID'].map(apoe4_mapping)
            df_with_info['apoe4_label'] = df_with_info['apoe4'].map(apoe4_labels)
            df_with_info['diagnosis'] = df_with_info['PTID'].map(diagnosis_mapping)
            df_with_info['diagnosis_label'] = df_with_info['diagnosis'].map(diagnosis_labels)
            
            # Remove rows with missing information (excluding unknown APOE4)
            df_with_info = df_with_info.dropna(subset=['diagnosis'])
            df_with_info = df_with_info[df_with_info['apoe4'] != -1]  # Remove unknown APOE4
            
            if df_with_info.empty:
                print(f"No data with complete information for {biomarker_name}, skipping...")
                continue
            
            # Calculate absolute error
            df_with_info['absolute_error'] = np.abs(df_with_info['true_value'] - df_with_info['predicted_value'])
            
            # Print interpretable statistics
            print(f"\n=== {biomarker_name} Analysis ===")
            print(f"Total samples: {len(df_with_info)}")
            
            # APOE4 statistics
            apoe4_stats = df_with_info.groupby('apoe4_label')['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nAPOE4 Error Statistics:")
            for apoe4_group, stats in apoe4_stats.iterrows():
                print(f"  {apoe4_group}: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Diagnosis statistics
            diagnosis_stats = df_with_info.groupby('diagnosis_label')['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nDiagnosis Error Statistics:")
            for diagnosis_group, stats in diagnosis_stats.iterrows():
                print(f"  {diagnosis_group}: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Combined statistics
            combined_stats = df_with_info.groupby(['apoe4_label', 'diagnosis_label'])['absolute_error'].agg(['mean', 'std', 'count'])
            print(f"\nCombined APOE4 + Diagnosis Error Statistics:")
            for (apoe4, diagnosis), stats in combined_stats.iterrows():
                print(f"  {apoe4} + {diagnosis}: Mean Error = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            # Create figure with multiple subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Error distribution by APOE4 alleles
            apoe4_groups = df_with_info.groupby('apoe4_label')
            error_data_apoe4 = [group['absolute_error'].values for name, group in apoe4_groups]
            labels_apoe4 = [name for name, group in apoe4_groups]
            colors_apoe4 = [apoe4_colors.get(df_with_info[df_with_info['apoe4_label'] == label]['apoe4'].iloc[0], '#1f77b4') 
                           for label in labels_apoe4]
            
            bp1 = ax1.boxplot(error_data_apoe4, labels=labels_apoe4, patch_artist=True,
                             boxprops=dict(facecolor='white', alpha=0.8),
                             medianprops=dict(color='black', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='gray', markersize=3))
            
            # Color the boxes
            for patch, color in zip(bp1['boxes'], colors_apoe4):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_xlabel('APOE4 Alleles', fontsize=14)
            ax1.set_ylabel('Absolute Error', fontsize=14)
            ax1.set_title(f'Error Distribution by APOE4 Alleles\n{biomarker_name}', fontsize=14)
            ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add statistics text box for APOE4
            stats_text_apoe4 = "Error Statistics by APOE4:\n"
            for name, group in apoe4_groups:
                mean_error = group['absolute_error'].mean()
                std_error = group['absolute_error'].std()
                count = len(group)
                stats_text_apoe4 += f"{name}: {mean_error:.3f}±{std_error:.3f} (n={count})\n"
            
            ax1.text(0.02, 0.98, stats_text_apoe4, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Plot 2: Error distribution by diagnosis
            diagnosis_groups = df_with_info.groupby('diagnosis_label')
            error_data_diagnosis = [group['absolute_error'].values for name, group in diagnosis_groups]
            labels_diagnosis = [name for name, group in diagnosis_groups]
            colors_diagnosis = [diagnosis_colors.get(df_with_info[df_with_info['diagnosis_label'] == label]['diagnosis'].iloc[0], '#1f77b4') 
                               for label in labels_diagnosis]
            
            bp2 = ax2.boxplot(error_data_diagnosis, labels=labels_diagnosis, patch_artist=True,
                             boxprops=dict(facecolor='white', alpha=0.8),
                             medianprops=dict(color='black', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='gray', markersize=3))
            
            # Color the boxes
            for patch, color in zip(bp2['boxes'], colors_diagnosis):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_xlabel('Diagnosis Group', fontsize=14)
            ax2.set_ylabel('Absolute Error', fontsize=14)
            ax2.set_title(f'Error Distribution by Diagnosis\n{biomarker_name}', fontsize=14)
            ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add statistics text box for diagnosis
            stats_text_diagnosis = "Error Statistics by Diagnosis:\n"
            for name, group in diagnosis_groups:
                mean_error = group['absolute_error'].mean()
                std_error = group['absolute_error'].std()
                count = len(group)
                stats_text_diagnosis += f"{name}: {mean_error:.3f}±{std_error:.3f} (n={count})\n"
            
            ax2.text(0.02, 0.98, stats_text_diagnosis, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Plot 3: Error vs Time by APOE4 alleles
            for apoe4_code, apoe4_label in apoe4_labels.items():
                if apoe4_code == -1:  # Skip unknown
                    continue
                if apoe4_code in df_with_info['apoe4'].values:
                    group_data = df_with_info[df_with_info['apoe4'] == apoe4_code]
                    color = apoe4_colors.get(apoe4_code, '#1f77b4')
                    
                    ax3.scatter(group_data['time'], group_data['absolute_error'], 
                               alpha=0.6, s=20, color=color, label=apoe4_label)
                    
                    # Add trend line for this group
                    if len(group_data) > 1:
                        z = np.polyfit(group_data['time'], group_data['absolute_error'], 1)
                        p = np.poly1d(z)
                        ax3.plot(group_data['time'], p(group_data['time']), 
                               color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax3.set_xlabel('Time from baseline (months)', fontsize=14)
            ax3.set_ylabel('Absolute Error', fontsize=14)
            ax3.set_title(f'Error vs Time by APOE4 Alleles\n{biomarker_name}', fontsize=14)
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Plot 4: Error vs Time by diagnosis
            for diagnosis_code, diagnosis_label in diagnosis_labels.items():
                if diagnosis_code in df_with_info['diagnosis'].values:
                    group_data = df_with_info[df_with_info['diagnosis'] == diagnosis_code]
                    color = diagnosis_colors.get(diagnosis_code, '#1f77b4')
                    
                    ax4.scatter(group_data['time'], group_data['absolute_error'], 
                               alpha=0.6, s=20, color=color, label=diagnosis_label)
                    
                    # Add trend line for this group
                    if len(group_data) > 1:
                        z = np.polyfit(group_data['time'], group_data['absolute_error'], 1)
                        p = np.poly1d(z)
                        ax4.plot(group_data['time'], p(group_data['time']), 
                               color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax4.set_xlabel('Time from baseline (months)', fontsize=14)
            ax4.set_ylabel('Absolute Error', fontsize=14)
            ax4.set_title(f'Error vs Time by Diagnosis\n{biomarker_name}', fontsize=14)
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add overall statistics
            overall_stats = f"Overall Statistics:\n"
            overall_stats += f"Total samples: {len(df_with_info)}\n"
            overall_stats += f"Mean error: {df_with_info['absolute_error'].mean():.3f}\n"
            overall_stats += f"Std error: {df_with_info['absolute_error'].std():.3f}"
            
            ax3.text(0.02, 0.98, overall_stats, transform=ax3.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'error_by_apoe4_and_diagnosis_{biomarker_key}.png'
            plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved APOE4 and diagnosis error plot: {plot_filename}")
        
        # Create combined APOE4 and diagnosis summary plot
        self.create_combined_apoe4_diagnosis_summary(apoe4_mapping, apoe4_labels, diagnosis_mapping, diagnosis_labels, apoe4_colors, diagnosis_colors)
    
    def create_combined_apoe4_diagnosis_summary(self, apoe4_mapping, apoe4_labels, diagnosis_mapping, diagnosis_labels, apoe4_colors, diagnosis_colors):
        """Create a combined summary plot for all biomarkers by APOE4 and diagnosis"""
        print("Creating combined APOE4 and diagnosis summary...")
        
        # Create a single figure with subplots for each biomarker
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Biomarker Error Analysis by APOE4 Alleles and Diagnosis Status', fontsize=12, y=0.95)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Define biomarkers
        biomarkers = [
            ('roi_17', 'Hippocampus (ROI 17)', 0),
            ('spare_ad', 'SPARE-AD', 1),
            ('spare_ef', 'SPARE-EF', 2),
            ('adas', 'ADAS', 3),
            ('mmse', 'MMSE', 4)
        ]
        
        for biomarker_key, biomarker_name, idx in biomarkers:
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get data for this biomarker
            if biomarker_key == 'roi_17':
                df = self.roi_results.get(17)
            elif biomarker_key in self.spare_results:
                df = self.spare_results[biomarker_key]
            elif biomarker_key in self.cognitive_results:
                df = self.cognitive_results[biomarker_key]
            else:
                continue
            
            if df is None or df.empty:
                continue
            
            # Add APOE4 and diagnosis information
            df_with_info = df.copy()
            df_with_info['apoe4'] = df_with_info['PTID'].map(apoe4_mapping)
            df_with_info['apoe4_label'] = df_with_info['apoe4'].map(apoe4_labels)
            df_with_info['diagnosis'] = df_with_info['PTID'].map(diagnosis_mapping)
            df_with_info['diagnosis_label'] = df_with_info['diagnosis'].map(diagnosis_labels)
            
            # Remove rows with missing information
            df_with_info = df_with_info.dropna(subset=['diagnosis'])
            df_with_info = df_with_info[df_with_info['apoe4'] != -1]
            
            if df_with_info.empty:
                continue
            
            # Calculate absolute error
            df_with_info['absolute_error'] = np.abs(df_with_info['true_value'] - df_with_info['predicted_value'])
            
            # Create grouped box plot by APOE4 and diagnosis
            # Create a combined category
            df_with_info['group'] = df_with_info['apoe4_label'] + ' + ' + df_with_info['diagnosis_label']
            
            # Get unique groups and their data
            groups = df_with_info['group'].unique()
            error_data = [df_with_info[df_with_info['group'] == group]['absolute_error'].values for group in groups]
            
            # Create box plot
            bp = ax.boxplot(error_data, labels=groups, patch_artist=True,
                          boxprops=dict(facecolor='white', alpha=0.8),
                          medianprops=dict(color='black', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='gray', markersize=2))
            
            # Color the boxes based on diagnosis (primary grouping)
            for i, group in enumerate(groups):
                diagnosis_part = group.split(' + ')[1]
                diagnosis_code = next(code for code, label in diagnosis_labels.items() if label == diagnosis_part)
                color = diagnosis_colors.get(diagnosis_code, '#1f77b4')
                bp['boxes'][i].set_facecolor(color)
                bp['boxes'][i].set_alpha(0.7)
            
            ax.set_title(f'{biomarker_name}', fontsize=14)
            ax.set_ylabel('Absolute Error', fontsize=13)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Add sample size annotations
            for i, group in enumerate(groups):
                count = len(df_with_info[df_with_info['group'] == group])
                ax.text(i+1, df_with_info[df_with_info['group'] == group]['absolute_error'].median(), 
                       f'n={count}', ha='center', va='bottom', fontsize=7)
        
        # Remove the last unused subplot
        if len(biomarkers) < len(axes_flat):
            fig.delaxes(axes_flat[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'combined_apoe4_diagnosis_summary.png'
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined APOE4 and diagnosis summary: {plot_filename}")
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting comprehensive biomarker evaluation...")
        
        # Load all results
        self.load_all_results()
        
        if not self.roi_results and not self.spare_results and not self.cognitive_results:
            print("No inference results found! Please run inference first.")
            return
        
        # Calculate metrics
        self.calculate_overall_metrics()
        
        # Create visualizations
        self.create_performance_summary()
        
        # Analyze hippocampus
        self.analyze_hippocampus()
        
        # Create trajectory plots
        self.create_trajectory_plots()
        
        # Create error plots with time
        self.create_error_plots_with_time()
        
        # Create error plots by progression status
        self.create_error_plots_by_progression()
        
        # Create error plots by baseline age
        self.create_error_plots_by_baseline_age()

        # Create error plots by APOE4 and diagnosis
        self.create_error_plots_by_apoe4_and_diagnosis()
        
        # Generate report
        self.generate_report()
        
        print(f"Evaluation completed! Results saved to {self.output_dir}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("Generating evaluation report...")
        
        report = f"""
# Comprehensive Biomarker Evaluation Report
## HMUSE Version {self.hmuse_version}

## Summary Statistics

### ROIs ({len(self.roi_results)} models)
"""
        
        if 'rois' in self.overall_metrics and not self.overall_metrics['rois'].empty:
            report += f"- Mean R²: {self.overall_metrics['rois']['r2'].mean():.3f} ± {self.overall_metrics['rois']['r2'].std():.3f}\n"
            report += f"- Mean MAE: {self.overall_metrics['rois']['mae'].mean():.3f} ± {self.overall_metrics['rois']['mae'].std():.3f}\n"
            report += f"- Mean Coverage: {self.overall_metrics['rois']['coverage'].mean():.3f} ± {self.overall_metrics['rois']['coverage'].std():.3f}\n"
        else:
            report += "- No ROI results available\n"
        
        report += f"\n### Spare Scores ({len(self.spare_results)} models)\n"
        if 'spare_scores' in self.overall_metrics and not self.overall_metrics['spare_scores'].empty:
            report += f"- Mean R²: {self.overall_metrics['spare_scores']['r2'].mean():.3f} ± {self.overall_metrics['spare_scores']['r2'].std():.3f}\n"
            report += f"- Mean MAE: {self.overall_metrics['spare_scores']['mae'].mean():.3f} ± {self.overall_metrics['spare_scores']['mae'].std():.3f}\n"
            report += f"- Mean Coverage: {self.overall_metrics['spare_scores']['coverage'].mean():.3f} ± {self.overall_metrics['spare_scores']['coverage'].std():.3f}\n"
        else:
            report += "- No spare score results available\n"
        
        report += f"\n### Cognitive Scores ({len(self.cognitive_results)} models)\n"
        if 'cognitive_scores' in self.overall_metrics and not self.overall_metrics['cognitive_scores'].empty:
            report += f"- Mean R²: {self.overall_metrics['cognitive_scores']['r2'].mean():.3f} ± {self.overall_metrics['cognitive_scores']['r2'].std():.3f}\n"
            report += f"- Mean MAE: {self.overall_metrics['cognitive_scores']['mae'].mean():.3f} ± {self.overall_metrics['cognitive_scores']['mae'].std():.3f}\n"
            report += f"- Mean Coverage: {self.overall_metrics['cognitive_scores']['coverage'].mean():.3f} ± {self.overall_metrics['cognitive_scores']['coverage'].std():.3f}\n"
        else:
            report += "- No cognitive score results available\n"
        
        report += "\n## Top Performing ROIs\n"
        
        # Add top ROIs
        if 'rois' in self.overall_metrics and not self.overall_metrics['rois'].empty:
            top_rois = self.overall_metrics['rois'].nlargest(10, 'r2')
            for _, roi in top_rois.iterrows():
                report += f"- ROI {roi['roi_idx']}: R² = {roi['r2']:.3f}, MAE = {roi['mae']:.3f}\n"
        else:
            report += "- No ROI results available\n"
        
        # Add hippocampus analysis
        if 'rois' in self.overall_metrics and not self.overall_metrics['rois'].empty and 17 in self.roi_results:
            hippo_metrics = self.overall_metrics['rois'][self.overall_metrics['rois']['roi_idx'] == 17].iloc[0]
            report += f"""
## Hippocampus (ROI 17) Analysis
- R²: {hippo_metrics['r2']:.3f}
- MAE: {hippo_metrics['mae']:.3f}
- RMSE: {hippo_metrics['rmse']:.3f}
- Coverage: {hippo_metrics['coverage']:.3f}
- Interval Width: {hippo_metrics['interval_width']:.3f}
"""
        
        # Save report
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        print(f"Report saved to {self.output_dir / 'evaluation_report.md'}")
    
    def load_diagnosis_data(self):
        """Load diagnosis information from longitudinal covariates file"""
        print("Loading diagnosis information...")
        try:
            df = pd.read_csv('../LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
            # Create a mapping of PTID to diagnosis
            # For each subject, we'll use their most common diagnosis across visits
            diagnosis_mapping = df.groupby('PTID')['Diagnosis'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
            
            # Create diagnosis labels
            diagnosis_labels = {0: 'Normal', 1: 'MCI', 2: 'AD'}
            
            print(f"Loaded diagnosis information for {len(diagnosis_mapping)} subjects")
            print("Diagnosis distribution:")
            for code, label in diagnosis_labels.items():
                count = sum(1 for d in diagnosis_mapping.values() if d == code)
                print(f"  {label} (Code {code}): {count} subjects")
            
            return diagnosis_mapping, diagnosis_labels
        except Exception as e:
            print(f"Warning: Could not load diagnosis information: {e}")
            return {}, {}

if __name__ == "__main__":
    import sys
    
    hmuse_version = 0
    if len(sys.argv) > 1:
        hmuse_version = int(sys.argv[1])
    
    evaluator = BiomarkerEvaluator(hmuse_version)
    evaluator.run_full_evaluation() 