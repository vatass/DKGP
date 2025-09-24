#!/usr/bin/env python3
'''
Population Demographics Analysis
- Load training subject IDs from fold 0 for each biomarker
- Calculate comprehensive demographics and statistics
- Generate detailed population description
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from pathlib import Path
import logging
import matplotlib as mpl

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

class PopulationDemographicsAnalyzer:
    def __init__(self):
        self.output_dir = Path("population_demographics_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define biomarkers and their corresponding data files
        self.biomarkers = {
            'hippocampus': 'Hippocampus (ROI 17)',
            'spare_ad': 'SPARE-AD',
            'spare_ba': 'SPARE-BA',  # Note: SPARE-BA instead of SPARE-EF
            'adas': 'ADAS',
            'mmse': 'MMSE'
        }
        
        # Define file mappings for each biomarker
        self.biomarker_files = {
            'hippocampus': 'train_subject_allstudies_ids_dl_hmuse0.pkl',
            'spare_ad': 'train_subject_allstudies_ids_dl_muse_spare0.pkl',
            'spare_ba': 'train_subject_allstudies_ids_dl_muse_spare0.pkl',  # Same as SPARE-AD
            'adas': 'train_subject_adni_ids_adas0.pkl',
            'mmse': 'train_subject_allstudies_ids_mmse0.pkl'
        }
        
        # Define directories for data
        self.data_dir = Path("data")
        self.covariates_file = Path("../LongGPRegressionBaseline/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv")
        
        # Colors for plots
        self.colors = {
            'hippocampus': '#1f77b4',
            'spare_ad': '#ff7f0e',
            'spare_ba': '#2ca02c',
            'adas': '#d62728',
            'mmse': '#9467bd'
        }
        
        # Store results
        self.training_subjects = {}
        self.demographics_data = {}
        self.statistics = {}
        
    def load_training_subjects_fold0(self):
        """Load training subject IDs from fold 0 for each biomarker"""
        print("Loading training subjects from fold 0...")
        
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.biomarker_files:
                train_file = self.data_dir / self.biomarker_files[biomarker_key]
                
                if train_file.exists():
                    try:
                        import pickle
                        with open(train_file, 'rb') as f:
                            self.training_subjects[biomarker_key] = pickle.load(f)
                        
                        print(f"  {self.biomarkers[biomarker_key]}: {len(self.training_subjects[biomarker_key])} training subjects")
                        
                    except Exception as e:
                        print(f"  Error loading {train_file}: {e}")
                        self.training_subjects[biomarker_key] = []
                else:
                    print(f"  Warning: {train_file} not found")
                    self.training_subjects[biomarker_key] = []
            else:
                print(f"  Warning: No file mapping found for {biomarker_key}")
                self.training_subjects[biomarker_key] = []
    
    def load_covariates_data(self):
        """Load longitudinal covariates data"""
        print("Loading covariates data...")
        
        if self.covariates_file.exists():
            try:
                self.covariates_df = pd.read_csv(self.covariates_file)
                print(f"  Loaded {len(self.covariates_df)} records from covariates file")
                print(f"  Unique subjects: {self.covariates_df['PTID'].nunique()}")
                return True
            except Exception as e:
                print(f"  Error loading covariates file: {e}")
                return False
        else:
            print(f"  Warning: {self.covariates_file} not found")
            return False
    
    def extract_baseline_demographics(self):
        """Extract baseline demographics for each biomarker's training subjects"""
        print("Extracting baseline demographics...")
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            print(f"\n=== {biomarker_name} ===")
            
            if not self.training_subjects[biomarker_key]:
                print("  No training subjects found")
                continue
            
            # Get baseline data (Time = 0) for training subjects
            training_subjects_set = set(self.training_subjects[biomarker_key])
            baseline_data = self.covariates_df[
                (self.covariates_df['PTID'].isin(training_subjects_set)) & 
                (self.covariates_df['Time'] == 0)
            ].copy()
            
            if baseline_data.empty:
                print("  No baseline data found for training subjects")
                continue
            
            print(f"  Baseline subjects: {len(baseline_data)}")
            
            # Store demographics data
            self.demographics_data[biomarker_key] = baseline_data
            
            # Calculate and print basic statistics
            self.calculate_demographic_statistics(biomarker_key, baseline_data)
    
    def calculate_demographic_statistics(self, biomarker_key, data):
        """Calculate comprehensive demographic statistics"""
        print(f"  Calculating statistics for {self.biomarkers[biomarker_key]}...")
        
        # Print key demographic insights
        print(f"    📊 Population Overview:")
        print(f"      - Total subjects: {len(data):,}")
        print(f"      - Unique subjects: {data['PTID'].nunique():,}")
        
        stats = {}
        
        # Basic counts
        stats['total_subjects'] = len(data)
        stats['unique_subjects'] = data['PTID'].nunique()
        
        # Age statistics
        if 'Age' in data.columns:
            age_stats = data['Age'].describe()
            stats['age'] = {
                'mean': age_stats['mean'],
                'std': age_stats['std'],
                'min': age_stats['min'],
                'max': age_stats['max'],
                'median': age_stats['50%'],
                'q25': age_stats['25%'],
                'q75': age_stats['75%']
            }
            print(f"    Age: {stats['age']['mean']:.1f} ± {stats['age']['std']:.1f} years (range: {stats['age']['min']:.1f}-{stats['age']['max']:.1f})")
        
        # Gender statistics
        if 'Gender' in data.columns:
            gender_counts = data['Gender'].value_counts()
            stats['gender'] = {
                'male': gender_counts.get(1, 0),
                'female': gender_counts.get(0, 0),
                'male_pct': gender_counts.get(1, 0) / len(data) * 100,
                'female_pct': gender_counts.get(0, 0) / len(data) * 100
            }
            print(f"    Gender: {stats['gender']['male']} male ({stats['gender']['male_pct']:.1f}%), {stats['gender']['female']} female ({stats['gender']['female_pct']:.1f}%)")
        
        # Diagnosis statistics
        if 'Diagnosis' in data.columns:
            diagnosis_counts = data['Diagnosis'].value_counts()
            diagnosis_labels = {0: 'Normal', 1: 'MCI', 2: 'AD'}
            stats['diagnosis'] = {}
            print("    Diagnosis:")
            for code, count in diagnosis_counts.items():
                label = diagnosis_labels.get(code, f'Code_{code}')
                pct = count / len(data) * 100
                stats['diagnosis'][label] = {'count': count, 'percentage': pct}
                print(f"      {label}: {count} ({pct:.1f}%)")
        
        # APOE4 statistics
        if 'APOE4_Alleles' in data.columns:
            apoe4_counts = data['APOE4_Alleles'].value_counts()
            apoe4_labels = {-1: 'Unknown', 0: '0 alleles', 1: '1 allele', 2: '2 alleles'}
            stats['apoe4'] = {}
            print("    APOE4 Alleles:")
            for code, count in apoe4_counts.items():
                label = apoe4_labels.get(code, f'Code_{code}')
                pct = count / len(data) * 100
                stats['apoe4'][label] = {'count': count, 'percentage': pct}
                print(f"      {label}: {count} ({pct:.1f}%)")
        
        # Education statistics
        if 'Education' in data.columns:
            edu_stats = data['Education'].describe()
            stats['education'] = {
                'mean': edu_stats['mean'],
                'std': edu_stats['std'],
                'min': edu_stats['min'],
                'max': edu_stats['max'],
                'median': edu_stats['50%']
            }
            print(f"    Education: {stats['education']['mean']:.1f} ± {stats['education']['std']:.1f} years (range: {stats['education']['min']:.1f}-{stats['education']['max']:.1f})")
        
        # Scanner statistics
        if 'MRI_Scanner_Model' in data.columns:
            scanner_counts = data['MRI_Scanner_Model'].value_counts()
            stats['scanner'] = {
                'unique_models': len(scanner_counts),
                'most_common': scanner_counts.index[0] if len(scanner_counts) > 0 else 'Unknown',
                'most_common_count': scanner_counts.iloc[0] if len(scanner_counts) > 0 else 0
            }
            print(f"    MRI Scanner Models: {stats['scanner']['unique_models']} unique models")
            print(f"      Most common: {stats['scanner']['most_common']} ({stats['scanner']['most_common_count']} subjects)")
        
        # Study statistics
        if 'Study' in data.columns:
            study_counts = data['Study'].value_counts()
            stats['study'] = {
                'unique_studies': len(study_counts),
                'most_common': study_counts.index[0] if len(study_counts) > 0 else 'Unknown',
                'most_common_count': study_counts.iloc[0] if len(study_counts) > 0 else 0
            }
            print(f"    Studies ({stats['study']['unique_studies']} unique studies):")
            for study, count in study_counts.items():
                pct = count / len(data) * 100
                stats['study'][study] = {'count': count, 'percentage': pct}
                print(f"      {study}: {count:,} ({pct:.1f}%)")
            
            # Print study diversity insight
            if stats['study']['unique_studies'] == 1:
                print(f"      → Single-study population: {stats['study']['most_common']}")
            else:
                print(f"      → Multi-study population with {stats['study']['unique_studies']} studies")
                print(f"        Most represented: {stats['study']['most_common']} ({stats['study']['most_common_count']:,} subjects)")
        
        # Store statistics
        self.statistics[biomarker_key] = stats
    
    def create_demographic_plots(self):
        """Create demographic visualization plots"""
        print("Creating demographic plots...")
        
        # Set up plotting parameters with larger fonts
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.linewidth': 0.8,
            'axes.labelsize': 13,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.2
        })
        
        # 1. Age distribution comparison
        self.plot_age_distribution()
        
        # 2. Diagnosis distribution
        self.plot_diagnosis_distribution()
        
        # 3. Gender distribution
        self.plot_gender_distribution()
        
        # 4. APOE4 distribution
        self.plot_apoe4_distribution()
        
        # 5. Subject overlap between biomarkers
        self.plot_subject_overlap()
        
        # 6. Study distribution
        self.plot_study_distribution()
        
        # 7. Comprehensive summary
        self.create_comprehensive_summary()
    
    def plot_age_distribution(self):
        """Plot age distribution for all biomarkers"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.demographics_data and 'Age' in self.demographics_data[biomarker_key].columns:
                data = self.demographics_data[biomarker_key]['Age'].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=20, alpha=0.6, label=biomarker_name, 
                           color=self.colors[biomarker_key], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Age (years)', fontsize=13)
        ax.set_ylabel('Number of Subjects', fontsize=13)
        ax.set_title('Age Distribution by Biomarker', fontsize=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'age_distribution.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: age_distribution.png and age_distribution.svg")
    
    def plot_diagnosis_distribution(self):
        """Plot diagnosis distribution for all biomarkers"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (biomarker_key, biomarker_name) in enumerate(self.biomarkers.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if biomarker_key in self.demographics_data and 'Diagnosis' in self.demographics_data[biomarker_key].columns:
                data = self.demographics_data[biomarker_key]['Diagnosis'].value_counts()
                diagnosis_labels = {0: 'Normal', 1: 'MCI', 2: 'AD'}
                
                labels = [diagnosis_labels.get(code, f'Code_{code}') for code in data.index]
                colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
                
                ax.pie(data.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title(f'{biomarker_name}\n(n={len(self.demographics_data[biomarker_key])})', fontsize=11)
        
        # Remove unused subplots
        for i in range(len(self.biomarkers), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diagnosis_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'diagnosis_distribution.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: diagnosis_distribution.png and diagnosis_distribution.svg")
    
    def plot_gender_distribution(self):
        """Plot gender distribution for all biomarkers"""
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        
        for i, (biomarker_key, biomarker_name) in enumerate(self.biomarkers.items()):
            ax = axes[i]
            
            if biomarker_key in self.demographics_data and 'Gender' in self.demographics_data[biomarker_key].columns:
                data = self.demographics_data[biomarker_key]['Gender'].value_counts()
                
                labels = ['Female', 'Male']
                values = [data.get(0, 0), data.get(1, 0)]
                colors = ['#ff7f0e', '#1f77b4']
                
                bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_title(f'{biomarker_name}\n(n={len(self.demographics_data[biomarker_key])})', fontsize=13)
                ax.set_ylabel('Number of Subjects', fontsize=12)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gender_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'gender_distribution.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: gender_distribution.png and gender_distribution.svg")
    
    def plot_apoe4_distribution(self):
        """Plot APOE4 distribution for all biomarkers"""
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        
        for i, (biomarker_key, biomarker_name) in enumerate(self.biomarkers.items()):
            ax = axes[i]
            
            if biomarker_key in self.demographics_data and 'APOE4_Alleles' in self.demographics_data[biomarker_key].columns:
                data = self.demographics_data[biomarker_key]['APOE4_Alleles'].value_counts()
                
                labels = ['0 alleles', '1 allele', '2 alleles', 'Unknown']
                values = [data.get(0, 0), data.get(1, 0), data.get(2, 0), data.get(-1, 0)]
                colors = ['#2ca02c', '#ff7f0e', '#d62728', '#7f7f7f']
                
                bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_title(f'{biomarker_name}\n(n={len(self.demographics_data[biomarker_key])})', fontsize=13)
                ax.set_ylabel('Number of Subjects', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'apoe4_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'apoe4_distribution.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: apoe4_distribution.png and apoe4_distribution.svg")
    
    def plot_subject_overlap(self):
        """Plot subject overlap between biomarkers"""
        # Create sets of subjects for each biomarker
        subject_sets = {}
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.training_subjects:
                subject_sets[biomarker_key] = set(self.training_subjects[biomarker_key])
        
        if len(subject_sets) < 2:
            print("  Skipping overlap plot: need at least 2 biomarkers")
            return
        
        # Create overlap matrix
        biomarkers_list = list(subject_sets.keys())
        overlap_matrix = np.zeros((len(biomarkers_list), len(biomarkers_list)))
        
        for i, bio1 in enumerate(biomarkers_list):
            for j, bio2 in enumerate(biomarkers_list):
                if i == j:
                    overlap_matrix[i, j] = len(subject_sets[bio1])
                else:
                    overlap_matrix[i, j] = len(subject_sets[bio1] & subject_sets[bio2])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(overlap_matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(len(biomarkers_list)):
            for j in range(len(biomarkers_list)):
                text = ax.text(j, i, int(overlap_matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=10)
        
        # Customize plot
        ax.set_xticks(range(len(biomarkers_list)))
        ax.set_yticks(range(len(biomarkers_list)))
        ax.set_xticklabels([self.biomarkers[bio] for bio in biomarkers_list], rotation=45, ha='right')
        ax.set_yticklabels([self.biomarkers[bio] for bio in biomarkers_list])
        ax.set_title('Subject Overlap Between Biomarkers', fontsize=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Subjects', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'subject_overlap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'subject_overlap.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: subject_overlap.png and subject_overlap.svg")
    
    def plot_study_distribution(self):
        """Plot study distribution for all biomarkers"""
        print("Creating study distribution plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (biomarker_key, biomarker_name) in enumerate(self.biomarkers.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if biomarker_key in self.demographics_data and 'Study' in self.demographics_data[biomarker_key].columns:
                data = self.demographics_data[biomarker_key]['Study'].value_counts()
                
                # Create pie chart
                colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
                wedges, texts, autotexts = ax.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                                                 colors=colors, startangle=90)
                
                # Customize text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
                
                for text in texts:
                    text.set_fontsize(11)
                
                ax.set_title(f'{biomarker_name}\n(n={len(self.demographics_data[biomarker_key])})', 
                           fontsize=13, fontweight='bold')
        
        # Remove unused subplots
        for i in range(len(self.biomarkers), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'study_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'study_distribution.svg', bbox_inches='tight')
        plt.close()
        print("  Saved: study_distribution.png and study_distribution.svg")
    
    def create_comprehensive_summary(self):
        """Create a comprehensive summary table"""
        print("Creating comprehensive summary...")
        
        # Create summary DataFrame
        summary_data = []
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.statistics:
                stats = self.statistics[biomarker_key]
                
                row = {
                    'Biomarker': biomarker_name,
                    'Total_Subjects': stats.get('total_subjects', 0),
                    'Unique_Subjects': stats.get('unique_subjects', 0)
                }
                
                # Age statistics
                if 'age' in stats:
                    row.update({
                        'Age_Mean': f"{stats['age']['mean']:.1f}",
                        'Age_Std': f"{stats['age']['std']:.1f}",
                        'Age_Range': f"{stats['age']['min']:.1f}-{stats['age']['max']:.1f}"
                    })
                
                # Gender statistics
                if 'gender' in stats:
                    row.update({
                        'Male_Count': stats['gender']['male'],
                        'Female_Count': stats['gender']['female'],
                        'Male_Percentage': f"{stats['gender']['male_pct']:.1f}%"
                    })
                
                # Diagnosis statistics
                if 'diagnosis' in stats:
                    for diag, diag_stats in stats['diagnosis'].items():
                        row[f'{diag}_Count'] = diag_stats['count']
                        row[f'{diag}_Percentage'] = f"{diag_stats['percentage']:.1f}%"
                
                # APOE4 statistics
                if 'apoe4' in stats:
                    for apoe4, apoe4_stats in stats['apoe4'].items():
                        row[f'APOE4_{apoe4}_Count'] = apoe4_stats['count']
                        row[f'APOE4_{apoe4}_Percentage'] = f"{apoe4_stats['percentage']:.1f}%"
                
                summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'demographics_summary.csv', index=False)
        print("  Saved: demographics_summary.csv")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE DEMOGRAPHICS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Save detailed statistics as JSON
        with open(self.output_dir / 'detailed_statistics.json', 'w') as f:
            json.dump(self.statistics, f, indent=2, default=str)
        print("  Saved: detailed_statistics.json")
    
    def print_comprehensive_summary(self):
        """Print a comprehensive, interpretable summary of all findings"""
        print("\n" + "="*80)
        print("COMPREHENSIVE POPULATION DEMOGRAPHICS SUMMARY")
        print("="*80)
        
        # Overall summary
        total_unique_subjects = set()
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.training_subjects:
                total_unique_subjects.update(self.training_subjects[biomarker_key])
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   • Total unique subjects across all biomarkers: {len(total_unique_subjects)}")
        print(f"   • Biomarkers analyzed: {len(self.biomarkers)}")
        
        # Biomarker-specific summaries
        print(f"\n🔬 BIOMARKER-SPECIFIC ANALYSES:")
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.statistics:
                stats = self.statistics[biomarker_key]
                print(f"\n   📈 {biomarker_name}:")
                print(f"      • Training subjects: {stats.get('total_subjects', 0):,}")
                
                # Age information
                if 'age' in stats:
                    age_stats = stats['age']
                    print(f"      • Age: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years")
                    print(f"        (Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years)")
                
                # Diagnosis breakdown
                if 'diagnosis' in stats:
                    print(f"      • Diagnosis distribution:")
                    for diag, diag_stats in stats['diagnosis'].items():
                        print(f"        - {diag}: {diag_stats['count']:,} ({diag_stats['percentage']:.1f}%)")
                
                # APOE4 breakdown
                if 'apoe4' in stats:
                    print(f"      • APOE4 allele distribution:")
                    for apoe4, apoe4_stats in stats['apoe4'].items():
                        if apoe4 != 'Unknown':
                            print(f"        - {apoe4}: {apoe4_stats['count']:,} ({apoe4_stats['percentage']:.1f}%)")
                
                # Study information
                if 'study' in stats:
                    print(f"      • Studies: {stats['study']['unique_studies']} unique studies")
                    print(f"        - Most represented: {stats['study']['most_common']} ({stats['study']['most_common_count']:,} subjects)")
        
        # Cross-biomarker comparisons
        print(f"\n🔄 CROSS-BIOMARKER COMPARISONS:")
        
        # Age comparison
        ages = []
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.statistics and 'age' in self.statistics[biomarker_key]:
                ages.append((self.biomarkers[biomarker_key], self.statistics[biomarker_key]['age']['mean']))
        
        if ages:
            ages.sort(key=lambda x: x[1])
            print(f"   • Age ranking (youngest to oldest):")
            for i, (name, age) in enumerate(ages, 1):
                print(f"     {i}. {name}: {age:.1f} years")
        
        # Subject count comparison
        counts = []
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.statistics:
                counts.append((self.biomarkers[biomarker_key], self.statistics[biomarker_key]['total_subjects']))
        
        if counts:
            counts.sort(key=lambda x: x[1], reverse=True)
            print(f"   • Subject count ranking (largest to smallest):")
            for i, (name, count) in enumerate(counts, 1):
                print(f"     {i}. {name}: {count:,} subjects")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Find the biomarker with most subjects
        if counts:
            largest_biomarker = counts[0]
            print(f"   • {largest_biomarker[0]} has the largest training population ({largest_biomarker[1]:,} subjects)")
        
        # Find age differences
        if len(ages) >= 2:
            age_diff = ages[-1][1] - ages[0][1]
            print(f"   • Age difference between youngest and oldest populations: {age_diff:.1f} years")
            print(f"     ({ages[0][0]}: {ages[0][1]:.1f} years vs {ages[-1][0]}: {ages[-1][1]:.1f} years)")
        
        # ADAS-specific insights
        if 'adas' in self.statistics:
            adas_stats = self.statistics['adas']
            if 'diagnosis' in adas_stats:
                mci_pct = adas_stats['diagnosis'].get('MCI', {}).get('percentage', 0)
                if mci_pct > 45:
                    print(f"   • ADAS has a MCI-focused population ({mci_pct:.1f}% MCI subjects)")
        
        # Study diversity insights
        multi_study_biomarkers = []
        single_study_biomarkers = []
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.statistics and 'study' in self.statistics[biomarker_key]:
                study_count = self.statistics[biomarker_key]['study']['unique_studies']
                if study_count > 1:
                    multi_study_biomarkers.append(self.biomarkers[biomarker_key])
                else:
                    single_study_biomarkers.append(self.biomarkers[biomarker_key])
        
        if multi_study_biomarkers:
            print(f"   • Multi-study biomarkers: {', '.join(multi_study_biomarkers)}")
        if single_study_biomarkers:
            print(f"   • Single-study biomarkers: {', '.join(single_study_biomarkers)}")
        
        # Data quality insights
        print(f"\n📋 DATA QUALITY ASSESSMENT:")
        for biomarker_key in self.biomarkers.keys():
            if biomarker_key in self.statistics:
                stats = self.statistics[biomarker_key]
                missing_apoe4 = stats.get('apoe4', {}).get('Unknown', {}).get('count', 0)
                total_subjects = stats.get('total_subjects', 0)
                if total_subjects > 0:
                    missing_pct = (missing_apoe4 / total_subjects) * 100
                    if missing_pct > 0:
                        print(f"   • {self.biomarkers[biomarker_key]}: {missing_pct:.1f}% missing APOE4 data")
                    else:
                        print(f"   • {self.biomarkers[biomarker_key]}: Complete APOE4 data")
        
        print("\n" + "="*80)
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("Generating comprehensive report...")
        
        report = f"""
# Population Demographics Analysis Report
## Training Subjects from Fold 0

### Overview
This report provides comprehensive demographic analysis of training subjects from fold 0 for each biomarker.

### Biomarkers Analyzed
"""
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.training_subjects:
                count = len(self.training_subjects[biomarker_key])
                report += f"- **{biomarker_name}**: {count} training subjects\n"
        
        report += f"""
### Key Findings

#### Subject Counts
"""
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.statistics:
                stats = self.statistics[biomarker_key]
                report += f"- **{biomarker_name}**: {stats.get('total_subjects', 0)} subjects\n"
        
        report += f"""
#### Age Distribution
"""
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.statistics and 'age' in self.statistics[biomarker_key]:
                age_stats = self.statistics[biomarker_key]['age']
                report += f"- **{biomarker_name}**: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years (range: {age_stats['min']:.1f}-{age_stats['max']:.1f})\n"
        
        report += f"""
#### Diagnosis Distribution
"""
        
        for biomarker_key, biomarker_name in self.biomarkers.items():
            if biomarker_key in self.statistics and 'diagnosis' in self.statistics[biomarker_key]:
                report += f"- **{biomarker_name}**:\n"
                for diag, diag_stats in self.statistics[biomarker_key]['diagnosis'].items():
                    report += f"  - {diag}: {diag_stats['count']} ({diag_stats['percentage']:.1f}%)\n"
        
        report += f"""
### Generated Files
- `age_distribution.png`: Age distribution comparison across biomarkers
- `diagnosis_distribution.png`: Diagnosis distribution pie charts
- `gender_distribution.png`: Gender distribution bar charts
- `apoe4_distribution.png`: APOE4 allele distribution
- `subject_overlap.png`: Subject overlap heatmap between biomarkers
- `demographics_summary.csv`: Comprehensive summary table
- `detailed_statistics.json`: Detailed statistics in JSON format

### Analysis Notes
- All analyses are based on baseline data (Time = 0)
- Training subjects are from fold 0 for each biomarker
- Demographics include age, gender, diagnosis, APOE4 status, education, and scanner information
- Subject overlap analysis shows common subjects across different biomarkers

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(self.output_dir / 'demographics_report.md', 'w') as f:
            f.write(report)
        
        print("  Saved: demographics_report.md")
    
    def run_analysis(self):
        """Run the complete demographics analysis"""
        print("="*60)
        print("POPULATION DEMOGRAPHICS ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_training_subjects_fold0()
        
        if not self.load_covariates_data():
            print("Error: Could not load covariates data. Analysis cannot proceed.")
            return
        
        # Extract demographics
        self.extract_baseline_demographics()
        
        # Print comprehensive summary
        self.print_comprehensive_summary()
        
        # Create visualizations
        self.create_demographic_plots()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

if __name__ == "__main__":
    analyzer = PopulationDemographicsAnalyzer()
    analyzer.run_analysis() 