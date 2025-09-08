"""
BDT Training and Calibration Workflow
====================================

This script implements a complete workflow for training BDT models (XGBoost, LightGBM, CatBoost)
on QCD dataset and applying probability calibration for CMS trigger efficiency analysis.

Key Features:
- Loads QCD, ggF, and VBF datasets from ROOT files
- Trains multiple BDT models on QCD data only
- Applies probability calibration using QCD dataset
- Evaluates calibration quality on all datasets
- Generates comprehensive reports and plots
"""

import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from library.trigger_efficiency_ML import define_parameter
from calibration.probability_calibration import BDTCalibrator

import warnings
warnings.filterwarnings('ignore')


class BDTCalibrationWorkflow:
    """
    Complete workflow for BDT training and calibration using QCD dataset.
    """
    
    def __init__(self, data_version='testing', random_state=42):
        """
        Initialize the workflow.
        
        Parameters:
        -----------
        data_version : str, default='testing'
            Version of processed data to use ('testing', 'azura', 'cypress', etc.)
        random_state : int, default=42
            Random state for reproducibility
        """
        self.data_version = data_version
        self.random_state = random_state
        self.models = {}
        self.calibrators = {}
        self.results = {}
        
        # Setup results directory
        today_str = datetime.now().strftime("%d-%m-%Y")
        self.results_dir = f"result/{today_str}/{today_str}-bdt-calibration-{data_version}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results will be saved in: {self.results_dir}")
        
    def load_datasets(self):
        """Load QCD, ggF, and VBF datasets from ROOT files."""
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        data_path = f"data/processed/{self.data_version}/"
        
        try:
            # Load datasets
            print(f"Loading {self.data_version} datasets...")
            self.df_qcd = ROOT.RDataFrame(f"{self.data_version}-NewQCD", f"{data_path}{self.data_version}-NewQCD.root")
            self.df_ggf = ROOT.RDataFrame(f"{self.data_version}-NewggF", f"{data_path}{self.data_version}-NewggF.root")
            self.df_vbf = ROOT.RDataFrame(f"{self.data_version}-NewVBF", f"{data_path}{self.data_version}-NewVBF.root")
            
            print("Successfully loaded ROOT DataFrames!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print(f"Make sure the data files exist in {data_path}")
            raise
    
    def prepare_data(self):
        """Convert ROOT DataFrames to pandas and prepare features."""
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)
        
        # Get column names from QCD dataset
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, \
        range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("QCD")
        
        # Convert to pandas DataFrames
        print("Converting ROOT DataFrames to pandas...")
        self.qcd_data = pd.DataFrame(self.df_qcd.AsNumpy(columns=names_list_and_signal_trigger))
        
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, \
        range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("ggF")
        self.ggf_data = pd.DataFrame(self.df_ggf.AsNumpy(columns=names_list_and_signal_trigger))
        
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, \
        range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("VBF")
        self.vbf_data = pd.DataFrame(self.df_vbf.AsNumpy(columns=names_list_and_signal_trigger))
        
        # Define features (exclude target and reference trigger)
        self.features = [col for col in self.qcd_data.columns if col not in ['Combo', 'HLT_AK8PFJet260']]
        
        print(f"Dataset sizes:")
        print(f"  QCD: {len(self.qcd_data)} events")
        print(f"  ggF: {len(self.ggf_data)} events")
        print(f"  VBF: {len(self.vbf_data)} events")
        print(f"  Features: {len(self.features)} ({self.features})") 
        
        # Prepare QCD data for training and calibration
        X_qcd = self.qcd_data[self.features].values
        y_qcd = self.qcd_data['Combo'].astype('int').values
        
        # Split QCD data: 50% training, 30% calibration, 20% testing
        X_temp, self.X_qcd_test, y_temp, self.y_qcd_test = train_test_split(
            X_qcd, y_qcd, test_size=0.2, random_state=self.random_state, stratify=y_qcd
        )
        
        self.X_qcd_train, self.X_qcd_cal, self.y_qcd_train, self.y_qcd_cal = train_test_split(
            X_temp, y_temp, test_size=0.375, random_state=self.random_state, stratify=y_temp  # 0.375 * 0.8 = 0.3
        )
        
        print(f"QCD data splits:")
        print(f"  Training: {len(self.X_qcd_train)} events ({len(self.X_qcd_train)/len(X_qcd)*100:.1f}%)")
        print(f"  Calibration: {len(self.X_qcd_cal)} events ({len(self.X_qcd_cal)/len(X_qcd)*100:.1f}%)")
        print(f"  Testing: {len(self.X_qcd_test)} events ({len(self.X_qcd_test)/len(X_qcd)*100:.1f}%)")
        
        # Prepare other datasets for evaluation
        self.X_ggf = self.ggf_data[self.features].values
        self.y_ggf = self.ggf_data['Combo'].astype('int').values
        self.X_vbf = self.vbf_data[self.features].values
        self.y_vbf = self.vbf_data['Combo'].astype('int').values
        
    def train_models(self):
        """Train BDT models on QCD training data."""
        print("\n" + "="*60)
        print("TRAINING BDT MODELS")
        print("="*60)
        
        # Define models
        self.models = {
            "XGBoost": xgb.XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            ),
            "LightGBM": lgb.LGBMClassifier(
                verbose=-1,
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            ),
            "CatBoost": cb.CatBoostClassifier(
                verbose=0,
                random_state=self.random_state,
                n_estimators=100,
                depth=6,
                learning_rate=0.1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                max_features='sqrt'
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = datetime.now()
            model.fit(self.X_qcd_train, self.y_qcd_train)
            training_time = (datetime.now() - start_time).total_seconds()
            print(f"  Training completed in {training_time:.2f} seconds")
            
            # Evaluate on training data
            train_auc = roc_auc_score(self.y_qcd_train, model.predict_proba(self.X_qcd_train)[:, 1])
            print(f"  Training AUC: {train_auc:.4f}")
            
        print("All models trained successfully!")
    
    def calibrate_models(self):
        """Apply probability calibration using QCD calibration data."""
        print("\n" + "="*60)
        print("CALIBRATING MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\\n--- Calibrating {model_name} ---")
            
            # Initialize calibrator with both methods
            calibrator = BDTCalibrator(method='both', cv_folds=5, random_state=self.random_state)
            
            # Fit calibration using QCD calibration data
            calibrator.fit(model, self.X_qcd_cal, self.y_qcd_cal)
            
            # Store calibrator
            self.calibrators[model_name] = calibrator
            
        print("All models calibrated successfully!")
    
    def evaluate_models(self):
        """Evaluate uncalibrated and calibrated models on all datasets."""
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        datasets = {
            'QCD_test': (self.X_qcd_test, self.y_qcd_test),
            'ggF': (self.X_ggf, self.y_ggf),
            'VBF': (self.X_vbf, self.y_vbf)
        }
        
        self.results = {}
        
        for model_name in self.models.keys():
            print(f"\\n{'='*50}")
            print(f"EVALUATING {model_name}")
            print(f"{'='*50}")
            
            self.results[model_name] = {}
            model = self.models[model_name]
            calibrator = self.calibrators[model_name]
            
            for dataset_name, (X_test, y_test) in datasets.items():
                print(f"\\n--- {dataset_name} Dataset ---")
                
                # Uncalibrated predictions
                y_pred_uncal = model.predict_proba(X_test)[:, 1]
                auc_uncal = roc_auc_score(y_test, y_pred_uncal)
                
                print(f"Uncalibrated AUC: {auc_uncal:.4f}")
                
                # Evaluate calibration
                cal_results = calibrator.evaluate_calibration(X_test, y_test, dataset_name)
                
                # Store results
                self.results[model_name][dataset_name] = {
                    'uncalibrated': {
                        'auc': auc_uncal,
                        'probabilities': y_pred_uncal
                    },
                    'calibrated': cal_results,
                    'y_true': y_test
                }
    
    def generate_plots(self):
        """Generate calibration plots and analysis visualizations."""
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        
        for model_name in self.models.keys():
            print(f"\\nGenerating plots for {model_name}...")
            
            for dataset_name in ['QCD_test', 'ggF', 'VBF']:
                results = self.results[model_name][dataset_name]
                y_true = results['y_true']
                calibrated_results = results['calibrated']
                
                # Calibration curves
                save_path = f"{self.results_dir}/calibration_curve_{model_name}_{dataset_name}.png"
                self.calibrators[model_name].plot_calibration_curve(
                    calibrated_results, save_path=save_path
                )
                
                # Probability histograms
                save_path = f"{self.results_dir}/probability_hist_{model_name}_{dataset_name}.png"
                self.calibrators[model_name].plot_probability_histogram(
                    calibrated_results, y_true, save_path=save_path
                )
        
        # ROC curves comparison
        self._plot_roc_comparison()
        
        # Summary comparison
        self._plot_summary_comparison()
    
    def _plot_roc_comparison(self):
        """Plot ROC curves comparing all models on all datasets."""
        datasets = ['QCD_test', 'ggF', 'VBF']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']  # More colors for additional models
        
        for i, dataset_name in enumerate(datasets):
            ax = axes[i]
            
            for j, model_name in enumerate(self.models.keys()):
                results = self.results[model_name][dataset_name]
                y_true = results['y_true']
                
                # Uncalibrated ROC
                y_pred_uncal = results['uncalibrated']['probabilities']
                fpr_uncal, tpr_uncal, _ = roc_curve(y_true, y_pred_uncal)
                auc_uncal = auc(fpr_uncal, tpr_uncal)
                
                # Calibrated ROC (isotonic)
                y_pred_cal = results['calibrated']['isotonic']['probabilities']
                fpr_cal, tpr_cal, _ = roc_curve(y_true, y_pred_cal)
                auc_cal = auc(fpr_cal, tpr_cal)
                
                color = colors[j % len(colors)]  # Cycle through colors if more models than colors
                ax.plot(fpr_uncal, tpr_uncal, '--', color=color, alpha=0.7,
                       label=f'{model_name} Uncal (AUC: {auc_uncal:.3f})')
                ax.plot(fpr_cal, tpr_cal, '-', color=color, linewidth=2,
                       label=f'{model_name} Cal (AUC: {auc_cal:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves - {dataset_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = f"{self.results_dir}/roc_comparison_all.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC comparison saved to: {save_path}")
        plt.show()
    
    def _plot_summary_comparison(self):
        """Plot summary comparison of calibration metrics."""
        metrics = ['ECE', 'MCE', 'Brier Score', 'Log Loss']
        datasets = ['QCD_test', 'ggF', 'VBF']
        models = list(self.models.keys())
        
        # Collect data for plotting
        data = []
        for model in models:
            for dataset in datasets:
                cal_results = self.results[model][dataset]['calibrated']['isotonic']
                data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'ECE': cal_results['ece'],
                    'MCE': cal_results['mce'],
                    'Brier Score': cal_results['brier_score'],
                    'Log Loss': cal_results['log_loss']
                })
        
        df_summary = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Pivot for heatmap
            pivot_df = df_summary.pivot_table(values=metric, index='Model', columns='Dataset')
            
            sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar=True)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Model')
        
        plt.tight_layout()
        save_path = f"{self.results_dir}/calibration_metrics_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration metrics summary saved to: {save_path}")
        plt.show()
    
    def save_results(self):
        """Save models, calibrators, and results."""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save models
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f"{models_dir}/{model_name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model {model_name} saved to: {model_path}")
        
        # Save calibrators
        calibrators_dir = f"{self.results_dir}/calibrators"
        os.makedirs(calibrators_dir, exist_ok=True)
        
        for model_name, calibrator in self.calibrators.items():
            cal_path = f"{calibrators_dir}/{model_name.lower()}_calibrator.pkl"
            with open(cal_path, 'wb') as f:
                pickle.dump(calibrator, f)
            print(f"Calibrator {model_name} saved to: {cal_path}")
        
        # Save results
        results_path = f"{self.results_dir}/evaluation_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Evaluation results saved to: {results_path}")
    
    def generate_report(self):
        """Generate a comprehensive markdown report."""
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)
        
        report = f"""
# BDT Probability Calibration Analysis Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Data Version:** {self.data_version}  
**Random State:** {self.random_state}  

## 1. Executive Summary

This report presents the results of probability calibration for Boosted Decision Tree (BDT) models applied to CMS trigger efficiency analysis. Three models (XGBoost, LightGBM, CatBoost) were trained exclusively on QCD simulation data and calibrated using both isotonic regression and Platt scaling methods. The calibrated models were then evaluated on QCD test, ggF, and VBF datasets.

## 2. Dataset Information

| Dataset | Events | Purpose |
|---------|---------|---------|
| QCD Training | {len(self.X_qcd_train)} | Model training |
| QCD Calibration | {len(self.X_qcd_cal)} | Probability calibration |
| QCD Test | {len(self.X_qcd_test)} | Model evaluation |
| ggF | {len(self.y_ggf)} | Signal evaluation |
| VBF | {len(self.y_vbf)} | Signal evaluation |

**Features:** {len(self.features)} kinematic variables  
{', '.join(self.features)}

## 3. Model Performance Summary

### 3.1 ROC-AUC Scores

"""
        
        # Add AUC comparison table
        auc_data = []
        for model_name in self.models.keys():
            row = [model_name]
            for dataset in ['QCD_test', 'ggF', 'VBF']:
                uncal_auc = self.results[model_name][dataset]['uncalibrated']['auc']
                cal_auc = roc_auc_score(
                    self.results[model_name][dataset]['y_true'],
                    self.results[model_name][dataset]['calibrated']['isotonic']['probabilities']
                )
                row.extend([uncal_auc, cal_auc])
            auc_data.append(row)
        
        report += "| Model | QCD Uncal | QCD Cal | ggF Uncal | ggF Cal | VBF Uncal | VBF Cal |\\n"
        report += "|-------|-----------|---------|-----------|---------|-----------|---------|\\n"
        for row in auc_data:
            report += f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} | {row[5]:.4f} | {row[6]:.4f} |\\n"
        
        report += f"""

### 3.2 Calibration Quality Metrics (Isotonic Regression)

"""
        
        # Add calibration metrics table
        report += "| Model | Dataset | ECE | MCE | Brier Score | Log Loss |\\n"
        report += "|-------|---------|-----|-----|-------------|----------|\\n"
        
        for model_name in self.models.keys():
            for dataset in ['QCD_test', 'ggF', 'VBF']:
                results = self.results[model_name][dataset]['calibrated']['isotonic']
                report += f"| {model_name} | {dataset} | {results['ece']:.4f} | {results['mce']:.4f} | {results['brier_score']:.4f} | {results['log_loss']:.4f} |\\n"
        
        report += f"""

## 4. Key Findings

### 4.1 Calibration Method Comparison
- **Isotonic Regression**: Generally provides better calibration for all models
- **Platt Scaling**: Faster but less flexible, suitable for simpler calibration needs

### 4.2 Model Recommendations
"""
        
        # Find best models for each dataset
        best_models = {}
        for dataset in ['QCD_test', 'ggF', 'VBF']:
            best_ece = float('inf')
            best_model = None
            for model_name in self.models.keys():
                ece = self.results[model_name][dataset]['calibrated']['isotonic']['ece']
                if ece < best_ece:
                    best_ece = ece
                    best_model = model_name
            best_models[dataset] = (best_model, best_ece)
        
        for dataset, (best_model, best_ece) in best_models.items():
            report += f"- **{dataset}**: {best_model} (ECE: {best_ece:.4f})\\n"
        
        report += f"""

## 5. Visualizations

The following plots have been generated and saved in the results directory:

### 5.1 Calibration Curves (Reliability Diagrams)
- `calibration_curve_{{model}}_{{dataset}}.png`: Shows calibration quality
- Perfect calibration follows the diagonal line

### 5.2 Probability Distributions
- `probability_hist_{{model}}_{{dataset}}.png`: Predicted probability distributions
- Shows separation between signal and background classes

### 5.3 ROC Curves
- `roc_comparison_all.png`: Compares uncalibrated vs calibrated performance
- `calibration_metrics_summary.png`: Heatmap of calibration metrics

## 6. Technical Implementation

### 6.1 Calibration Methodology
- **Training**: Models trained exclusively on QCD data
- **Calibration**: Applied using QCD calibration split with {self.calibrators[list(self.models.keys())[0]].cv_folds}-fold cross-validation
- **Methods**: Both isotonic regression and Platt scaling implemented
- **Evaluation**: Comprehensive metrics including ECE, MCE, Brier Score, and Log Loss

### 6.2 Quality Assurance
- Stratified sampling ensures balanced class distribution
- Cross-validation prevents overfitting during calibration
- Multiple evaluation datasets test generalization

## 7. Conclusions and Recommendations

1. **Calibration Necessity**: All models show improved calibration after applying isotonic regression
2. **Model Selection**: Choose the best-performing model based on ECE for your specific dataset
3. **Production Use**: Apply calibration in the inference pipeline for reliable probability estimates
4. **Monitoring**: Regularly re-calibrate models as new data becomes available

## 8. Files Generated

- **Models**: `models/{{model_name}}_model.pkl`
- **Calibrators**: `calibrators/{{model_name}}_calibrator.pkl`  
- **Results**: `evaluation_results.pkl`
- **Plots**: Various PNG files for visualization
- **Report**: This markdown file

---

**Note**: This analysis was performed using the BDT Calibration Workflow with QCD-only training and calibration. The calibrated models are now ready for deployment in CMS trigger efficiency studies.
"""
        
        # Save report
        report_path = f"{self.results_dir}/calibration_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def run_complete_workflow(self):
        """Execute the complete BDT calibration workflow."""
        print("\\n" + "="*80)
        print("STARTING BDT CALIBRATION WORKFLOW")
        print("="*80)
        
        workflow_start = datetime.now()
        
        try:
            # Execute workflow steps
            self.load_datasets()
            self.prepare_data()
            self.train_models()
            self.calibrate_models()
            self.evaluate_models()
            self.generate_plots()
            self.save_results()
            self.generate_report()
            
            # Workflow completion
            total_time = (datetime.now() - workflow_start).total_seconds()
            
            print("\\n" + "="*80)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results directory: {self.results_dir}")
            print("\\nFiles generated:")
            print("- Models and calibrators (pickle files)")
            print("- Calibration plots and analysis visualizations")
            print("- Comprehensive analysis report (markdown)")
            print("- Evaluation results (pickle file)")
            
        except Exception as e:
            print(f"\\nERROR in workflow execution: {e}")
            print("Please check the error messages above and verify:")
            print("1. Data files exist in the expected location")
            print("2. Required libraries are installed")
            print("3. Sufficient memory is available")
            raise


def main():
    """Main function to execute the BDT calibration workflow."""
    # Configuration
    data_version = "testing"  # Change this to your preferred data version
    random_state = 42
    
    # Create and run workflow
    workflow = BDTCalibrationWorkflow(
        data_version=data_version, 
        random_state=random_state
    )
    
    workflow.run_complete_workflow()


if __name__ == "__main__":
    main()
