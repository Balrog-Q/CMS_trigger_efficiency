"""
Probability Calibration Module for BDT Models
=============================================

This module implements probability calibration techniques for Boosted Decision Tree (BDT) models
used in CMS trigger efficiency analysis. It provides calibration using both Platt scaling and
isotonic regression methods, specifically designed to work with QCD dataset for calibration.

Key Features:
- Platt scaling (Sigmoid) calibration
- Isotonic regression calibration
- Cross-validation based calibration
- Calibration quality evaluation metrics
- Reliability diagrams and calibration plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')


class BDTCalibrator:
    """
    A comprehensive calibration class for BDT models using QCD dataset only.
    
    This class provides multiple calibration methods and evaluation metrics
    specifically designed for particle physics trigger efficiency studies.
    """
    
    def __init__(self, method='isotonic', cv_folds=5, random_state=42):
        """
        Initialize the BDT calibrator.
        
        Parameters:
        -----------
        method : str, default='isotonic'
            Calibration method. Options: 'isotonic', 'sigmoid', 'both'
        cv_folds : int, default=5
            Number of cross-validation folds for calibration
        random_state : int, default=42
            Random state for reproducibility
        """
        self.method = method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.calibrators = {}
        self.is_fitted = False
        
    def fit(self, model, X_qcd, y_qcd):
        """
        Fit calibration on QCD dataset using cross-validation.
        
        Parameters:
        -----------
        model : sklearn-compatible model
            The trained BDT model to calibrate
        X_qcd : array-like, shape (n_samples, n_features)
            QCD features for calibration
        y_qcd : array-like, shape (n_samples,)
            QCD target labels for calibration
            
        Returns:
        --------
        self : BDTCalibrator
            Returns self for method chaining
        """
        print(f"Fitting calibration using {self.method} method on QCD dataset...")
        print(f"QCD dataset size: {X_qcd.shape[0]} events")
        
        # Stratified K-fold to ensure balanced splits
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                             random_state=self.random_state)
        
        if self.method == 'both':
            methods = ['isotonic', 'sigmoid']
        else:
            methods = [self.method]
            
        for cal_method in methods:
            print(f"Training {cal_method} calibrator...")
            calibrator = CalibratedClassifierCV(
                model, 
                method=cal_method, 
                cv=skf,
                n_jobs=-1
            )
            calibrator.fit(X_qcd, y_qcd)
            self.calibrators[cal_method] = calibrator
            
        self.is_fitted = True
        print("Calibration fitting completed!")
        return self
    
    def predict_proba(self, X, method=None):
        """
        Predict calibrated probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        method : str, optional
            Specific calibration method to use. If None, uses self.method
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples,)
            Calibrated probabilities for the positive class
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before predicting!")
            
        if method is None:
            method = self.method if self.method != 'both' else 'isotonic'
            
        if method not in self.calibrators:
            raise ValueError(f"Method {method} not available. Available: {list(self.calibrators.keys())}")
            
        return self.calibrators[method].predict_proba(X)[:, 1]
    
    def evaluate_calibration(self, X_test, y_test, dataset_name="Test", n_bins=10):
        """
        Evaluate calibration quality using multiple metrics.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test features
        y_test : array-like, shape (n_samples,)
            True test labels
        dataset_name : str, default="Test"
            Name of the dataset for reporting
        n_bins : int, default=10
            Number of bins for calibration curve
            
        Returns:
        --------
        results : dict
            Dictionary containing calibration metrics
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation!")
            
        results = {}
        
        for method_name, calibrator in self.calibrators.items():
            print(f"\nEvaluating {method_name} calibration on {dataset_name} dataset...")
            
            # Get calibrated probabilities
            y_prob_cal = calibrator.predict_proba(X_test)[:, 1]
            
            # Calibration curve
            fraction_pos, mean_pred_value = calibration_curve(
                y_test, y_prob_cal, n_bins=n_bins, strategy='uniform'
            )
            
            # Calibration metrics
            brier_score = brier_score_loss(y_test, y_prob_cal)
            log_loss_score = log_loss(y_test, y_prob_cal)
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_test, y_prob_cal, n_bins=n_bins)
            
            # Maximum Calibration Error (MCE)
            mce = self._calculate_mce(y_test, y_prob_cal, n_bins=n_bins)
            
            # Reliability-Resolution decomposition
            reliability, resolution = self._reliability_resolution(y_test, y_prob_cal, n_bins=n_bins)
            
            results[method_name] = {
                'brier_score': brier_score,
                'log_loss': log_loss_score,
                'ece': ece,
                'mce': mce,
                'reliability': reliability,
                'resolution': resolution,
                'calibration_curve': (fraction_pos, mean_pred_value),
                'probabilities': y_prob_cal
            }
            
            print(f"  Brier Score: {brier_score:.4f}")
            print(f"  Log Loss: {log_loss_score:.4f}")
            print(f"  ECE: {ece:.4f}")
            print(f"  MCE: {mce:.4f}")
            print(f"  Reliability: {reliability:.4f}")
            print(f"  Resolution: {resolution:.4f}")
            
        return results
    
    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _calculate_mce(self, y_true, y_prob, n_bins=10):
        """Calculate Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
                
        return max_error
    
    def _reliability_resolution(self, y_true, y_prob, n_bins=10):
        """Calculate reliability and resolution components."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        base_rate = y_true.mean()
        reliability = 0
        resolution = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                resolution += prop_in_bin * (accuracy_in_bin - base_rate) ** 2
                
        return reliability, resolution
    
    def plot_calibration_curve(self, results_dict, save_path=None, figsize=(12, 8)):
        """
        Plot calibration curves for all methods.
        
        Parameters:
        -----------
        results_dict : dict
            Results from evaluate_calibration method
        save_path : str, optional
            Path to save the plot
        figsize : tuple, default=(12, 8)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (method, results) in enumerate(results_dict.items()):
            fraction_pos, mean_pred_value = results['calibration_curve']
            plt.plot(mean_pred_value, fraction_pos, 's-', 
                    color=colors[i % len(colors)], 
                    label=f'{method.capitalize()} (ECE: {results["ece"]:.3f})',
                    linewidth=2, markersize=6)
        
        plt.xlabel('Mean Predicted Probability', fontsize=14)
        plt.ylabel('Fraction of Positives', fontsize=14)
        plt.title('Calibration Curves (Reliability Diagram)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to: {save_path}")
        
        plt.show()
    
    def plot_probability_histogram(self, results_dict, y_true, save_path=None, figsize=(15, 10)):
        """
        Plot histograms of predicted probabilities.
        
        Parameters:
        -----------
        results_dict : dict
            Results from evaluate_calibration method
        y_true : array-like
            True labels
        save_path : str, optional
            Path to save the plot
        figsize : tuple, default=(15, 10)
            Figure size
        """
        n_methods = len(results_dict)
        fig, axes = plt.subplots(2, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        for i, (method, results) in enumerate(results_dict.items()):
            y_prob = results['probabilities']
            
            # Histogram for class 0 (background)
            axes[0, i].hist(y_prob[y_true == 0], bins=50, alpha=0.7, 
                           color='red', label='Background (y=0)', density=True)
            axes[0, i].set_title(f'{method.capitalize()}: Background', fontweight='bold')
            axes[0, i].set_xlabel('Predicted Probability')
            axes[0, i].set_ylabel('Density')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Histogram for class 1 (signal)
            axes[1, i].hist(y_prob[y_true == 1], bins=50, alpha=0.7, 
                           color='blue', label='Signal (y=1)', density=True)
            axes[1, i].set_title(f'{method.capitalize()}: Signal', fontweight='bold')
            axes[1, i].set_xlabel('Predicted Probability')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Probability histogram saved to: {save_path}")
            
        plt.show()
    
    def compare_methods_summary(self, results_dict):
        """
        Create a summary table comparing calibration methods.
        
        Parameters:
        -----------
        results_dict : dict
            Results from evaluate_calibration method
            
        Returns:
        --------
        summary_df : pd.DataFrame
            Summary table of calibration metrics
        """
        metrics = ['brier_score', 'log_loss', 'ece', 'mce', 'reliability', 'resolution']
        
        summary_data = {}
        for method, results in results_dict.items():
            summary_data[method.capitalize()] = [results[metric] for metric in metrics]
        
        summary_df = pd.DataFrame(
            summary_data, 
            index=['Brier Score', 'Log Loss', 'ECE', 'MCE', 'Reliability', 'Resolution']
        )
        
        print("\n" + "="*60)
        print("CALIBRATION METHODS COMPARISON SUMMARY")
        print("="*60)
        print(summary_df.round(4))
        print("="*60)
        
        # Highlight best methods
        print("\nBest Methods (lower is better for all metrics):")
        for metric in summary_df.index:
            best_method = summary_df.loc[metric].idxmin()
            best_value = summary_df.loc[metric].min()
            print(f"  {metric}: {best_method} ({best_value:.4f})")
        
        return summary_df


def demonstrate_calibration_workflow():
    """
    Demonstrate the complete calibration workflow with example usage.
    This is a template function showing how to use the BDTCalibrator class.
    """
    print("BDT Calibration Workflow Demonstration")
    print("=" * 50)
    
    # Example pseudo-code (replace with actual data loading)
    print("1. Load QCD dataset for calibration...")
    print("2. Train BDT model on QCD data...")
    print("3. Initialize calibrator...")
    print("4. Fit calibration using QCD dataset...")
    print("5. Evaluate on test datasets (QCD, ggF, VBF)...")
    print("6. Generate calibration plots and reports...")
    
    # Sample workflow structure
    workflow_code = '''
    # Example usage:
    from probability_calibration import BDTCalibrator
    import xgboost as xgb
    
    # 1. Load and prepare data
    qcd_data = load_qcd_data()  # Your data loading function
    X_qcd, y_qcd = prepare_features_and_labels(qcd_data)
    X_train, X_cal, y_train, y_cal = train_test_split(X_qcd, y_qcd, test_size=0.3)
    
    # 2. Train BDT model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    # 3. Initialize and fit calibrator
    calibrator = BDTCalibrator(method='both', cv_folds=5)
    calibrator.fit(model, X_cal, y_cal)
    
    # 4. Evaluate on different datasets
    for dataset_name, (X_test, y_test) in test_datasets.items():
        results = calibrator.evaluate_calibration(X_test, y_test, dataset_name)
        calibrator.plot_calibration_curve(results, f"calibration_{dataset_name}.png")
    '''
    
    print("\nWorkflow Code Structure:")
    print(workflow_code)


if __name__ == "__main__":
    demonstrate_calibration_workflow()
