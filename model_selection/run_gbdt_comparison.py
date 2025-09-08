#!/usr/bin/env python3
"""
GBDT Comparison - Usage Example

This script demonstrates how to use the GBDT comparison framework
to reproduce the methodology described in your research.

Usage:
    python run_gbdt_comparison.py

The script will:
1. Load QCD data from ROOT file (or generate simulated data if file not found)
2. Run Stage 1: Hyperparameter tuning using TPE from Optuna
3. Run Stage 2: Train ensemble models with optimal parameters
4. Evaluate all algorithms on test set
5. Save results and models
"""

import sys
import os
import numpy as np
from datetime import datetime

# Import the clean pipeline implementation
from pipeline_clean import run_gbdt_comparison, load_qcd_data_exact


def main():
    """Main function to run the GBDT comparison."""
    print("GBDT Algorithm Comparison for PID at MPD Experiment")
    print("=" * 60)
    print("\nImplementing the research methodology:")
    print("- XGBoost 3.0.3, LightGBM 4.6.0, CatBoost 1.2.68, GradientBoostingClassifier 1.7.0")
    print("- 5-fold cross-validation for robust performance")
    print("- Two-stage approach: hyperparameter tuning + model evaluation")
    print("- TPE algorithm from Optuna for hyperparameter optimization")
    print("- Ensemble averaging of classifier outputs")
    print("\nStage configurations:")
    print("- Stage 1 (Parameter tuning): LR=0.05, Max iter=5000, Early stop=200")
    print("- Stage 2 (Model evaluation): LR=0.015, Max iter=20000, Early stop=500")
    print()
    
    # Configuration
    root_file_path = "data/processed/azura/azura-NewQCD.root"
    n_trials = 50  # Number of optimization trials per algorithm
    test_size = 0.2  # 20% for testing
    
    print(f"Configuration:")
    print(f"  ROOT file: {root_file_path}")
    print(f"  Optimization trials per algorithm: {n_trials}")
    print(f"  Test set size: {test_size * 100}%")
    print(f"  Random seed: 42")
    print()
    
    try:
        # Run the complete comparison
        results = run_gbdt_comparison(
            root_file_path=root_file_path,
            n_trials=n_trials,
            test_size=test_size,
            save_results=True
        )
        
        # Print summary of results
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\nDataset Information:")
        print(f"  Total features: {results['data_info']['features']}")
        print(f"  Training samples: {results['data_info']['train_samples']:,}")
        print(f"  Test samples: {results['data_info']['test_samples']:,}")
        
        print(f"\nTest Set Results (Final Efficiency Metrics):")
        algorithms = ['xgboost', 'lightgbm', 'catboost', 'gbm']
        algorithm_names = ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']
        
        for i, algorithm in enumerate(algorithms):
            if algorithm in results['test_results']:
                res = results['test_results'][algorithm]
                print(f"  {algorithm_names[i]}:")
                print(f"    Accuracy: {res['accuracy']:.4f}")
                print(f"    AUC: {res['auc']:.4f}")
                print(f"    Log Loss: {res['log_loss']:.4f}")
        
        best_algorithm = results['best_algorithm']
        best_auc = results['test_results'][best_algorithm]['auc']
        algorithm_name_mapping = dict(zip(algorithms, algorithm_names))
        
        print(f"\nBest Performing Algorithm: {algorithm_name_mapping[best_algorithm]}")
        print(f"Test AUC: {best_auc:.4f}")
        
        print(f"\nOptimal Hyperparameters Found:")
        for algorithm, params in results['best_parameters'].items():
            print(f"  {algorithm_name_mapping[algorithm]}:")
            for param_name, param_value in params.items():
                if isinstance(param_value, float):
                    print(f"    {param_name}: {param_value:.6f}")
                else:
                    print(f"    {param_name}: {param_value}")
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the ROOT file exists at the specified path")
        print("2. Check that all required libraries are installed:")
        print("   pip install xgboost lightgbm catboost scikit-learn optuna pandas numpy")
        print("3. Ensure ROOT and the library.plotting_tools module are available")
        return None


def quick_test_with_simulated_data():
    """Run a quick test using simulated data to verify the implementation."""
    print("Running quick test with simulated data...")
    
    # Generate simulated QCD-like data
    np.random.seed(42)
    n_samples = 5000
    n_features = 14
    
    # Create features with some correlation structure
    X = np.random.randn(n_samples, n_features)
    
    # Create target with realistic class imbalance
    linear_combination = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.5
    y = (linear_combination > np.percentile(linear_combination, 70)).astype(int)
    
    print(f"Simulated data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run quick comparison with fewer trials
    from pipeline_clean import GBDTComparisonPipeline
    
    pipeline = GBDTComparisonPipeline(random_state=42)
    results = pipeline.run_comparison(X, y, test_size=0.2, n_trials=10)
    
    print("\nQuick test completed successfully!")
    return results


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run full comparison with ROOT data")
    print("2. Run quick test with simulated data")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        results = quick_test_with_simulated_data()
    else:
        results = main()
    
    if results:
        print(f"\nResults available in the 'results' variable")
        print("You can access:")
        print("- results['best_parameters'] for optimal hyperparameters")
        print("- results['test_results'] for test set evaluation")
        print("- results['cv_results'] for cross-validation results")
