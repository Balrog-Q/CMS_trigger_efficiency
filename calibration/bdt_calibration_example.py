"""
BDT Probability Calibration Example
==================================

This script demonstrates how to use the probability calibration system
for individual analysis or integration into existing workflows.

Key demonstrations:
- Loading pre-trained models and calibrators
- Applying calibration to new data
- Evaluating calibration quality
- Using calibrated models for predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from calibration.probability_calibration import BDTCalibrator
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_calibrated_model(results_dir, model_name):
    """Load a pre-trained model and its calibrator."""
    model_path = f"{results_dir}/models/{model_name.lower()}_model.pkl"
    calibrator_path = f"{results_dir}/calibrators/{model_name.lower()}_calibrator.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    return model, calibrator


def demonstrate_calibration_usage(results_dir="result/26-08-2025/26-08-2025-bdt-calibration-testing"):
    """Demonstrate how to use calibrated models for predictions."""
    
    print("="*60)
    print("BDT PROBABILITY CALIBRATION USAGE EXAMPLE")
    print("="*60)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Please run the calibration workflow first using:")
        print("  python run_bdt_calibration.py")
        return
    
    print(f"Loading models and calibrators from: {results_dir}")
    
    # Load evaluation results
    results_path = f"{results_dir}/evaluation_results.pkl"
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            evaluation_results = pickle.load(f)
    else:
        print("Evaluation results not found. Running basic example...")
        evaluation_results = None
    
    # Available models
    models = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]
    
    # Load models and calibrators
    loaded_models = {}
    loaded_calibrators = {}
    
    for model_name in models:
        try:
            model, calibrator = load_calibrated_model(results_dir, model_name)
            loaded_models[model_name] = model
            loaded_calibrators[model_name] = calibrator
            print(f"✓ Loaded {model_name} model and calibrator")
        except FileNotFoundError:
            print(f"✗ Could not load {model_name} (files not found)")
            continue
    
    if not loaded_models:
        print("No models could be loaded. Please run the calibration workflow first.")
        return
    
    print(f"\nSuccessfully loaded {len(loaded_models)} models")
    
    # Demonstrate usage scenarios
    print(f"\n" + "="*60)
    print("USAGE SCENARIOS")
    print("="*60)
    
    # Scenario 1: Compare uncalibrated vs calibrated predictions
    if evaluation_results:
        print(f"\n1. CALIBRATION QUALITY COMPARISON")
        print("-" * 40)
        
        datasets = ['QCD_test', 'ggF', 'VBF']
        
        for dataset in datasets:
            print(f"\nDataset: {dataset}")
            print(f"{'Model':<12} {'Uncal AUC':<12} {'Cal AUC':<12} {'ECE':<12} {'MCE':<12}")
            print("-" * 60)
            
            for model_name in loaded_models.keys():
                if model_name in evaluation_results and dataset in evaluation_results[model_name]:
                    uncal_auc = evaluation_results[model_name][dataset]['uncalibrated']['auc']
                    
                    # Calculate calibrated AUC
                    y_true = evaluation_results[model_name][dataset]['y_true']
                    cal_probs = evaluation_results[model_name][dataset]['calibrated']['isotonic']['probabilities']
                    cal_auc = roc_auc_score(y_true, cal_probs)
                    
                    # Get calibration metrics
                    ece = evaluation_results[model_name][dataset]['calibrated']['isotonic']['ece']
                    mce = evaluation_results[model_name][dataset]['calibrated']['isotonic']['mce']
                    
                    print(f"{model_name:<12} {uncal_auc:<12.4f} {cal_auc:<12.4f} {ece:<12.4f} {mce:<12.4f}")
    
    # Scenario 2: Show how to make predictions on new data
    print(f"\n2. MAKING PREDICTIONS ON NEW DATA")
    print("-" * 40)
    
    # Example: Create dummy data similar to your features
    print("Creating example feature data...")
    
    # These should match the features used in training
    # (You would replace this with your actual new data)
    example_features = [
        'HighestPt', 'HT', 'MET_pt', 'mHH', 'HighestMass', 'SecondHighestPt',
        'SecondHighestMass', 'FatHT', 'MET_FatJet', 'mHHwithMET', 'HighestEta',
        'SecondHighestEta', 'DeltaEta', 'DeltaPhi'
    ]
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    X_new = np.random.normal(0, 1, (n_samples, len(example_features)))
    
    # Make realistic feature values (approximate ranges from CMS data)
    X_new[:, 0] *= 100 + 300  # HighestPt
    X_new[:, 1] *= 200 + 800  # HT
    X_new[:, 2] *= 50 + 100   # MET_pt
    X_new[:, 3] *= 200 + 600  # mHH
    # ... (other features would be scaled similarly)
    
    print(f"Example data shape: {X_new.shape}")
    
    # Make predictions with each model
    for model_name, model in loaded_models.items():
        calibrator = loaded_calibrators[model_name]
        
        # Uncalibrated predictions
        uncal_probs = model.predict_proba(X_new)[:, 1]
        
        # Calibrated predictions (isotonic regression)
        cal_probs_iso = calibrator.predict_proba(X_new, method='isotonic')
        
        # Calibrated predictions (sigmoid/Platt scaling)
        if 'sigmoid' in calibrator.calibrators:
            cal_probs_sig = calibrator.predict_proba(X_new, method='sigmoid')
        else:
            cal_probs_sig = cal_probs_iso
        
        print(f"\n{model_name} predictions:")
        print(f"  Uncalibrated: mean={uncal_probs.mean():.3f}, std={uncal_probs.std():.3f}")
        print(f"  Isotonic:     mean={cal_probs_iso.mean():.3f}, std={cal_probs_iso.std():.3f}")
        print(f"  Sigmoid:      mean={cal_probs_sig.mean():.3f}, std={cal_probs_sig.std():.3f}")
    
    # Scenario 3: Show calibration visualization for one model
    print(f"\n3. CALIBRATION VISUALIZATION EXAMPLE")
    print("-" * 40)
    
    if evaluation_results:
        # Pick the first available model for demonstration
        demo_model = list(loaded_models.keys())[0]
        demo_dataset = 'QCD_test'
        
        if demo_model in evaluation_results and demo_dataset in evaluation_results[demo_model]:
            print(f"Creating calibration plot for {demo_model} on {demo_dataset}")
            
            results = evaluation_results[demo_model][demo_dataset]['calibrated']
            calibrator = loaded_calibrators[demo_model]
            
            # Create a simple calibration plot
            plt.figure(figsize=(10, 8))
            calibrator.plot_calibration_curve(
                results, 
                save_path=f"example_calibration_{demo_model}_{demo_dataset}.png"
            )
            
            print(f"Calibration plot saved as: example_calibration_{demo_model}_{demo_dataset}.png")
    
    # Scenario 4: Best practices for production use
    print(f"\n4. PRODUCTION USE RECOMMENDATIONS")
    print("-" * 40)
    
    if evaluation_results:
        # Find best model for each dataset based on ECE
        best_models = {}
        for dataset in ['QCD_test', 'ggF', 'VBF']:
            best_ece = float('inf')
            best_model = None
            
            for model_name in loaded_models.keys():
                if (model_name in evaluation_results and 
                    dataset in evaluation_results[model_name]):
                    ece = evaluation_results[model_name][dataset]['calibrated']['isotonic']['ece']
                    if ece < best_ece:
                        best_ece = ece
                        best_model = model_name
            
            if best_model:
                best_models[dataset] = (best_model, best_ece)
        
        print("Best models by dataset (based on ECE):")
        for dataset, (model_name, ece) in best_models.items():
            print(f"  {dataset}: {model_name} (ECE: {ece:.4f})")
        
        print(f"\nRecommendations:")
        print(f"1. Use isotonic regression for better calibration quality")
        print(f"2. Monitor calibration quality regularly with new data")
        print(f"3. Re-calibrate periodically as data distribution changes")
        print(f"4. Choose model based on your specific dataset requirements")
    
    # Scenario 5: Integration example
    print(f"\n5. INTEGRATION CODE EXAMPLE")
    print("-" * 40)
    
    integration_code = f'''
# Example integration into your analysis pipeline

def predict_with_calibration(model, calibrator, X_features):
    """
    Make calibrated predictions using a trained model.
    
    Parameters:
    -----------
    model : trained BDT model
    calibrator : fitted BDTCalibrator
    X_features : feature matrix
    
    Returns:
    --------
    calibrated_probabilities : array of calibrated probabilities
    """
    return calibrator.predict_proba(X_features, method='isotonic')

# Usage in your code:
# model, calibrator = load_calibrated_model(results_dir, 'XGBoost')
# probabilities = predict_with_calibration(model, calibrator, your_features)

# For trigger efficiency calculation:
def calculate_efficiency(X_features, model, calibrator, threshold=0.5):
    """Calculate trigger efficiency using calibrated probabilities."""
    cal_probs = calibrator.predict_proba(X_features, method='isotonic')
    return (cal_probs > threshold).mean()
'''
    
    print(integration_code)
    
    print(f"\n" + "="*60)
    print("EXAMPLE COMPLETED")
    print("="*60)
    print(f"This example demonstrated:")
    print(f"1. Loading pre-trained models and calibrators")
    print(f"2. Comparing uncalibrated vs calibrated performance")
    print(f"3. Making predictions on new data")
    print(f"4. Visualizing calibration quality")
    print(f"5. Best practices for production use")
    print(f"6. Integration code examples")
    
    return loaded_models, loaded_calibrators


def quick_calibration_demo():
    """Quick demonstration using synthetic data if no pre-trained models exist."""
    print("="*60)
    print("QUICK CALIBRATION DEMONSTRATION")
    print("="*60)
    print("This demonstration uses synthetic data to show calibration concepts")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    # Create somewhat realistic trigger efficiency scenario
    # Higher probability for events with higher feature values
    prob_signal = 1 / (1 + np.exp(-(X.sum(axis=1) - 2)))
    y = np.random.binomial(1, prob_signal)
    
    print(f"Generated {n_samples} synthetic events with {n_features} features")
    print(f"Signal fraction: {y.mean():.3f}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    print(f"Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")
    
    # Train a simple XGBoost model
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Apply calibration
    print("Applying calibration...")
    calibrator = BDTCalibrator(method='both', cv_folds=3, random_state=42)
    calibrator.fit(model, X_cal, y_cal)
    
    # Evaluate
    print("Evaluating calibration...")
    results = calibrator.evaluate_calibration(X_test, y_test, "Synthetic Test")
    
    # Show results
    calibrator.plot_calibration_curve(results, "synthetic_calibration_demo.png")
    print("Calibration plot saved as: synthetic_calibration_demo.png")
    
    # Compare methods
    summary = calibrator.compare_methods_summary(results)
    
    return model, calibrator, results


if __name__ == "__main__":
    # Try to run the full demonstration first
    try:
        loaded_models, loaded_calibrators = demonstrate_calibration_usage()
        
        if not loaded_models:
            print("\nNo pre-trained models found. Running quick synthetic demo...")
            quick_calibration_demo()
            
    except Exception as e:
        print(f"Error in main demonstration: {e}")
        print("\nRunning quick synthetic demo instead...")
        quick_calibration_demo()
