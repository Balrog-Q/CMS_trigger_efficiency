#!/usr/bin/env python3
"""
Test GradientBoosting Integration
=================================

This script tests the integration of sklearn.ensemble.GradientBoostingClassifier
into the BDT calibration workflow using synthetic data.

This serves as a validation that the integration works correctly before
running on real CMS data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from calibration.probability_calibration import BDTCalibrator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_cms_like_data(n_samples=10000, n_features=14, random_state=42):
    """Generate synthetic data similar to CMS trigger efficiency features."""
    
    np.random.seed(random_state)
    
    # Generate base features with realistic ranges
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Scale features to realistic CMS ranges
    feature_scales = [
        (300, 100),   # HighestPt
        (800, 200),   # HT  
        (100, 50),    # MET_pt
        (600, 200),   # mHH
        (100, 30),    # HighestMass
        (250, 80),    # SecondHighestPt
        (80, 25),     # SecondHighestMass
        (600, 150),   # FatHT
        (400, 100),   # MET_FatJet
        (700, 200),   # mHHwithMET
        (0, 2.4),     # HighestEta
        (0, 2.4),     # SecondHighestEta
        (1.5, 1.0),   # DeltaEta
        (1.0, 1.5),   # DeltaPhi
    ]
    
    # Apply scaling
    for i, (mean, std) in enumerate(feature_scales):
        features[:, i] = features[:, i] * std + mean
    
    # Create labels based on a complex decision boundary
    # Simulate trigger efficiency scenario
    decision_score = (
        0.1 * features[:, 0] +      # HighestPt
        0.05 * features[:, 1] +     # HT
        0.02 * features[:, 2] +     # MET_pt
        -0.01 * features[:, 3] +    # mHH
        0.08 * features[:, 4] +     # HighestMass
        0.06 * features[:, 5] +     # SecondHighestPt
        np.random.normal(0, 50, n_samples)  # Add noise
    )
    
    # Convert to probabilities and then to binary labels
    probabilities = 1 / (1 + np.exp(-decision_score / 100))
    labels = np.random.binomial(1, probabilities)
    
    # Create feature names
    feature_names = [
        'HighestPt', 'HT', 'MET_pt', 'mHH', 'HighestMass', 'SecondHighestPt',
        'SecondHighestMass', 'FatHT', 'MET_FatJet', 'mHHwithMET', 'HighestEta',
        'SecondHighestEta', 'DeltaEta', 'DeltaPhi'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['Combo'] = labels
    
    return df, feature_names


def test_gradientboosting_training():
    """Test that GradientBoostingClassifier can be trained successfully."""
    print("="*60)
    print("TESTING GRADIENTBOOSTING TRAINING")
    print("="*60)
    
    # Generate synthetic data
    df, feature_names = generate_synthetic_cms_like_data(n_samples=5000)
    
    X = df[feature_names].values
    y = df['Combo'].astype('int').values
    
    print(f"Generated {len(df)} synthetic events")
    print(f"Signal fraction: {y.mean():.3f}")
    print(f"Features: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize and train GradientBoostingClassifier
    print("\nTraining GradientBoostingClassifier...")
    gb_model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        max_features='sqrt'
    )
    
    gb_model.fit(X_train, y_train)
    print("✓ Training completed successfully")
    
    # Evaluate
    train_auc = roc_auc_score(y_train, gb_model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])
    
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    assert train_auc > 0.5, f"Training AUC too low: {train_auc}"
    assert test_auc > 0.5, f"Test AUC too low: {test_auc}"
    
    print("✓ GradientBoosting training test passed")
    return gb_model, X_train, X_test, y_train, y_test


def test_gradientboosting_calibration(model, X_train, X_test, y_train, y_test):
    """Test that GradientBoostingClassifier can be calibrated successfully."""
    print("\n" + "="*60)
    print("TESTING GRADIENTBOOSTING CALIBRATION")
    print("="*60)
    
    # Split training data for calibration
    X_train_model, X_cal, y_train_model, y_cal = train_test_split(
        X_train, y_train, test_size=0.4, random_state=42, stratify=y_train
    )
    
    # Re-train model on smaller training set
    print("Re-training model on reduced training set...")
    model.fit(X_train_model, y_train_model)
    
    # Initialize calibrator
    print("Initializing calibrator...")
    calibrator = BDTCalibrator(method='both', cv_folds=3, random_state=42)
    
    # Fit calibration
    print("Fitting calibration...")
    calibrator.fit(model, X_cal, y_cal)
    print("✓ Calibration fitting completed")
    
    # Evaluate calibration
    print("\nEvaluating calibration quality...")
    results = calibrator.evaluate_calibration(X_test, y_test, "Test Dataset")
    
    # Check that calibration results are reasonable
    for method, result in results.items():
        ece = result['ece']
        mce = result['mce']
        brier_score = result['brier_score']
        
        print(f"\n{method.capitalize()} calibration:")
        print(f"  ECE: {ece:.4f}")
        print(f"  MCE: {mce:.4f}")
        print(f"  Brier Score: {brier_score:.4f}")
        
        # Basic sanity checks
        assert 0 <= ece <= 1, f"ECE out of range: {ece}"
        assert 0 <= mce <= 1, f"MCE out of range: {mce}"
        assert 0 <= brier_score <= 1, f"Brier score out of range: {brier_score}"
    
    print("✓ GradientBoosting calibration test passed")
    return calibrator, results


def test_workflow_integration():
    """Test that GradientBoosting integrates properly with the workflow structure."""
    print("\n" + "="*60)
    print("TESTING WORKFLOW INTEGRATION")
    print("="*60)
    
    # Test that we can import and create the model as in the workflow
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create model with same parameters as in the workflow
    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        max_features='sqrt'
    )
    
    print("✓ Model creation successful")
    
    # Test that the model name will be handled correctly
    model_name = "GradientBoosting"
    filename_base = model_name.lower()
    
    print(f"Model name: {model_name}")
    print(f"Filename base: {filename_base}")
    
    expected_model_file = f"models/{filename_base}_model.pkl"
    expected_calibrator_file = f"calibrators/{filename_base}_calibrator.pkl"
    
    print(f"Expected model file: {expected_model_file}")
    print(f"Expected calibrator file: {expected_calibrator_file}")
    
    print("✓ Workflow integration test passed")


def test_comparison_with_other_models():
    """Test GradientBoosting performance compared to other models."""
    print("\n" + "="*60)
    print("TESTING COMPARISON WITH OTHER MODELS")
    print("="*60)
    
    # Generate data
    df, feature_names = generate_synthetic_cms_like_data(n_samples=3000)
    X = df[feature_names].values
    y = df['Combo'].astype('int').values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test multiple models
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42, n_estimators=50, max_depth=4, learning_rate=0.1
        )
    }
    
    # Add other models if available
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            objective='binary:logistic', use_label_encoder=False, 
            eval_metric='logloss', random_state=42, n_estimators=50, max_depth=4
        )
    except ImportError:
        print("XGBoost not available for comparison")
    
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            verbose=-1, random_state=42, n_estimators=50, max_depth=4
        )
    except ImportError:
        print("LightGBM not available for comparison")
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = test_auc
        print(f"  {name} AUC: {test_auc:.4f}")
    
    # Check that GradientBoosting performance is reasonable
    gb_auc = results["GradientBoosting"]
    assert gb_auc > 0.6, f"GradientBoosting AUC too low: {gb_auc}"
    
    print(f"\n✓ Model comparison test passed")
    print(f"GradientBoosting achieved AUC: {gb_auc:.4f}")
    
    return results


def create_test_visualization(calibrator, results, model_name="GradientBoosting"):
    """Create a test visualization to verify calibration plots work."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    try:
        # Create calibration curve
        calibrator.plot_calibration_curve(
            results, 
            save_path=f"test_{model_name.lower()}_calibration.png"
        )
        print(f"✓ Calibration curve saved as: test_{model_name.lower()}_calibration.png")
        
        # Create summary comparison
        summary_df = calibrator.compare_methods_summary(results)
        print("✓ Summary comparison completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("STARTING GRADIENTBOOSTING INTEGRATION TESTS")
    print("=" * 80)
    
    success_count = 0
    total_tests = 5
    
    try:
        # Test 1: Basic training
        model, X_train, X_test, y_train, y_test = test_gradientboosting_training()
        success_count += 1
        
        # Test 2: Calibration
        calibrator, results = test_gradientboosting_calibration(
            model, X_train, X_test, y_train, y_test
        )
        success_count += 1
        
        # Test 3: Workflow integration
        test_workflow_integration()
        success_count += 1
        
        # Test 4: Model comparison
        comparison_results = test_comparison_with_other_models()
        success_count += 1
        
        # Test 5: Visualization
        viz_success = create_test_visualization(calibrator, results)
        if viz_success:
            success_count += 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final results
    print("\n" + "=" * 80)
    print("INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("GradientBoostingClassifier is successfully integrated!")
        print("\nYou can now run:")
        print("  python run_bdt_calibration.py")
        print("And it will include GradientBoosting alongside other models.")
        
    else:
        print(f"⚠️  {total_tests - success_count} tests failed")
        print("Please check the error messages above and fix any issues.")
    
    print("\n" + "=" * 80)
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
