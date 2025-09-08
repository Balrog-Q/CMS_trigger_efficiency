"""
GBDT Comparison Framework for PID at MPD Experiment

This implementation follows the methodology described in the research paper:
- Two-stage training approach (hyperparameter tuning + model evaluation)
- Four GBDT algorithms: XGBoost, LightGBM, CatBoost, GradientBoostingClassifier
- 5-fold cross-validation with ensemble averaging
- TPE optimization from Optuna for hyperparameter tuning
- Early stopping during training process
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Machine Learning Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, log_loss

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# ROOT and plotting
import ROOT
from library.trigger_efficiency_ML import define_parameter, get_plot_directory, RD_to_pandas

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class GBDTComparison:
    """
    A comprehensive framework for comparing GBDT algorithms following the research methodology.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the GBDT comparison framework.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.algorithms = ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']
        
        # Training configuration from the research
        self.stage1_config = {
            'learning_rate': 0.05,
            'max_iterations': 5000,
            'early_stopping': 200
        }
        
        self.stage2_config = {
            'learning_rate': 0.015,
            'max_iterations': 20000,
            'early_stopping': 500
        }
        
        # Hyperparameter search spaces from the research tables
        self.hyperparameter_spaces = self._define_hyperparameter_spaces()
        
        # Storage for results
        self.best_params = {}
        self.trained_models = {}
        self.cv_results = {}
        self.ensemble_predictions = {}
        
    def _define_hyperparameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Define hyperparameter search spaces for each algorithm based on the research table.
        
        Returns:
            Dictionary containing hyperparameter spaces for each algorithm
        """
        spaces = {
            'XGBoost': {
                'max_depth': (3, 12),
                'min_child_weight': (1e-5, 10.0),
                'reg_lambda': (0.1, 50.0),  # L2 leaf regularization
                'subsample': (0.5, 1.0)     # Row sampling rate
            },
            'LightGBM': {
                'max_depth': (3, 12),
                'min_data_in_leaf': (1, 100),
                'reg_lambda': (0.1, 50.0),  # L2 leaf regularization
                'subsample': (0.5, 1.0)     # Row sampling rate
            },
            'CatBoost': {
                'depth': (3, 12),           # max_depth equivalent
                'min_data_in_leaf': (1, 100),
                'l2_leaf_reg': (0.1, 50.0), # L2 leaf regularization
                'subsample': (0.5, 1.0)     # Row sampling rate
            },
            'GradientBoosting': {
                'max_depth': (3, 12),
                'min_samples_leaf': (1, 100),
                'alpha': (0.1, 50.0),       # L2 equivalent (alpha parameter)
                'subsample': (0.5, 1.0)     # Row sampling rate
            }
        }
        return spaces
    
    def _create_model(self, algorithm: str, params: Dict[str, Any], stage: int) -> Any:
        """
        Create a model instance with specified parameters and stage configuration.
        
        Args:
            algorithm: Algorithm name
            params: Hyperparameters
            stage: Training stage (1 or 2)
            
        Returns:
            Configured model instance
        """
        config = self.stage1_config if stage == 1 else self.stage2_config
        
        if algorithm == 'XGBoost':
            return xgb.XGBClassifier(
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                reg_lambda=params['reg_lambda'],
                subsample=params['subsample'],
                learning_rate=config['learning_rate'],
                n_estimators=config['max_iterations'],
                early_stopping_rounds=config['early_stopping'],
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            )
            
        elif algorithm == 'LightGBM':
            return lgb.LGBMClassifier(
                max_depth=params['max_depth'],
                min_data_in_leaf=params['min_data_in_leaf'],
                reg_lambda=params['reg_lambda'],
                subsample=params['subsample'],
                learning_rate=config['learning_rate'],
                n_estimators=config['max_iterations'],
                early_stopping_rounds=config['early_stopping'],
                random_state=self.random_state,
                verbosity=-1
            )
            
        elif algorithm == 'CatBoost':
            return cb.CatBoostClassifier(
                depth=params['depth'],
                min_data_in_leaf=params['min_data_in_leaf'],
                l2_leaf_reg=params['l2_leaf_reg'],
                subsample=params['subsample'],
                learning_rate=config['learning_rate'],
                iterations=config['max_iterations'],
                early_stopping_rounds=config['early_stopping'],
                random_state=self.random_state,
                verbose=False
            )
            
        elif algorithm == 'GradientBoosting':
            return GradientBoostingClassifier(
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                alpha=params['alpha'],
                subsample=params['subsample'],
                learning_rate=config['learning_rate'],
                n_estimators=config['max_iterations'],
                random_state=self.random_state,
                validation_fraction=0.1,
                n_iter_no_change=config['early_stopping']
            )
    
    def _objective_function(self, trial: optuna.Trial, algorithm: str, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            algorithm: Algorithm name
            X: Feature matrix
            y: Target vector
            
        Returns:
            Average cross-validation score (negative log loss)
        """
        # Suggest hyperparameters based on the search space
        space = self.hyperparameter_spaces[algorithm]
        
        if algorithm == 'XGBoost':
            params = {
                'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
                'min_child_weight': trial.suggest_float('min_child_weight', *space['min_child_weight'], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *space['reg_lambda']),
                'subsample': trial.suggest_float('subsample', *space['subsample'])
            }
        elif algorithm == 'LightGBM':
            params = {
                'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *space['min_data_in_leaf']),
                'reg_lambda': trial.suggest_float('reg_lambda', *space['reg_lambda']),
                'subsample': trial.suggest_float('subsample', *space['subsample'])
            }
        elif algorithm == 'CatBoost':
            params = {
                'depth': trial.suggest_int('depth', *space['depth']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *space['min_data_in_leaf']),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *space['l2_leaf_reg']),
                'subsample': trial.suggest_float('subsample', *space['subsample'])
            }
        elif algorithm == 'GradientBoosting':
            params = {
                'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *space['min_samples_leaf']),
                'alpha': trial.suggest_float('alpha', *space['alpha']),
                'subsample': trial.suggest_float('subsample', *space['subsample'])
            }
        
        # Create model with suggested parameters
        model = self._create_model(algorithm, params, stage=1)
        
        # Perform 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train the model
            if algorithm in ['XGBoost', 'LightGBM']:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif algorithm == 'CatBoost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            else:  # GradientBoosting
                model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate log loss (negative for maximization)
            score = -log_loss(y_val, y_pred_proba)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Stage 1: Optimize hyperparameters for each algorithm using TPE.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing best parameters for each algorithm
        """
        print("Stage 1: Hyperparameter Optimization")
        print("=" * 50)
        
        best_params = {}
        
        for algorithm in self.algorithms:
            print(f"\nOptimizing {algorithm}...")
            
            # Create Optuna study with TPE sampler
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            
            # Optimize
            study.optimize(
                lambda trial: self._objective_function(trial, algorithm, X, y),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            best_params[algorithm] = study.best_params
            print(f"Best parameters for {algorithm}: {study.best_params}")
            print(f"Best CV score: {study.best_value:.4f}")
        
        self.best_params = best_params
        return best_params
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[Any]]:
        """
        Stage 2: Train ensemble models using 5-fold cross-validation with optimal parameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing trained models for each algorithm
        """
        print("\nStage 2: Model Training with Optimal Parameters")
        print("=" * 50)
        
        trained_models = {}
        cv_results = {}
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for algorithm in self.algorithms:
            print(f"\nTraining {algorithm} with 5-fold CV...")
            
            models = []
            fold_scores = []
            fold_auc_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"  Fold {fold + 1}/5")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model with optimal parameters from stage 1
                model = self._create_model(algorithm, self.best_params[algorithm], stage=2)
                
                # Train the model
                if algorithm in ['XGBoost', 'LightGBM']:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif algorithm == 'CatBoost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                else:  # GradientBoosting
                    model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                accuracy = accuracy_score(y_val, y_pred)
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                models.append(model)
                fold_scores.append(accuracy)
                fold_auc_scores.append(auc_score)
                
                print(f"    Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
            
            trained_models[algorithm] = models
            cv_results[algorithm] = {
                'accuracy_mean': np.mean(fold_scores),
                'accuracy_std': np.std(fold_scores),
                'auc_mean': np.mean(fold_auc_scores),
                'auc_std': np.std(fold_auc_scores)
            }
            
            print(f"  Average CV Accuracy: {cv_results[algorithm]['accuracy_mean']:.4f} ± {cv_results[algorithm]['accuracy_std']:.4f}")
            print(f"  Average CV AUC: {cv_results[algorithm]['auc_mean']:.4f} ± {cv_results[algorithm]['auc_std']:.4f}")
        
        self.trained_models = trained_models
        self.cv_results = cv_results
        return trained_models
    
    def ensemble_predict(self, X: np.ndarray, algorithm: str) -> np.ndarray:
        """
        Generate ensemble predictions by averaging outputs from 5 classifiers.
        
        Args:
            X: Feature matrix
            algorithm: Algorithm name
            
        Returns:
            Averaged probability predictions
        """
        if algorithm not in self.trained_models:
            raise ValueError(f"Algorithm {algorithm} not trained yet.")
        
        models = self.trained_models[algorithm]
        predictions = []
        
        for model in models:
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Average predictions from all 5 models
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all algorithms on the test set using efficiency metrics.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dictionary containing evaluation results for each algorithm
        """
        print("\nTest Set Evaluation")
        print("=" * 50)
        
        results = {}
        
        for algorithm in self.algorithms:
            print(f"\nEvaluating {algorithm}...")
            
            # Get ensemble predictions
            ensemble_pred = self.ensemble_predict(X_test, algorithm)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, ensemble_pred_binary)
            auc_score = roc_auc_score(y_test, ensemble_pred)
            log_loss_score = log_loss(y_test, ensemble_pred)
            
            results[algorithm] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'log_loss': log_loss_score
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc_score:.4f}")
            print(f"  Log Loss: {log_loss_score:.4f}")
        
        return results
    
    def load_qcd_data(self, root_file_path: str = "data/processed/azura/azura-NewQCD.root") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load QCD data from ROOT file using the methodology from the research.
        
        Args:
            root_file_path: Path to the ROOT file
            
        Returns:
            Tuple of (X, y) - features and targets
        """
        print("Loading QCD data...")
        
        # Load ROOT data following the research methodology
        df_QCD = ROOT.RDataFrame("azura-NewQCD", root_file_path)
        
        # Get parameter definitions
        (variable_list, names_list, names_list_and_signal_trigger, 
         names_list_plot, range_min_list, range_max_list, 
         num_bins_list, y_min_list, y_max_list) = define_parameter("QCD")
        
        # Apply filters (assuming filter_all_meas is defined in the environment)
        try:
            df_ref = df_QCD.Filter("filter_all_meas", "")  # This might need adjustment based on actual filter
        except:
            # If filter is not available, use all data
            print("Warning: filter_all_meas not found, using all data")
            df_ref = df_QCD
        
        # Convert to pandas
        npy = df_ref.AsNumpy(columns=names_list_and_signal_trigger)
        df = pd.DataFrame(npy)
        
        # Prepare features and targets
        X = df.drop('Combo', axis=1).values
        y = df['Combo'].astype('int').values
        
        # Apply bootstrap sampling as in the original code
        index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sample = X[index_x]
        y_sample = y[index_x]
        
        print(f"Data loaded: {X_sample.shape[0]} samples, {X_sample.shape[1]} features")
        return X_sample, y_sample
    
    def run_complete_comparison(self, X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run the complete two-stage GBDT comparison pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            n_trials: Number of optimization trials
            
        Returns:
            Complete results dictionary
        """
        print("Starting Complete GBDT Comparison Pipeline")
        print("=" * 60)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Stage 1: Hyperparameter optimization
        best_params = self.optimize_hyperparameters(X_train, y_train, n_trials)
        
        # Stage 2: Train ensemble models
        trained_models = self.train_ensemble_models(X_train, y_train)
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Compile complete results
        complete_results = {
            'best_parameters': best_params,
            'cv_results': self.cv_results,
            'test_results': test_results,
            'data_info': {
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            }
        }
        
        return complete_results
    
    def save_models(self, filepath: str):
        """Save trained models to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'models': self.trained_models,
            'best_params': self.best_params,
            'cv_results': self.cv_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.trained_models = save_data['models']
        self.best_params = save_data['best_params']
        self.cv_results = save_data['cv_results']
        
        print(f"Models loaded from {filepath}")
    
    def print_comparison_summary(self, results: Dict[str, Any]):
        """
        Print a comprehensive summary of the GBDT comparison results.
        
        Args:
            results: Complete results dictionary
        """
        print("\n" + "=" * 60)
        print("GBDT COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"\nDataset Information:")
        print(f"  Training samples: {results['data_info']['train_samples']}")
        print(f"  Test samples: {results['data_info']['test_samples']}")
        print(f"  Features: {results['data_info']['features']}")
        
        print(f"\nCross-Validation Results:")
        for algorithm in self.algorithms:
            cv_res = results['cv_results'][algorithm]
            print(f"  {algorithm}:")
            print(f"    Accuracy: {cv_res['accuracy_mean']:.4f} ± {cv_res['accuracy_std']:.4f}")
            print(f"    AUC: {cv_res['auc_mean']:.4f} ± {cv_res['auc_std']:.4f}")
        
        print(f"\nTest Set Results:")
        best_algorithm = None
        best_auc = 0
        
        for algorithm in self.algorithms:
            test_res = results['test_results'][algorithm]
            print(f"  {algorithm}:")
            print(f"    Accuracy: {test_res['accuracy']:.4f}")
            print(f"    AUC: {test_res['auc']:.4f}")
            print(f"    Log Loss: {test_res['log_loss']:.4f}")
            
            if test_res['auc'] > best_auc:
                best_auc = test_res['auc']
                best_algorithm = algorithm
        
        print(f"\nBest performing algorithm: {best_algorithm} (AUC: {best_auc:.4f})")
        
        print(f"\nOptimal Hyperparameters:")
        for algorithm in self.algorithms:
            print(f"  {algorithm}: {results['best_parameters'][algorithm]}")


def main():
    """
    Main function to demonstrate the GBDT comparison framework.
    """
    # Initialize the comparison framework
    gbdt_comparison = GBDTComparison(random_state=42)
    
    try:
        # Load data using the methodology from the research
        X, y = gbdt_comparison.load_qcd_data()
        
        # Run complete comparison pipeline
        results = gbdt_comparison.run_complete_comparison(X, y, n_trials=50)
        
        # Print summary
        gbdt_comparison.print_comparison_summary(results)
        
        # Save models
        model_path = "models/gbdt_comparison_models.pkl"
        gbdt_comparison.save_models(model_path)
        
        # Save results
        results_path = "results/gbdt_comparison_results.pkl"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to {results_path}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure ROOT data files are available and properly configured.")


if __name__ == "__main__":
    main()
