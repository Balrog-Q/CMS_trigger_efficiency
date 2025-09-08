"""
Hyperparameter Optimization for GBDT Algorithms

This module implements the hyperparameter optimization process described in the research:
- TPE (Tree-structured Parzen Estimator) algorithm from Optuna
- Hyperparameter search spaces as defined in the research table
- Support for all four GBDT algorithms: XGBoost, LightGBM, CatBoost, GradientBoostingClassifier
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import time
import pickle
import os
from pathlib import Path

# Import the GBDT algorithms
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

class GBDTHyperparameterOptimizer:
    """
    Hyperparameter optimizer for GBDT algorithms using Optuna TPE.
    
    This class implements the exact optimization approach described in the research,
    with the hyperparameter ranges specified in the research table.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Define the hyperparameter search spaces from the research table
        self.search_spaces = self._define_search_spaces()
        
        # Training stage configurations from the research
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
        
        # Results storage
        self.studies = {}
        self.best_params = {}
        self.optimization_history = {}
    
    def _define_search_spaces(self) -> Dict[str, Dict[str, Tuple]]:
        """
        Define the hyperparameter search spaces for each GBDT algorithm based on the research table.
        
        From the research:
        - Max tree depth: All algorithms, Int, Default 6, Search [3:12]
        - Min child weight: XGBoost, Float, Default 1e-5, Search [1e-5:10]
        - Min data in leaf: All except XGBoost, Int, Default 1, Search [1:100]
        - L2 leaf regularization: All, Float, Default 1, Search [0.1:50]
        - Rows sampling rate: All, Float, Default 1.0, Search [0.5:1]
        
        Returns:
            Dictionary containing search spaces for each algorithm
        """
        return {
            'xgboost': {
                'max_depth': (3, 12),
                'min_child_weight': (1e-5, 10.0),
                'lambda': (0.1, 50.0),  # L2 regularization
                'subsample': (0.5, 1.0)  # Row sampling rate
            },
            'lightgbm': {
                'max_depth': (3, 12),
                'min_data_in_leaf': (1, 100),
                'lambda_l2': (0.1, 50.0),  # L2 regularization
                'subsample': (0.5, 1.0)  # Row sampling rate
            },
            'catboost': {
                'depth': (3, 12),  # max_depth equivalent
                'min_data_in_leaf': (1, 100),
                'l2_leaf_reg': (0.1, 50.0),  # L2 regularization
                'subsample': (0.5, 1.0)  # Row sampling rate
            },
            'gbm': {
                'max_depth': (3, 12),
                'min_samples_leaf': (1, 100),  # min_data_in_leaf equivalent
                'ccp_alpha': (0.1, 50.0),  # L2 regularization equivalent
                'subsample': (0.5, 1.0)  # Row sampling rate
            }
        }
    
    def _create_objective(self, algorithm: str, X: np.ndarray, y: np.ndarray, stage: int = 1) -> Callable:
        """
        Create an objective function for Optuna optimization.
        
        Args:
            algorithm: Algorithm name ('xgboost', 'lightgbm', 'catboost', 'gbm')
            X: Feature matrix
            y: Target vector
            stage: Training stage (1 or 2)
            
        Returns:
            Objective function for Optuna
        """
        config = self.stage1_config if stage == 1 else self.stage2_config
        search_space = self.search_spaces[algorithm]
        
        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for hyperparameter optimization.
            
            Args:
                trial: Optuna trial
                
            Returns:
                Negative log loss (to be minimized)
            """
            # Suggest hyperparameters based on the algorithm
            if algorithm == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'min_child_weight': trial.suggest_float('min_child_weight', *search_space['min_child_weight'], log=True),
                    'lambda': trial.suggest_float('lambda', *search_space['lambda']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'learning_rate': config['learning_rate'],
                    'n_estimators': config['max_iterations'],
                    'random_state': self.random_state,
                    'verbosity': 0,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif algorithm == 'lightgbm':
                params = {
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *search_space['min_data_in_leaf']),
                    'lambda_l2': trial.suggest_float('lambda_l2', *search_space['lambda_l2']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'learning_rate': config['learning_rate'],
                    'n_estimators': config['max_iterations'],
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif algorithm == 'catboost':
                params = {
                    'depth': trial.suggest_int('depth', *search_space['depth']),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *search_space['min_data_in_leaf']),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *search_space['l2_leaf_reg']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'learning_rate': config['learning_rate'],
                    'iterations': config['max_iterations'],
                    'random_state': self.random_state,
                    'verbose': False
                }
                model = cb.CatBoostClassifier(**params)
                
            elif algorithm == 'gbm':
                params = {
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_space['min_samples_leaf']),
                    'ccp_alpha': trial.suggest_float('ccp_alpha', *search_space['ccp_alpha']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'learning_rate': config['learning_rate'],
                    'n_estimators': config['max_iterations'],
                    'random_state': self.random_state
                }
                model = GradientBoostingClassifier(**params)
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Split the data for cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train the model
                if algorithm in ['xgboost', 'lightgbm']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=config['early_stopping'],
                        verbose=False
                    )
                elif algorithm == 'catboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=config['early_stopping'],
                        verbose=False
                    )
                else:  # gbm
                    model.fit(X_train, y_train)
                
                # Predict probabilities
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate log loss
                score = -log_loss(y_val, y_pred_proba)  # Negative because Optuna maximizes
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
    def optimize_hyperparameters(self, algorithm: str, X: np.ndarray, y: np.ndarray, 
                                n_trials: int = 100, stage: int = 1) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific GBDT algorithm.
        
        Args:
            algorithm: Algorithm name ('xgboost', 'lightgbm', 'catboost', 'gbm')
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            stage: Training stage (1 or 2)
            
        Returns:
            Dictionary containing best parameters and optimization history
        """
        algorithm = algorithm.lower()
        if algorithm not in self.search_spaces:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from: {list(self.search_spaces.keys())}")
        
        print(f"Optimizing hyperparameters for {algorithm} (Stage {stage})...")
        start_time = time.time()
        
        # Create Optuna study with TPE sampler
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        objective = self._create_objective(algorithm, X, y, stage)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store results
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to regular log loss (positive)
        
        # Record optimization history
        history = {
            'best_params': best_params,
            'best_log_loss': best_value,
            'n_trials': n_trials,
            'stage': stage,
            'trials': study.trials,
            'duration_seconds': time.time() - start_time
        }
        
        self.studies[algorithm] = study
        self.best_params[algorithm] = best_params
        self.optimization_history[algorithm] = history
        
        print(f"Optimization completed in {history['duration_seconds']:.2f} seconds")
        print(f"Best log loss: {best_value:.6f}")
        print(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_log_loss': best_value,
            'study': study,
            'history': history
        }
    
    def optimize_all_algorithms(self, X: np.ndarray, y: np.ndarray, n_trials: int = 100, 
                              stage: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all GBDT algorithms.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials per algorithm
            stage: Training stage (1 or 2)
            
        Returns:
            Dictionary containing results for each algorithm
        """
        algorithms = list(self.search_spaces.keys())
        results = {}
        
        print(f"Starting hyperparameter optimization for {len(algorithms)} algorithms (Stage {stage})")
        print(f"Number of trials per algorithm: {n_trials}")
        print(f"Total trials: {n_trials * len(algorithms)}")
        print("=" * 60)
        
        total_start_time = time.time()
        
        for algorithm in algorithms:
            print(f"\nOptimizing {algorithm}...")
            result = self.optimize_hyperparameters(algorithm, X, y, n_trials, stage)
            results[algorithm] = result
            print("-" * 40)
        
        total_time = time.time() - total_start_time
        print("\nAll optimizations completed!")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Print summary of results
        print("\nSummary of best parameters:")
        for algorithm, result in results.items():
            print(f"{algorithm}: {result['best_log_loss']:.6f} - {result['best_params']}")
        
        return results
    
    def visualize_optimization_results(self, algorithm: str = None, save_dir: str = None) -> None:
        """
        Visualize the optimization results for one or all algorithms.
        
        Args:
            algorithm: Algorithm name (if None, visualize all)
            save_dir: Directory to save the plots (if None, don't save)
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        algorithms = [algorithm] if algorithm else list(self.studies.keys())
        
        for alg in algorithms:
            if alg not in self.studies:
                print(f"No optimization results found for {alg}")
                continue
            
            study = self.studies[alg]
            best_params = self.best_params[alg]
            
            # Optimization history
            fig_history = plot_optimization_history(study)
            fig_history.update_layout(title=f"{alg} Optimization History")
            
            # Parameter importances
            fig_importance = plot_param_importances(study)
            fig_importance.update_layout(title=f"{alg} Parameter Importances")
            
            # Slice plots for important parameters
            param_names = list(best_params.keys())
            figs_slice = []
            
            for param in param_names:
                fig = plot_slice(study, params=[param])
                fig.update_layout(title=f"{alg} Parameter Slice: {param}")
                figs_slice.append(fig)
            
            # Display the plots
            fig_history.show()
            fig_importance.show()
            for fig in figs_slice:
                fig.show()
            
            # Save the plots if directory is provided
            if save_dir:
                fig_history.write_image(f"{save_dir}/{alg}_history.png")
                fig_importance.write_image(f"{save_dir}/{alg}_importance.png")
                for i, param in enumerate(param_names):
                    figs_slice[i].write_image(f"{save_dir}/{alg}_slice_{param}.png")
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a file.
        
        Args:
            filepath: Path to save the results
        """
        results = {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Optimization results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load optimization results from a file.
        
        Args:
            filepath: Path to load the results from
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.best_params = results['best_params']
        self.optimization_history = results['optimization_history']
        
        print(f"Optimization results loaded from {filepath}")


def run_hyperparameter_optimization(X: np.ndarray, y: np.ndarray, n_trials: int = 50, 
                                  save_path: str = 'results/hyperparameters.pkl', 
                                  visualize: bool = True) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for all GBDT algorithms and save results.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of optimization trials per algorithm
        save_path: Path to save the results
        visualize: Whether to visualize the results
        
    Returns:
        Dictionary containing the best parameters for each algorithm
    """
    # Create optimizer
    optimizer = GBDTHyperparameterOptimizer(random_state=42)
    
    # Stage 1: Parameter tuning with cross-validation
    results = optimizer.optimize_all_algorithms(X, y, n_trials=n_trials, stage=1)
    
    # Save results
    if save_path:
        optimizer.save_results(save_path)
    
    # Visualize results
    if visualize:
        viz_dir = os.path.join(os.path.dirname(save_path), 'visualizations')
        optimizer.visualize_optimization_results(save_dir=viz_dir)
    
    # Return best parameters for all algorithms
    return optimizer.best_params


if __name__ == "__main__":
    print("GBDT Hyperparameter Optimization Tool")
    print("=" * 50)
    print("\nTo use this tool, import the functions in your script:")
    print("from hyperparameter_optimization import run_hyperparameter_optimization")
    
    # Example usage
    print("\nExample usage:")
    print("best_params = run_hyperparameter_optimization(X, y, n_trials=50)")
    print("print(best_params['xgboost'])  # Get best parameters for XGBoost")
