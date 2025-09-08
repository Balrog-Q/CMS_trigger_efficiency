#!/usr/bin/env python3
"""
XGBoost Scale Factor Predictor for CMS Trigger Efficiency
========================================================

This module implements an advanced XGBoost-based approach to predict 
data/MC scale factors for CMS trigger efficiency using all 14 kinematic 
variables from the azura dataset with optimized hyperparameters.

Features:
- Loads data from ROOT files using your existing structure
- Uses reference trigger (HLT_AK8PFJet260) for tag-and-probe methodology
- Implements comprehensive hyperparameter optimization
- All 14 variables: HighestPt, HT, MET_pt, mHH, HighestMass, SecondHighestPt,
  SecondHighestMass, FatHT, MET_FatJet, mHHwithMET, HighestEta, SecondHighestEta,
  DeltaEta, DeltaPhi
- Advanced visualization and evaluation metrics

Author: Kha Tran
Date: 2025-01-06
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

# ROOT imports for HEP analysis
try:
    import ROOT
    ROOT_AVAILABLE = True
    print("ROOT available - using native ROOT data loading")
except ImportError:
    ROOT_AVAILABLE = False
    print("Warning: ROOT not available. Limited functionality.")

# Machine learning imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from scipy.optimize import minimize
import optuna

# Plotting setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class XGBoostScaleFactorPredictor:
    """
    Advanced XGBoost-based scale factor predictor for CMS trigger efficiency.
    """
    
    def __init__(self, output_dir: str = "xgboost_scale_factor_results", 
                 version: str = "v1"):
        """
        Initialize the XGBoost scale factor predictor.
        
        Args:
            output_dir: Directory to save results and plots
            version: Version identifier for this analysis
        """
        self.version = version
        self.output_dir = Path(output_dir) / f"{version}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all 14 variables used in the analysis
        self.feature_variables = [
            "HighestPt", "HT", "MET_pt", "mHH", "HighestMass", 
            "SecondHighestPt", "SecondHighestMass", "FatHT", 
            "MET_FatJet", "mHHwithMET", "HighestEta", 
            "SecondHighestEta", "DeltaEta", "DeltaPhi"
        ]
        
        # Trigger columns
        self.signal_trigger = "Combo"
        self.reference_trigger = "HLT_AK8PFJet260"
        self.all_columns = self.feature_variables + [self.signal_trigger, self.reference_trigger]
        
        # Storage for data and results
        self.data_samples = {}
        self.efficiency_results = {}
        self.scale_factor_models = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        
        print(f"XGBoost Scale Factor Predictor initialized")
        print(f"Version: {self.version}")
        print(f"Output directory: {self.output_dir}")
        print(f"Features: {len(self.feature_variables)} variables")

    def load_azura_data(self, data_dir: str = "data/processed/azura") -> bool:
        """
        Load all azura data samples from ROOT files.
        
        Args:
            data_dir: Path to the azura data directory
            
        Returns:
            bool: Success status
        """
        print(f"Loading azura data from {data_dir}")
        
        if not ROOT_AVAILABLE:
            print("ROOT not available. Generating demo data.")
            self._generate_demo_data()
            return True
        
        # Define sample files
        samples = {
            "QCD": "azura-NewQCD.root",
            "ggF": "azura-NewggF.root", 
            "VBF": "azura-NewVBF.root",
            "DATA": "azura-NewDATA.root"
        }
        
        data_path = Path(data_dir)
        loaded_samples = {}
        
        for sample_name, filename in samples.items():
            file_path = data_path / filename
            
            if not file_path.exists():
                print(f"Warning: File {file_path} not found")
                continue
                
            try:
                print(f"  Loading {sample_name} from {filename}")
                
                # Create RDataFrame
                tree_name = filename.replace(".root", "")
                df = ROOT.RDataFrame(tree_name, str(file_path))
                
                # Check if required columns exist
                available_columns = [col for col in self.all_columns 
                                   if df.HasColumn(col)]
                
                if len(available_columns) < len(self.feature_variables):
                    missing = set(self.all_columns) - set(available_columns)
                    print(f"    Warning: Missing columns in {sample_name}: {missing}")
                
                # Convert to pandas DataFrame
                data_dict = {}
                for col in available_columns:
                    data_dict[col] = df.AsNumpy([col])[col]
                
                sample_df = pd.DataFrame(data_dict)
                
                # Basic data cleaning
                sample_df = self._clean_data(sample_df, sample_name)
                
                loaded_samples[sample_name] = sample_df
                print(f"    Loaded {len(sample_df)} events")
                
            except Exception as e:
                print(f"Error loading {sample_name}: {e}")
                continue
        
        if not loaded_samples:
            print("No data loaded successfully. Generating demo data.")
            self._generate_demo_data()
            return True
        
        self.data_samples = loaded_samples
        
        # Print data summary
        print("\nData loading summary:")
        for name, df in self.data_samples.items():
            print(f"  {name}: {len(df)} events, {len(df.columns)} variables")
        
        return True

    def _clean_data(self, df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
        """
        Clean and validate the loaded data.
        
        Args:
            df: Input DataFrame
            sample_name: Name of the sample for logging
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        initial_events = len(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Apply reasonable kinematic cuts
        cuts = {
            "HighestPt": (0, 5000),
            "HT": (0, 10000), 
            "MET_pt": (0, 2000),
            "mHH": (0, 5000),
            "HighestMass": (0, 500),
            "HighestEta": (-5, 5),
            "SecondHighestEta": (-5, 5)
        }
        
        for var, (min_val, max_val) in cuts.items():
            if var in df.columns:
                initial_len = len(df)
                df = df[(df[var] >= min_val) & (df[var] <= max_val)]
                if len(df) < initial_len:
                    removed = initial_len - len(df)
                    print(f"    Removed {removed} events with {var} outside [{min_val}, {max_val}]")
        
        # Ensure trigger columns are binary
        for trigger in [self.signal_trigger, self.reference_trigger]:
            if trigger in df.columns:
                df[trigger] = df[trigger].astype(int)
                df[trigger] = df[trigger].clip(0, 1)
        
        final_events = len(df)
        if initial_events != final_events:
            print(f"    Data cleaning: {initial_events} -> {final_events} events")
        
        return df

    def _generate_demo_data(self, n_events: int = 10000):
        """
        Generate realistic demo data for testing when ROOT is not available.
        
        Args:
            n_events: Number of events to generate per sample
        """
        print(f"Generating {n_events} demo events per sample")
        
        np.random.seed(42)
        
        # Generate base kinematic variables with realistic correlations
        highest_pt = np.random.lognormal(5.8, 0.6, n_events).clip(200, 2000)
        ht = highest_pt * np.random.uniform(1.5, 3.0, n_events) + np.random.normal(0, 100, n_events)
        ht = ht.clip(400, 8000)
        
        met_pt = np.random.exponential(50, n_events).clip(0, 500)
        mhh = np.random.normal(1200, 400, n_events).clip(400, 3000)
        highest_mass = np.random.lognormal(3.8, 0.5, n_events).clip(20, 300)
        
        # Secondary variables
        second_highest_pt = highest_pt * np.random.uniform(0.4, 0.9, n_events)
        second_highest_mass = highest_mass * np.random.uniform(0.6, 1.4, n_events)
        fat_ht = highest_pt + second_highest_pt + np.random.normal(100, 50, n_events)
        met_fatjet = np.sqrt(met_pt**2 + (0.1 * fat_ht)**2)
        mhh_with_met = mhh + np.random.normal(0, 150, n_events)
        
        # Angular variables
        highest_eta = np.random.normal(0, 1.8, n_events).clip(-4.5, 4.5)
        second_highest_eta = np.random.normal(0, 1.8, n_events).clip(-4.5, 4.5)
        delta_eta = np.abs(highest_eta - second_highest_eta)
        delta_phi = np.random.uniform(-np.pi, np.pi, n_events)
        
        # Base data structure
        base_data = {
            'HighestPt': highest_pt,
            'HT': ht,
            'MET_pt': met_pt,
            'mHH': mhh,
            'HighestMass': highest_mass,
            'SecondHighestPt': second_highest_pt,
            'SecondHighestMass': second_highest_mass,
            'FatHT': fat_ht,
            'MET_FatJet': met_fatjet,
            'mHHwithMET': mhh_with_met,
            'HighestEta': highest_eta,
            'SecondHighestEta': second_highest_eta,
            'DeltaEta': delta_eta,
            'DeltaPhi': delta_phi,
            'HLT_AK8PFJet260': np.random.binomial(1, 0.95, n_events)  # Reference trigger
        }
        
        # Generate different samples with varying trigger efficiencies
        samples = {}
        
        # QCD (background) - lower efficiency
        qcd_eff_prob = self._calculate_efficiency_prob(base_data, efficiency_type="qcd")
        samples["QCD"] = {**base_data, 'Combo': np.random.binomial(1, qcd_eff_prob)}
        
        # ggF signal - higher efficiency 
        ggf_eff_prob = self._calculate_efficiency_prob(base_data, efficiency_type="signal")
        samples["ggF"] = {**base_data, 'Combo': np.random.binomial(1, ggf_eff_prob)}
        
        # VBF signal - similar to ggF
        vbf_eff_prob = self._calculate_efficiency_prob(base_data, efficiency_type="signal") * 1.02
        samples["VBF"] = {**base_data, 'Combo': np.random.binomial(1, vbf_eff_prob.clip(0, 1))}
        
        # DATA - slightly different from MC (creates scale factors != 1)
        data_eff_prob = self._calculate_efficiency_prob(base_data, efficiency_type="data")
        samples["DATA"] = {**base_data, 'Combo': np.random.binomial(1, data_eff_prob)}
        
        # Convert to DataFrames
        self.data_samples = {name: pd.DataFrame(data) for name, data in samples.items()}
        
        print("Demo data generated successfully")

    def _calculate_efficiency_prob(self, data: dict, efficiency_type: str) -> np.ndarray:
        """
        Calculate trigger efficiency probabilities for different sample types.
        
        Args:
            data: Dictionary containing kinematic variables
            efficiency_type: Type of efficiency curve ("qcd", "signal", "data")
            
        Returns:
            np.ndarray: Efficiency probabilities
        """
        highest_pt = data['HighestPt']
        ht = data['HT']
        met_pt = data['MET_pt']
        
        if efficiency_type == "qcd":
            # QCD background - lower efficiency
            eff = 1 / (1 + np.exp(-(highest_pt - 350) / 80))
            eff *= 1 / (1 + np.exp(-(ht - 600) / 200))
            eff *= 0.8  # Overall lower efficiency
            
        elif efficiency_type == "signal":
            # Signal samples - higher efficiency
            eff = 1 / (1 + np.exp(-(highest_pt - 300) / 60))
            eff *= 1 / (1 + np.exp(-(ht - 500) / 150))
            eff *= 0.95  # Higher overall efficiency
            
        elif efficiency_type == "data":
            # Data - slightly different turn-on curves
            eff = 1 / (1 + np.exp(-(highest_pt - 320) / 65))
            eff *= 1 / (1 + np.exp(-(ht - 520) / 160))
            eff *= 0.92  # Creates scale factors
            
        else:
            eff = np.full_like(highest_pt, 0.9)
        
        # Add MET dependence
        eff *= (1 - 0.1 * np.exp(-met_pt / 50))
        
        return eff.clip(0, 1)

    def calculate_binned_efficiencies(self, sample_name: str, variable: str, 
                                    n_bins: int = 15,
                                    bins: Optional[np.ndarray] = None) -> dict:
        """
        Calculate trigger efficiency in bins of a kinematic variable.
        
        Args:
            sample_name: Name of the sample
            variable: Variable to bin on
            n_bins: Number of bins (ignored if bins is provided)
            bins: Optional explicit bin edges to use
            
        Returns:
            dict: Efficiency results
        """
        if sample_name not in self.data_samples:
            raise ValueError(f"Sample {sample_name} not loaded")
        
        df = self.data_samples[sample_name]
        
        if variable not in df.columns:
            raise ValueError(f"Variable {variable} not found in {sample_name}")
        
        # Define bins
        if bins is None:
            var_min, var_max = df[variable].quantile([0.01, 0.99])
            bins = np.linspace(var_min, var_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        efficiencies = []
        uncertainties = []
        
        for i in range(len(bins) - 1):
            mask = (df[variable] >= bins[i]) & (df[variable] < bins[i + 1])
            
            if sample_name == "DATA":
                # Use tag-and-probe for data
                denominator = df[mask & (df[self.reference_trigger] == 1)]
                numerator = denominator[denominator[self.signal_trigger] == 1]
            else:
                # Direct efficiency for MC
                denominator = df[mask]
                numerator = denominator[denominator[self.signal_trigger] == 1]
            
            if len(denominator) == 0:
                efficiencies.append(0.0)
                uncertainties.append(0.0)
                continue
            
            eff = len(numerator) / len(denominator)
            # Clopper-Pearson uncertainty approximation
            unc = np.sqrt(eff * (1 - eff) / len(denominator))
            
            efficiencies.append(eff)
            uncertainties.append(unc)
        
        return {
            'variable': variable,
            'bins': bins,
            'bin_centers': bin_centers,
            'efficiencies': np.array(efficiencies),
            'uncertainties': np.array(uncertainties),
            'sample': sample_name
        }

    def calculate_scale_factors(self, mc_sample: str = "QCD", 
                              variables: Optional[List[str]] = None,
                              n_bins: int = 15) -> dict:
        """
        Calculate scale factors (Data/MC efficiency ratios) for multiple variables.
        
        Args:
            mc_sample: MC sample to compare against DATA
            variables: Variables to calculate scale factors for
            n_bins: Number of bins for efficiency calculation
            
        Returns:
            dict: Scale factor results
        """
        if "DATA" not in self.data_samples:
            raise ValueError("DATA sample not loaded")
        
        if mc_sample not in self.data_samples:
            raise ValueError(f"MC sample {mc_sample} not loaded")
        
        if variables is None:
            variables = ['HighestPt', 'HT', 'MET_pt', 'mHH']
        
        print(f"Calculating scale factors: DATA/{mc_sample}")
        
        scale_factor_results = {}
        
        for var in variables:
            print(f"  Processing {var}...")
            
            # Define common bins using combined DATA and MC ranges (quantiles)
            data_series = self.data_samples['DATA'][var]
            mc_series = self.data_samples[mc_sample][var]
            var_min = min(data_series.quantile(0.01), mc_series.quantile(0.01))
            var_max = max(data_series.quantile(0.99), mc_series.quantile(0.99))
            bins = np.linspace(var_min, var_max, n_bins + 1)
            
            # Calculate efficiencies with shared bins
            data_eff = self.calculate_binned_efficiencies("DATA", var, n_bins=n_bins, bins=bins)
            mc_eff = self.calculate_binned_efficiencies(mc_sample, var, n_bins=n_bins, bins=bins)
            
            # Calculate scale factors
            data_values = data_eff['efficiencies']
            mc_values = mc_eff['efficiencies']
            data_unc = data_eff['uncertainties']
            mc_unc = mc_eff['uncertainties']
            
            # Avoid division by zero
            scale_factors = np.divide(data_values, mc_values, 
                                    out=np.ones_like(data_values), 
                                    where=mc_values > 0)
            
            # Propagate uncertainties
            sf_uncertainties = scale_factors * np.sqrt(
                np.divide(data_unc**2, np.maximum(data_values, 1e-12)**2, out=np.zeros_like(data_unc), where=data_values > 0) +
                np.divide(mc_unc**2, np.maximum(mc_values, 1e-12)**2, out=np.zeros_like(mc_unc), where=mc_values > 0)
            )
            
            scale_factor_results[var] = {
                'bins': bins,
                'bin_centers': data_eff['bin_centers'],
                'data_efficiency': data_values,
                'mc_efficiency': mc_values,
                'data_uncertainty': data_unc,
                'mc_uncertainty': mc_unc,
                'scale_factors': scale_factors,
                'sf_uncertainties': sf_uncertainties,
                'mc_sample': mc_sample
            }
        
        self.efficiency_results = scale_factor_results
        return scale_factor_results

    def optimize_xgboost_parameters(self, target_variable: str = 'HighestPt',
                                  optimization_method: str = 'optuna',
                                  n_trials: int = 100) -> dict:
        """
        Optimize XGBoost hyperparameters using various optimization methods.
        
        Args:
            target_variable: Variable to optimize scale factor prediction for
            optimization_method: 'optuna', 'grid', or 'random'
            n_trials: Number of optimization trials
            
        Returns:
            dict: Best parameters and optimization results
        """
        print(f"Optimizing XGBoost parameters for {target_variable}")
        print(f"Method: {optimization_method}, Trials: {n_trials}")
        
        if target_variable not in self.efficiency_results:
            raise ValueError(f"Scale factors for {target_variable} not calculated. Run calculate_scale_factors first.")
        
        # Prepare training data
        X, y = self._prepare_training_data(target_variable)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if optimization_method == 'optuna':
            best_params = self._optimize_with_optuna(X_train_scaled, y_train, 
                                                   X_val_scaled, y_val, n_trials)
        elif optimization_method == 'grid':
            best_params = self._optimize_with_grid_search(X_train_scaled, y_train)
        elif optimization_method == 'random':
            best_params = self._optimize_with_random_search(X_train_scaled, y_train, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        results = {
            'best_params': best_params,
            'best_model': best_model,
            'validation_mse': mse,
            'validation_r2': r2,
            'validation_mae': mae,
            'target_variable': target_variable,
            'optimization_method': optimization_method
        }
        
        self.best_params[target_variable] = results
        self.scale_factor_models[target_variable] = best_model
        
        print(f"Optimization complete!")
        print(f"  Best MSE: {mse:.6f}")
        print(f"  Best R²: {r2:.4f}")
        print(f"  Best MAE: {mae:.6f}")
        
        return results

    def _prepare_training_data(self, target_variable: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for XGBoost scale factor prediction.
        
        Args:
            target_variable: Variable to create training data for
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target scale factors
        """
        sf_results = self.efficiency_results[target_variable]
        bins = sf_results['bins']
        bin_centers = sf_results['bin_centers']
        scale_factors = sf_results['scale_factors']
        
        X_train = []
        y_train = []
        
        # Use all available MC samples for training
        mc_samples = [name for name in self.data_samples.keys() if name != "DATA"]
        
        for sample_name in mc_samples:
            df = self.data_samples[sample_name]
            
            # For each bin, collect events and assign the bin's scale factor
            for i in range(len(bins) - 1):
                bin_start, bin_end = bins[i], bins[i + 1]
                sf_value = scale_factors[i]
                mask = (df[target_variable] >= bin_start) & (df[target_variable] < bin_end)
                
                if mask.sum() > 0:
                    # Sample events to avoid overfitting
                    selected_events = df[mask].sample(n=min(100, mask.sum()), random_state=42)
                    
                    X_train.extend(selected_events[self.feature_variables].values)
                    y_train.extend([sf_value] * len(selected_events))
        
        return np.array(X_train), np.array(y_train)

    def _optimize_with_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray, 
                             n_trials: int) -> dict:
        """Optimize using Optuna framework."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            # Use callback-based early stopping for broad version compatibility
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True, verbose=False)],
                )
            except TypeError:
                # Fallback: no early stopping if callbacks unsupported
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params

    def _optimize_with_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Optimize using GridSearchCV."""
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, 
                                  scoring='neg_mean_squared_error', 
                                  n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_

    def _optimize_with_random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                                   n_trials: int) -> dict:
        """Optimize using RandomizedSearchCV."""
        
        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(1e-8, 10),
            'reg_lambda': uniform(1e-8, 10),
            'min_child_weight': randint(1, 10)
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(xgb_model, param_distributions, 
                                         n_iter=n_trials, cv=3,
                                         scoring='neg_mean_squared_error',
                                         n_jobs=-1, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)
        
        return random_search.best_params_

    def predict_scale_factors_ml(self, sample_data: pd.DataFrame, 
                               target_variable: str = 'HighestPt') -> np.ndarray:
        """
        Predict scale factors for new data using trained XGBoost model.
        
        Args:
            sample_data: Input data to predict scale factors for
            target_variable: Variable model to use for prediction
            
        Returns:
            np.ndarray: Predicted scale factors
        """
        if target_variable not in self.scale_factor_models:
            raise ValueError(f"No trained model for {target_variable}")
        
        model = self.scale_factor_models[target_variable]
        
        # Prepare features
        X = sample_data[self.feature_variables].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predicted_sfs = model.predict(X_scaled) 
        
        return predicted_sfs

    def apply_scale_factors_to_mc(self, mc_sample_name: str, 
                                target_variable: str = 'HighestPt',
                                method: str = 'ml') -> np.ndarray:
        """
        Apply scale factor corrections to MC sample weights.
        
        Args:
            mc_sample_name: Name of MC sample to correct
            target_variable: Variable to use for scale factor lookup
            method: 'ml' for XGBoost prediction, 'interpolation' for binned SFs
            
        Returns:
            np.ndarray: Corrected event weights
        """
        if mc_sample_name not in self.data_samples:
            raise ValueError(f"MC sample {mc_sample_name} not loaded")
        
        df = self.data_samples[mc_sample_name]
        
        if method == 'ml':
            if target_variable not in self.scale_factor_models:
                raise ValueError(f"No ML model trained for {target_variable}")
            
            scale_factors = self.predict_scale_factors_ml(df, target_variable)
            
        elif method == 'interpolation':
            if target_variable not in self.efficiency_results:
                raise ValueError(f"No binned scale factors for {target_variable}")
            
            sf_data = self.efficiency_results[target_variable]
            bin_centers = sf_data['bin_centers']
            sf_values = sf_data['scale_factors']
            
            # Interpolate scale factors
            scale_factors = np.interp(df[target_variable], bin_centers, sf_values)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Assume unit weights if not present
        original_weights = df.get('weight', np.ones(len(df)))
        corrected_weights = original_weights * scale_factors
        
        return corrected_weights

    def create_comprehensive_plots(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualization of results.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        print("Creating comprehensive plots...")
        
        # 1. Efficiency comparison plots
        self._plot_efficiency_comparisons(save_plots)
        
        # 2. Scale factor plots
        self._plot_scale_factors(save_plots)
        
        # 3. ML model performance
        self._plot_ml_performance(save_plots)
        
        # 4. Feature importance
        self._plot_feature_importance(save_plots)
        
        # 5. Correlation analysis
        self._plot_correlation_analysis(save_plots)
        
        print(f"Plots saved to: {self.output_dir}")

    def _plot_efficiency_comparisons(self, save_plots: bool):
        """Plot efficiency comparisons between data and MC."""
        
        for var in self.efficiency_results.keys():
            sf_data = self.efficiency_results[var]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            bin_centers = sf_data['bin_centers']
            
            # Top panel: Efficiencies
            ax1.errorbar(bin_centers, sf_data['data_efficiency'],
                        yerr=sf_data['data_uncertainty'],
                        fmt='ko', label='Data (Tag-and-Probe)', 
                        markersize=6, capsize=4, linewidth=2)
            
            ax1.errorbar(bin_centers, sf_data['mc_efficiency'],
                        yerr=sf_data['mc_uncertainty'],
                        fmt='ro', label=f'MC ({sf_data["mc_sample"]})', 
                        markersize=6, capsize=4, linewidth=2)
            
            # MC corrected by scale factors
            corrected_mc = sf_data['mc_efficiency'] * sf_data['scale_factors']
            ax1.plot(bin_centers, corrected_mc, 'b-', 
                    label='MC × Scale Factor', linewidth=3, alpha=0.8)
            
            ax1.set_ylabel('Trigger Efficiency', fontsize=14)
            ax1.set_title(f'CMS Trigger Efficiency vs {var}', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05)
            
            # Bottom panel: Scale factors
            ax2.errorbar(bin_centers, sf_data['scale_factors'],
                        yerr=sf_data['sf_uncertainties'],
                        fmt='go', markersize=6, capsize=4, linewidth=2)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax2.fill_between(bin_centers,
                           sf_data['scale_factors'] - sf_data['sf_uncertainties'],
                           sf_data['scale_factors'] + sf_data['sf_uncertainties'],
                           alpha=0.2, color='green')
            
            ax2.set_xlabel(f'{var}', fontsize=14)
            ax2.set_ylabel('Scale Factor\n(Data/MC)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0.7, 1.3)
            
            # CMS label
            ax1.text(0.02, 0.95, 'CMS', transform=ax1.transAxes,
                    fontsize=18, fontweight='bold')
            ax1.text(0.02, 0.88, 'Preliminary', transform=ax1.transAxes,
                    fontsize=14, style='italic')
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.output_dir / f"efficiency_comparison_{var}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def _plot_scale_factors(self, save_plots: bool):
        """Plot scale factor summary across variables."""
        
        if not self.efficiency_results:
            return
        
        variables = list(self.efficiency_results.keys())
        n_vars = len(variables)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(variables[:4]):  # Plot first 4 variables
            if i >= len(axes):
                break
                
            sf_data = self.efficiency_results[var]
            bin_centers = sf_data['bin_centers']
            scale_factors = sf_data['scale_factors']
            uncertainties = sf_data['sf_uncertainties']
            
            axes[i].errorbar(bin_centers, scale_factors, yerr=uncertainties,
                           fmt='o-', markersize=6, capsize=4, linewidth=2,
                           label=f'Scale Factors')
            axes[i].axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[i].fill_between(bin_centers, scale_factors - uncertainties,
                               scale_factors + uncertainties, alpha=0.2)
            
            axes[i].set_xlabel(var, fontsize=12)
            axes[i].set_ylabel('Scale Factor', fontsize=12)
            axes[i].set_title(f'{var} Scale Factors', fontsize=14)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0.8, 1.2)
            
            # Add mean scale factor text
            mean_sf = np.mean(scale_factors)
            std_sf = np.std(scale_factors)
            axes[i].text(0.05, 0.95, f'Mean: {mean_sf:.3f}±{std_sf:.3f}',
                        transform=axes[i].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.suptitle('Scale Factor Summary Across Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "scale_factor_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_ml_performance(self, save_plots: bool):
        """Plot ML model performance metrics."""
        
        if not self.best_params:
            return
        
        for var, results in self.best_params.items():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Performance metrics
            metrics = {
                'MSE': results['validation_mse'],
                'R²': results['validation_r2'], 
                'MAE': results['validation_mae']
            }
            
            ax1.bar(metrics.keys(), metrics.values(), 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
            ax1.set_title(f'XGBoost Performance - {var}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Metric Value')
            for i, (k, v) in enumerate(metrics.items()):
                ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
            
            # Feature importance
            model = results['best_model']
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            ax2.bar(range(len(self.feature_variables)), importance[indices])
            ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Importance')
            ax2.set_xticks(range(len(self.feature_variables)))
            ax2.set_xticklabels([self.feature_variables[i] for i in indices], rotation=45)
            
            # Hyperparameters visualization
            best_params = results['best_params']
            param_names = list(best_params.keys())[:8]  # Show top 8 parameters
            param_values = [best_params[p] for p in param_names]
            
            ax3.barh(param_names, param_values, color='gold')
            ax3.set_title('Best Hyperparameters', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Parameter Value')
            
            # Training summary
            ax4.text(0.1, 0.9, f"Optimization: {results['optimization_method']}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.8, f"Target Variable: {results['target_variable']}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f"Features: {len(self.feature_variables)}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f"Best MSE: {results['validation_mse']:.6f}", 
                    transform=ax4.transAxes, fontsize=12, fontweight='bold', color='red')
            ax4.text(0.1, 0.5, f"Best R²: {results['validation_r2']:.4f}", 
                    transform=ax4.transAxes, fontsize=12, fontweight='bold', color='blue')
            ax4.text(0.1, 0.4, f"Best MAE: {results['validation_mae']:.6f}", 
                    transform=ax4.transAxes, fontsize=12, fontweight='bold', color='green')
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Model Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.output_dir / f"ml_performance_{var}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def _plot_feature_importance(self, save_plots: bool):
        """Plot comprehensive feature importance analysis."""
        
        if not self.scale_factor_models:
            return
        
        # Collect feature importance from all models
        all_importances = {}
        
        for var, model in self.scale_factor_models.items():
            importance = model.feature_importances_
            all_importances[var] = importance
        
        # Create importance dataframe
        importance_df = pd.DataFrame(all_importances, index=self.feature_variables)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='viridis',
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Across Target Variables', fontsize=16, fontweight='bold')
        plt.xlabel('Target Variables', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "feature_importance_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Individual feature ranking
        mean_importance = importance_df.mean(axis=1)
        std_importance = importance_df.std(axis=1)
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(mean_importance)[::-1]
        
        plt.bar(range(len(self.feature_variables)), mean_importance[indices],
               yerr=std_importance[indices], capsize=5, 
               color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Average Feature Importance', fontsize=14)
        plt.title('Average Feature Importance Ranking', fontsize=16, fontweight='bold')
        plt.xticks(range(len(self.feature_variables)), 
                  [self.feature_variables[i] for i in indices], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "feature_importance_ranking.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_correlation_analysis(self, save_plots: bool):
        """Plot correlation analysis between variables."""
        
        if "QCD" not in self.data_samples:
            return
        
        df = self.data_samples["QCD"]
        correlation_matrix = df[self.feature_variables].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "feature_correlation_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_results(self, filename_prefix: str = "xgboost_scale_factors") -> None:
        """
        Save all results to files.
        
        Args:
            filename_prefix: Prefix for output files
        """
        print(f"Saving results to {self.output_dir}")
        
        # Save scale factor results
        sf_file = self.output_dir / f"{filename_prefix}_results.npz"
        save_data = {}
        for var, data in self.efficiency_results.items():
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    save_data[f"{var}_{key}"] = value
                elif isinstance(value, (int, float, str)):
                    save_data[f"{var}_{key}"] = value
        
        np.savez_compressed(sf_file, **save_data)
        
        # Save models
        for var, model in self.scale_factor_models.items():
            model_file = self.output_dir / f"{filename_prefix}_model_{var}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Save best parameters
        params_file = self.output_dir / f"{filename_prefix}_best_params.json"
        serializable_params = {}
        for var, results in self.best_params.items():
            serializable_params[var] = {
                'best_params': results['best_params'],
                'validation_mse': float(results['validation_mse']),
                'validation_r2': float(results['validation_r2']),
                'validation_mae': float(results['validation_mae']),
                'optimization_method': results['optimization_method']
            }
        
        with open(params_file, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        
        # Save summary report
        self._generate_summary_report(filename_prefix)
        
        print(f"Results saved successfully!")

    def _generate_summary_report(self, filename_prefix: str) -> None:
        """Generate a comprehensive summary report."""
        
        report_file = self.output_dir / f"{filename_prefix}_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"XGBoost Scale Factor Prediction Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Version: {self.version}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write(f"Data Summary:\n")
            f.write(f"-" * 20 + "\n")
            for name, df in self.data_samples.items():
                f.write(f"  {name}: {len(df):,} events\n")
            f.write(f"  Features: {len(self.feature_variables)}\n")
            f.write(f"  Variables: {', '.join(self.feature_variables)}\n\n")
            
            # Scale factor summary
            f.write(f"Scale Factor Results:\n")
            f.write(f"-" * 20 + "\n")
            for var, results in self.efficiency_results.items():
                sf_mean = np.mean(results['scale_factors'])
                sf_std = np.std(results['scale_factors'])
                f.write(f"  {var}: {sf_mean:.3f} ± {sf_std:.3f}\n")
            f.write(f"\n")
            
            # Model performance
            f.write(f"XGBoost Model Performance:\n")
            f.write(f"-" * 30 + "\n")
            for var, results in self.best_params.items():
                f.write(f"  {var}:\n")
                f.write(f"    MSE: {results['validation_mse']:.6f}\n")
                f.write(f"    R²:  {results['validation_r2']:.4f}\n")
                f.write(f"    MAE: {results['validation_mae']:.6f}\n")
                f.write(f"    Method: {results['optimization_method']}\n\n")
            
            # Best parameters summary
            f.write(f"Best Hyperparameters:\n")
            f.write(f"-" * 20 + "\n")
            for var, results in self.best_params.items():
                f.write(f"  {var}:\n")
                for param, value in results['best_params'].items():
                    f.write(f"    {param}: {value}\n")
                f.write(f"\n")

    def run_complete_analysis(self, data_dir: str = "data/processed/azura",
                            optimization_method: str = 'optuna',
                            n_trials: int = 50) -> None:
        """
        Run the complete XGBoost scale factor analysis workflow.
        
        Args:
            data_dir: Path to azura data directory
            optimization_method: Hyperparameter optimization method
            n_trials: Number of optimization trials
        """
        print("=" * 80)
        print("XGBoost Scale Factor Prediction - Complete Analysis")
        print("=" * 80)
        
        # Step 1: Load data
        print("\nStep 1: Loading azura data...")
        success = self.load_azura_data(data_dir)
        if not success:
            print("Failed to load data. Exiting.")
            return
        
        # Step 2: Calculate binned scale factors
        print("\nStep 2: Calculating scale factors...")
        variables_to_analyze = self.feature_variables  # Use all 14 variables
        self.calculate_scale_factors('QCD', variables_to_analyze)
        
        # Step 3: Optimize XGBoost parameters for each variable
        print("\nStep 3: Optimizing XGBoost models...")
        for var in variables_to_analyze:
            print(f"\n  Optimizing for {var}...")
            self.optimize_xgboost_parameters(var, optimization_method, n_trials)
        
        # Step 4: Apply corrections and evaluate
        print("\nStep 4: Applying scale factor corrections...")
        for var in variables_to_analyze:
            # ML-based corrections
            ml_corrections = self.apply_scale_factors_to_mc('QCD', var, 'ml')
            interp_corrections = self.apply_scale_factors_to_mc('QCD', var, 'interpolation')
            interp_corrections_ggF = self.apply_scale_factors_to_mc('ggF', var, 'interpolation')
            interp_corrections_VBF = self.apply_scale_factors_to_mc('VBF', var, 'interpolation')
            
            print(f"  {var}:")
            print(f"    ML correction factor: {np.mean(ml_corrections):.3f} ± {np.std(ml_corrections):.3f}")
            print(f"    Interpolation correction QCD: {np.mean(interp_corrections):.3f} ± {np.std(interp_corrections):.3f}")
            print(f"    Interpolation correction ggF: {np.mean(interp_corrections_ggF):.3f} ± {np.std(interp_corrections_ggF):.3f}")
            print(f"    Interpolation correction VBF: {np.mean(interp_corrections_VBF):.3f} ± {np.std(interp_corrections_VBF):.3f}")
        
        # Step 5: Create comprehensive plots
        print("\nStep 5: Creating visualizations...")
        self.create_comprehensive_plots(save_plots=True)
        
        # Step 6: Save results
        print("\nStep 6: Saving results...")
        self.save_results()
        
        # Step 7: Summary
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print(f"Results saved in: {self.output_dir}")
        print("=" * 80)
        
        # Print summary statistics
        print("\nSUMMARY:")
        print("-" * 40)
        for var, results in self.best_params.items():
            print(f"{var:>12}: R² = {results['validation_r2']:.3f}, MSE = {results['validation_mse']:.6f}")


def main():
    """
    Main function to run the XGBoost scale factor analysis.
    """
    print("CMS Trigger Efficiency - XGBoost Scale Factor Predictor")
    print("=" * 60)
    
    # Create predictor instance
    predictor = XGBoostScaleFactorPredictor(
        output_dir="xgboost_scale_factor_results",
        version="v2"
    )
    
    # Run complete analysis
    predictor.run_complete_analysis(
        data_dir="data/processed/azura",
        optimization_method='optuna',  # Can also use 'grid' or 'random'
        n_trials=25  # Adjust based on available time
    )
    
    print("\nAnalysis completed successfully!")
    print(f"Check results in: {predictor.output_dir}")


if __name__ == "__main__":
    main()
