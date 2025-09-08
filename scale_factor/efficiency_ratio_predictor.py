#!/usr/bin/env python3
"""
CMS Trigger Efficiency Ratio Predictor
=====================================

This module implements methods to predict the ratio of trigger efficiencies 
between real data and Monte Carlo simulation using both traditional 
tag-and-probe methods and machine learning approaches.

Author: Kha Tran
Date: 2025-08-06
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ROOT imports for HEP analysis
try:
    import ROOT
    from ROOT import TEfficiency, TLegend, TH1F, TCanvas
    ROOT_AVAILABLE = True
except ImportError:
    print("Warning: ROOT not available. Some functionality will be limited.")
    ROOT_AVAILABLE = False

# Machine learning imports
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import binned_statistic


class EfficiencyRatioPredictor:
    """
    Main class for predicting data/MC efficiency ratios using multiple approaches.
    """
    
    def __init__(self, output_dir="efficiency_ratio_results"):
        """
        Initialize the predictor with output directory.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store efficiencies and scale factors
        self.efficiencies_data = {}
        self.efficiencies_mc = {}
        self.scale_factors = {}
        self.ml_models = {}
        
        print(f"EfficiencyRatioPredictor initialized. Results will be saved to: {self.output_dir}")
    
    def load_sample_data(self, n_events=10000):
        """
        Generate sample data for testing purposes.
        In practice, this would load from ROOT files.
        
        Args:
            n_events (int): Number of events to generate
            
        Returns:
            dict: Dictionary containing MC and data samples
        """
        print(f"Generating {n_events} sample events for demonstration...")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate kinematic variables
        highest_pt = np.random.exponential(200, n_events) + 250
        ht = np.random.exponential(300, n_events) + 450
        met_pt = np.random.exponential(50, n_events) + 10
        mhh = np.random.normal(1000, 300, n_events)
        highest_mass = np.random.exponential(40, n_events) + 20
        
        # Generate trigger responses (MC has slightly different efficiency)
        # Data efficiency depends on kinematic variables
        eff_data_base = 1 / (1 + np.exp(-(highest_pt - 300) / 50))  # Sigmoid efficiency
        eff_mc_base = 1 / (1 + np.exp(-(highest_pt - 290) / 50))    # Slightly different for MC
        
        # Add some HT dependence
        eff_data_base *= 1 / (1 + np.exp(-(ht - 500) / 100))
        eff_mc_base *= 1 / (1 + np.exp(-(ht - 490) / 100))
        
        # Generate actual trigger decisions
        combo_data = np.random.binomial(1, eff_data_base)
        combo_mc = np.random.binomial(1, eff_mc_base)
        
        # Reference trigger (always fires for selected events)
        ref_trigger = np.ones(n_events)
        
        # Create DataFrames
        data_sample = pd.DataFrame({
            'HighestPt': highest_pt,
            'HT': ht,
            'MET_pt': met_pt,
            'mHH': mhh,
            'HighestMass': highest_mass,
            'Combo': combo_data,
            'HLT_AK8PFJet260': ref_trigger,
            'weight': np.ones(n_events)  # Event weights
        })
        
        mc_sample = pd.DataFrame({
            'HighestPt': highest_pt,
            'HT': ht,
            'MET_pt': met_pt,
            'mHH': mhh,
            'HighestMass': highest_mass,
            'Combo': combo_mc,
            'HLT_AK8PFJet260': ref_trigger,
            'weight': np.ones(n_events)
        })
        
        return {'data': data_sample, 'mc': mc_sample}
    
    def calculate_binned_efficiency(self, df, variable, bins, signal_trigger='Combo', 
                                  reference_trigger=None):
        """
        Calculate trigger efficiency in bins of a kinematic variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            variable (str): Variable to bin on
            bins (array): Bin edges
            signal_trigger (str): Signal trigger column name
            reference_trigger (str): Reference trigger for tag-and-probe
            
        Returns:
            tuple: (bin_centers, efficiencies, uncertainties)
        """
        bin_centers = (bins[:-1] + bins[1:]) / 2
        efficiencies = []
        uncertainties = []
        
        for i in range(len(bins) - 1):
            # Select events in this bin
            mask = (df[variable] >= bins[i]) & (df[variable] < bins[i + 1])
            
            if reference_trigger is not None:
                # Tag-and-probe method
                denominator = df[mask & (df[reference_trigger] == 1)]
                numerator = denominator[denominator[signal_trigger] == 1]
            else:
                # Direct method (for MC truth)
                denominator = df[mask]
                numerator = denominator[denominator[signal_trigger] == 1]
            
            if len(denominator) == 0:
                efficiencies.append(0)
                uncertainties.append(0)
                continue
            
            # Calculate efficiency
            eff = len(numerator) / len(denominator)
            # Binomial uncertainty
            unc = np.sqrt(eff * (1 - eff) / len(denominator))
            
            efficiencies.append(eff)
            uncertainties.append(unc)
        
        return bin_centers, np.array(efficiencies), np.array(uncertainties)
    
    def calculate_scale_factors(self, data_df, mc_df, variables, bins_dict):
        """
        Calculate scale factors for multiple variables.
        
        Args:
            data_df (pd.DataFrame): Data sample
            mc_df (pd.DataFrame): MC sample
            variables (list): List of variables to calculate SFs for
            bins_dict (dict): Dictionary of bin edges for each variable
            
        Returns:
            dict: Scale factors and uncertainties for each variable
        """
        print("Calculating scale factors...")
        
        results = {}
        
        for var in variables:
            if var not in bins_dict:
                print(f"Warning: No bins defined for variable {var}")
                continue
            
            bins = bins_dict[var]
            
            # Calculate efficiencies
            bin_centers_data, eff_data, unc_data = self.calculate_binned_efficiency(
                data_df, var, bins, reference_trigger='HLT_AK8PFJet260'
            )
            
            bin_centers_mc, eff_mc, unc_mc = self.calculate_binned_efficiency(
                mc_df, var, bins 
            )
            
            # Calculate scale factors
            scale_factors = np.divide(eff_data, eff_mc, 
                                    out=np.ones_like(eff_data), 
                                    where=eff_mc!=0)
            
            # Propagate uncertainties
            sf_uncertainties = scale_factors * np.sqrt(
                np.divide(unc_data**2, eff_data**2, out=np.zeros_like(unc_data), where=eff_data!=0) +
                np.divide(unc_mc**2, eff_mc**2, out=np.zeros_like(unc_mc), where=eff_mc!=0)
            )
            
            results[var] = {
                'bin_centers': bin_centers_data,
                'eff_data': eff_data,
                'eff_mc': eff_mc,
                'unc_data': unc_data,
                'unc_mc': unc_mc,
                'scale_factors': scale_factors,
                'sf_uncertainties': sf_uncertainties
            }
            
            # Store for later use
            self.scale_factors[var] = results[var]
        
        return results
    
    def train_ml_ratio_predictor(self, data_df, mc_df, features, target_var='HighestPt', 
                                bins=None):
        """
        Train ML model to predict data/MC efficiency ratios.
        
        Args:
            data_df (pd.DataFrame): Data sample
            mc_df (pd.DataFrame): MC sample  
            features (list): Feature columns for training
            target_var (str): Variable to calculate ratios for
            bins (array): Bins to calculate target ratios
            
        Returns:
            dict: Trained model and performance metrics
        """
        print("Training ML ratio predictor...")
        
        if bins is None:
            bins = np.linspace(data_df[target_var].min(), data_df[target_var].max(), 20)
        
        # Calculate true ratios in bins
        _, eff_data, _ = self.calculate_binned_efficiency(
            data_df, target_var, bins, reference_trigger='HLT_AK8PFJet260'
        )
        _, eff_mc, _ = self.calculate_binned_efficiency(
            mc_df, target_var, bins
        )
        
        true_ratios = np.divide(eff_data, eff_mc, out=np.ones_like(eff_data), where=eff_mc!=0)
        
        # Create training dataset
        # For each bin, sample events and assign the bin's ratio as target
        X_train = []
        y_train = []
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            # Sample from both data and MC in this bin
            data_mask = (data_df[target_var] >= bin_start) & (data_df[target_var] < bin_end)
            mc_mask = (mc_df[target_var] >= bin_start) & (mc_df[target_var] < bin_end)
            
            # Add data samples
            if data_mask.sum() > 0:
                data_features = data_df[data_mask][features].values
                X_train.extend(data_features)
                y_train.extend([true_ratios[i]] * len(data_features))
            
            # Add MC samples  
            if mc_mask.sum() > 0:
                mc_features = mc_df[mc_mask][features].values
                X_train.extend(mc_features)
                y_train.extend([true_ratios[i]] * len(mc_features))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        model.fit(X_tr, y_tr)
        
        # Evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        print(f"ML Ratio Predictor MSE: {mse:.4f}")
        
        # Store model
        self.ml_models[target_var] = {
            'model': model,
            'features': features,
            'mse': mse,
            'true_ratios': true_ratios,
            'bin_centers': bin_centers
        }
        
        return self.ml_models[target_var]
    
    def apply_scale_factors(self, mc_df, variable, method='interpolation'):
        """
        Apply scale factors to MC events.
        
        Args:
            mc_df (pd.DataFrame): MC events to correct
            variable (str): Variable to use for scale factor lookup
            method (str): Method to apply SFs ('interpolation' or 'ml')
            
        Returns:
            np.array: Corrected event weights
        """
        if method == 'interpolation' and variable in self.scale_factors:
            # Use binned scale factors with interpolation
            sf_data = self.scale_factors[variable]
            bin_centers = sf_data['bin_centers']
            scale_factors = sf_data['scale_factors']
            
            # Interpolate scale factors
            corrected_weights = np.interp(
                mc_df[variable], 
                bin_centers, 
                scale_factors
            ) * mc_df['weight']
            
        elif method == 'ml' and variable in self.ml_models:
            # Use ML prediction
            model_data = self.ml_models[variable]
            model = model_data['model']
            features = model_data['features']
            
            # Predict scale factors
            X = mc_df[features].values
            predicted_sfs = model.predict(X)
            corrected_weights = predicted_sfs * mc_df['weight']
            
        else:
            print(f"Warning: Method {method} not available for variable {variable}")
            corrected_weights = mc_df['weight'].values
        
        return corrected_weights
    
    def plot_efficiency_comparison(self, variable, save_plot=True):
        """
        Plot efficiency comparison between data and MC with scale factors.
        
        Args:
            variable (str): Variable to plot
            save_plot (bool): Whether to save the plot
        """
        if variable not in self.scale_factors:
            print(f"No scale factors calculated for variable {variable}")
            return
        
        sf_data = self.scale_factors[variable]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Top panel: Efficiencies
        bin_centers = sf_data['bin_centers']
        
        ax1.errorbar(bin_centers, sf_data['eff_data'], yerr=sf_data['unc_data'],
                    fmt='ko', label='Data', markersize=4)
        ax1.errorbar(bin_centers, sf_data['eff_mc'], yerr=sf_data['unc_mc'],
                    fmt='ro', label='MC', markersize=4)
        
        # Corrected MC (multiply by scale factors)
        eff_mc_corrected = sf_data['eff_mc'] * sf_data['scale_factors']
        ax1.plot(bin_centers, eff_mc_corrected, 'b-', label='MC × SF', linewidth=2)
        
        ax1.set_ylabel('Trigger Efficiency')
        ax1.set_title(f'Trigger Efficiency vs {variable}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Bottom panel: Scale factors
        ax2.errorbar(bin_centers, sf_data['scale_factors'], 
                    yerr=sf_data['sf_uncertainties'],
                    fmt='go', markersize=4)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel(variable)
        ax2.set_ylabel('Scale Factor\n(Data/MC)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.8, 1.2)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"efficiency_comparison_{variable}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()
    
    def plot_ml_performance(self, variable, save_plot=True):
        """
        Plot ML model performance for ratio prediction.
        
        Args:
            variable (str): Variable to plot
            save_plot (bool): Whether to save the plot
        """
        if variable not in self.ml_models:
            print(f"No ML model trained for variable {variable}")
            return
        
        model_data = self.ml_models[variable]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Left: True vs predicted ratios
        bin_centers = model_data['bin_centers']
        true_ratios = model_data['true_ratios']
        
        # Get predictions for bin centers (create dummy features)
        # This is simplified - in practice you'd use actual feature values
        ax1.plot(bin_centers, true_ratios, 'ro-', label='True Ratios', markersize=6)
        ax1.set_xlabel(variable)
        ax1.set_ylabel('Data/MC Ratio')
        ax1.set_title('ML Model: True Ratios vs Bin Centers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Feature importance
        if hasattr(model_data['model'], 'feature_importances_'):
            importances = model_data['model'].feature_importances_
            features = model_data['features']
            
            indices = np.argsort(importances)[::-1]
            ax2.bar(range(len(features)), importances[indices])
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Importance')
            ax2.set_title('ML Model Feature Importance')
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels([features[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"ml_performance_{variable}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()
    
    def save_scale_factors(self, filename=None):
        """
        Save scale factors to file for later use.
        
        Args:
            filename (str): Output filename (optional)
        """
        if filename is None:
            filename = self.output_dir / "scale_factors.npz"
        
        # Prepare data for saving
        save_data = {}
        for var, sf_data in self.scale_factors.items():
            for key, value in sf_data.items():
                save_data[f"{var}_{key}"] = value
        
        np.savez(filename, **save_data)
        print(f"Scale factors saved to: {filename}")
    
    def run_full_analysis(self, n_events=10000):
        """
        Run complete efficiency ratio analysis.
        
        Args:
            n_events (int): Number of events to generate for demo
        """
        print("="*60)
        print("Running Complete Efficiency Ratio Analysis")
        print("="*60)
        
        # 1. Load data
        samples = self.load_sample_data(n_events)
        data_df = samples['data']
        mc_df = samples['mc']
        
        print(f"Loaded {len(data_df)} data events and {len(mc_df)} MC events")
        
        # 2. Define variables and bins
        variables = ['HighestPt', 'HT', 'MET_pt', 'mHH']
        bins_dict = {
            'HighestPt': np.linspace(250, 800, 15),
            'HT': np.linspace(450, 1500, 15),
            'MET_pt': np.linspace(0, 200, 15),
            'mHH': np.linspace(400, 2000, 15)
        }
        
        # 3. Calculate scale factors
        sf_results = self.calculate_scale_factors(data_df, mc_df, variables, bins_dict)
        
        # 4. Train ML models
        features = ['HighestPt', 'HT', 'MET_pt', 'HighestMass']
        for var in ['HighestPt', 'HT']:
            self.train_ml_ratio_predictor(data_df, mc_df, features, var, bins_dict[var])
        
        # 5. Apply corrections
        print("\nApplying scale factor corrections...")
        original_weights = mc_df['weight'].sum()
        
        corrected_weights_interp = self.apply_scale_factors(mc_df, 'HighestPt', 'interpolation')
        corrected_weights_ml = self.apply_scale_factors(mc_df, 'HighestPt', 'ml')
        
        print(f"Original MC weight sum: {original_weights:.1f}")
        print(f"Corrected (interpolation): {corrected_weights_interp.sum():.1f}")
        print(f"Corrected (ML): {corrected_weights_ml.sum():.1f}")
        
        # 6. Generate plots
        print("\nGenerating plots...")
        for var in ['HighestPt', 'HT']:
            self.plot_efficiency_comparison(var)
            if var in self.ml_models:
                self.plot_ml_performance(var)
        
        # 7. Save results
        self.save_scale_factors()
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"Results saved in: {self.output_dir}")
        print("="*60)


def main():
    """
    Main function to run the efficiency ratio prediction demo.
    """
    print("CMS Trigger Efficiency Ratio Predictor")
    print("======================================")
    
    # Create predictor instance
    predictor = EfficiencyRatioPredictor()
    
    # Run full analysis
    predictor.run_full_analysis(n_events=10000)


if __name__ == "__main__":
    main()
