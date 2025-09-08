#!/usr/bin/env python3
"""
Integrated CMS Trigger Efficiency Ratio Predictor
=================================================

This module extends the existing gradient boosting workflow to predict 
data/MC efficiency ratios. It's designed to work with the current 
codebase and ROOT data structures.

Usage:
    from integrated_ratio_predictor_fixed import DataMCRatioAnalyzer
    
    analyzer = DataMCRatioAnalyzer()
    analyzer.load_data_from_root(df_data, df_mc)
    analyzer.calculate_efficiency_ratios()
    analyzer.plot_results()
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import from the existing library
sys.path.append('library')
try:
    from library.trigger_efficiency_ML import get_plot_directory, efficiency_plot
    PLOTTING_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: plotting_tools not available. Using fallback methods.")
    PLOTTING_TOOLS_AVAILABLE = False

# Machine learning imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import binned_statistic

class DataMCRatioAnalyzer:
    """
    Analyzer for data/MC efficiency ratios that integrates with the existing workflow.
    """
    
    def __init__(self, version="v5", run_name="ratio_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            version (str): Version identifier for output files
            run_name (str): Run name for this analysis
        """
        self.version = version
        self.run_name = run_name
        self.suffix = f"-{version}-{run_name}"
        
        # Create output directory using the existing system
        if PLOTTING_TOOLS_AVAILABLE:
            self.output_dir = get_plot_directory(self.suffix)
        else:
            self.output_dir = Path(f"result/{datetime.now().strftime('%d-%m-%Y')}{self.suffix}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.data_sample = None
        self.mc_sample = None
        self.efficiency_results = {}
        self.ratio_predictors = {}
        
        print(f"DataMCRatioAnalyzer initialized for {version} - {run_name}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data_from_root(self, df_data, df_mc, variables=None):
        """
        Load data from ROOT DataFrames (the existing format).
        
        Args:
            df_data: ROOT RDataFrame for real data
            df_mc: ROOT RDataFrame for Monte Carlo
            variables (list): Variables to extract (optional)
        """
        if variables is None:
            variables = ["HighestPt", "HT", "MET_pt", "mHH", "HighestMass", 
                        "SecondHighestPt", "SecondHighestMass", "FatHT", 
                        "MET_FatJet", "mHHwithMET", "HighestEta", 
                        "SecondHighestEta", "DeltaEta", "DeltaPhi", 
                        "Combo", "HLT_AK8PFJet260"]
        
        print("Converting ROOT DataFrames to pandas...")
        
        try:
            # Convert to pandas (this assumes you have ROOT available)
            data_dict = {}
            mc_dict = {}
            
            for var in variables:
                if df_data.HasColumn(var):
                    data_dict[var] = df_data.AsNumpy([var])[var]
                if df_mc.HasColumn(var):
                    mc_dict[var] = df_mc.AsNumpy([var])[var]
            
            self.data_sample = pd.DataFrame(data_dict)
            self.mc_sample = pd.DataFrame(mc_dict)
            
            print(f"Loaded {len(self.data_sample)} data events and {len(self.mc_sample)} MC events")
            
        except Exception as e:
            print(f"Error loading from ROOT: {e}")
            print("Generating demo data instead...")
            self.generate_demo_data()
    
    def generate_demo_data(self, n_events=5000):
        """
        Generate demo data that matches the data structure for testing.
        
        Args:
            n_events (int): Number of events to generate
        """
        print(f"Generating {n_events} demo events...")
        
        np.random.seed(42)
        
        # Generate realistic kinematic distributions
        highest_pt = np.random.lognormal(5.5, 0.5, n_events)  # Log-normal for pT
        ht = np.random.lognormal(6.0, 0.4, n_events) 
        met_pt = np.random.exponential(40, n_events)
        mhh = np.random.normal(1000, 400, n_events)
        highest_mass = np.random.lognormal(3.5, 0.6, n_events)
        
        # Additional variables from the analysis
        second_highest_pt = highest_pt * np.random.uniform(0.3, 0.8, n_events)
        second_highest_mass = highest_mass * np.random.uniform(0.5, 1.2, n_events)
        fat_ht = highest_pt + second_highest_pt + np.random.normal(0, 50, n_events)
        met_fatjet = met_pt + highest_pt + second_highest_pt
        mhh_with_met = mhh + np.random.normal(0, 100, n_events)
        
        # Eta and angular variables
        highest_eta = np.random.normal(0, 1.5, n_events)
        second_highest_eta = np.random.normal(0, 1.5, n_events)
        delta_eta = np.abs(highest_eta - second_highest_eta)
        delta_phi = np.random.uniform(-np.pi, np.pi, n_events)
        
        # Trigger efficiency models (data vs MC difference)
        # Data efficiency: depends on pT and HT
        eff_data_prob = 1 / (1 + np.exp(-(highest_pt - 280) / 60))  # Sigmoid
        eff_data_prob *= 1 / (1 + np.exp(-(ht - 520) / 150))      # HT dependence   
        
        # MC efficiency: slightly different thresholds (common in CMS)
        eff_mc_prob = 1 / (1 + np.exp(-(highest_pt - 270) / 55))   # Different turn-on
        eff_mc_prob *= 1 / (1 + np.exp(-(ht - 510) / 140))       # Different HT threshold
        
        # Generate trigger decisions
        combo_data = np.random.binomial(1, eff_data_prob)
        combo_mc = np.random.binomial(1, eff_mc_prob)
        
        # Reference trigger (high efficiency, used for tag-and-probe)
        ref_trigger = np.random.binomial(1, 0.95, n_events)  # 95% efficient
        
        # Create DataFrames matching the structure
        common_data = {
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
            'HLT_AK8PFJet260': ref_trigger
        }
        
        self.data_sample = pd.DataFrame({**common_data, 'Combo': combo_data})
        self.mc_sample = pd.DataFrame({**common_data, 'Combo': combo_mc})
        
        print(f"Generated {len(self.data_sample)} data and {len(self.mc_sample)} MC events")
    
    def calculate_efficiency_ratios(self, variables=None, n_bins=12):
        """
        Calculate efficiency ratios for specified variables.
        
        Args:
            variables (list): Variables to analyze
            n_bins (int): Number of bins for efficiency calculation
        """
        if variables is None:
            variables = ['HighestPt', 'HT', 'MET_pt', 'mHH']
        
        print("Calculating efficiency ratios...")
        
        for var in variables:
            if var not in self.data_sample.columns or var not in self.mc_sample.columns:
                print(f"Warning: Variable {var} not found in data")
                continue
            
            print(f"Processing {var}...")
            
            # Define bins based on data range
            var_min = min(self.data_sample[var].min(), self.mc_sample[var].min())
            var_max = max(self.data_sample[var].max(), self.mc_sample[var].max())
            bins = np.linspace(var_min, var_max, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate efficiencies
            data_eff, data_err = self._calculate_binned_efficiency(
                self.data_sample, var, bins, use_reference=True
            )
            mc_eff, mc_err = self._calculate_binned_efficiency(
                self.mc_sample, var, bins, use_reference=False
            )
            
            # Calculate ratios (scale factors)
            ratios = np.divide(data_eff, mc_eff, out=np.ones_like(data_eff), where=mc_eff>0)
            
            # Propagate uncertainties
            ratio_err = ratios * np.sqrt(
                (data_err / np.maximum(data_eff, 1e-6))**2 + 
                (mc_err / np.maximum(mc_eff, 1e-6))**2
            )
            
            # Store results
            self.efficiency_results[var] = {
                'bins': bins,
                'bin_centers': bin_centers,
                'data_efficiency': data_eff,
                'mc_efficiency': mc_eff,
                'data_error': data_err,
                'mc_error': mc_err,
                'ratio': ratios,
                'ratio_error': ratio_err
            }
    
    def _calculate_binned_efficiency(self, df, variable, bins, use_reference=True):
        """
        Calculate trigger efficiency in bins using tag-and-probe or direct method.
        
        Args:
            df (pd.DataFrame): Input data
            variable (str): Variable to bin on
            bins (array): Bin edges
            use_reference (bool): Use reference trigger for tag-and-probe
            
        Returns:
            tuple: (efficiencies, uncertainties)
        """
        efficiencies = []
        uncertainties = []
        
        for i in range(len(bins) - 1):
            mask = (df[variable] >= bins[i]) & (df[variable] < bins[i + 1])
            
            if use_reference:
                # Tag-and-probe: denominator passes reference trigger
                denominator = df[mask & (df['HLT_AK8PFJet260'] == 1)]
                numerator = denominator[denominator['Combo'] == 1]
            else:
                # Direct method for MC
                denominator = df[mask]
                numerator = denominator[denominator['Combo'] == 1]
            
            if len(denominator) == 0:
                efficiencies.append(0.0)
                uncertainties.append(0.0)
                continue
            
            eff = len(numerator) / len(denominator)
            # Clopper-Pearson uncertainty (approximately binomial)
            err = np.sqrt(eff * (1 - eff) / len(denominator)) if len(denominator) > 0 else 0
            
            efficiencies.append(eff)
            uncertainties.append(err)
        
        return np.array(efficiencies), np.array(uncertainties)
    
    def train_ratio_predictors(self, variables=None, features=None):
        """
        Train ML models to predict efficiency ratios.
        
        Args:
            variables (list): Target variables to predict ratios for
            features (list): Input features for prediction
        """
        if variables is None:
            variables = ['HighestPt', 'HT']
        
        if features is None:
            features = ['HighestPt', 'HT', 'MET_pt', 'HighestMass']
        
        print("Training ML ratio predictors...")
        
        for var in variables:
            if var not in self.efficiency_results:
                print(f"Warning: No efficiency results for {var}. Run calculate_efficiency_ratios first.")
                continue
            
            print(f"Training predictor for {var}...")
            
            # Get the calculated ratios as targets
            results = self.efficiency_results[var]
            bin_centers = results['bin_centers']
            true_ratios = results['ratio']
            
            # Create training dataset
            X_train = []
            y_train = []
            
            bins = results['bins']
            
            # For each bin, collect events and assign the bin's ratio
            for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                # Sample from data
                data_mask = (self.data_sample[var] >= bin_start) & (self.data_sample[var] < bin_end)
                if data_mask.sum() > 0:
                    X_train.extend(self.data_sample[data_mask][features].values)
                    y_train.extend([true_ratios[i]] * data_mask.sum())
                
                # Sample from MC
                mc_mask = (self.mc_sample[var] >= bin_start) & (self.mc_sample[var] < bin_end)
                if mc_mask.sum() > 0:
                    X_train.extend(self.mc_sample[mc_mask][features].values)
                    y_train.extend([true_ratios[i]] * mc_mask.sum())
            
            if len(X_train) == 0:
                print(f"No training data available for {var}")
                continue
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train gradient boosting regressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            model.fit(X_tr, y_tr)
            
            # Evaluate
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"  {var}: MSE={mse:.4f}, R²={r2:.3f}")
            
            # Store model
            self.ratio_predictors[var] = {
                'model': model,
                'features': features,
                'mse': mse,
                'r2': r2,
                'bin_centers': bin_centers,
                'true_ratios': true_ratios
            }
    
    def predict_ratio_for_event(self, event_features, variable='HighestPt'):
        """
        Predict efficiency ratio for a single event.
        
        Args:
            event_features (dict or array): Event features
            variable (str): Variable model to use
            
        Returns:
            float: Predicted efficiency ratio
        """
        if variable not in self.ratio_predictors:
            print(f"No trained predictor for {variable}")
            return 1.0
        
        model_data = self.ratio_predictors[variable]
        model = model_data['model']
        features = model_data['features']
        
        if isinstance(event_features, dict):
            X = np.array([[event_features[f] for f in features]])
        else:
            X = np.array([event_features])
        
        return model.predict(X)[0]
    
    def apply_corrections_to_mc(self, variable='HighestPt', method='interpolation'):
        """
        Apply efficiency ratio corrections to MC sample.
        
        Args:
            variable (str): Variable to use for corrections
            method (str): 'interpolation' or 'ml'
            
        Returns:
            np.array: Correction weights
        """
        if method == 'interpolation' and variable in self.efficiency_results:
            results = self.efficiency_results[variable]
            bin_centers = results['bin_centers']
            ratios = results['ratio']
            
            # Interpolate ratios for each MC event
            corrections = np.interp(
                self.mc_sample[variable], 
                bin_centers, 
                ratios
            )
            
        elif method == 'ml' and variable in self.ratio_predictors:
            model_data = self.ratio_predictors[variable]
            model = model_data['model']
            features = model_data['features']
            
            # Predict corrections using ML
            X = self.mc_sample[features].values
            corrections = model.predict(X)
            
        else:
            print(f"Method {method} not available for {variable}")
            corrections = np.ones(len(self.mc_sample))
        
        return corrections
    
    def plot_results(self, variables=None, save_plots=True):
        """
        Create comprehensive plots of efficiency ratios and ML performance.
        
        Args:
            variables (list): Variables to plot
            save_plots (bool): Whether to save plots
        """
        if variables is None:
            variables = list(self.efficiency_results.keys())
        
        print("Creating efficiency ratio plots...")
        
        for var in variables:
            if var not in self.efficiency_results:
                continue
            
            self._plot_efficiency_comparison(var, save_plots)
            
            if var in self.ratio_predictors:
                self._plot_ml_performance(var, save_plots)
    
    def _plot_efficiency_comparison(self, variable, save_plot=True):
        """
        Plot efficiency comparison with ratios.
        """
        results = self.efficiency_results[variable]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        bin_centers = results['bin_centers']
        
        # Top: Efficiencies
        ax1.errorbar(bin_centers, results['data_efficiency'], 
                    yerr=results['data_error'],
                    fmt='ko', label='Data (Gradient-Boosting)', markersize=5, capsize=3)
        ax1.errorbar(bin_centers, results['mc_efficiency'], 
                    yerr=results['mc_error'],
                    fmt='ro', label='MC (Truth)', markersize=5, capsize=3)
        
        # MC corrected by scale factors
        corrected_mc = results['mc_efficiency'] * results['ratio']
        ax1.plot(bin_centers, corrected_mc, 'b-', 
                label='MC × Scale Factor', linewidth=2, alpha=0.8)
        
        ax1.set_ylabel('Trigger Efficiency', fontsize=12)
        ax1.set_title(f'CMS Trigger Efficiency vs {variable}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Bottom: Scale factors
        ax2.errorbar(bin_centers, results['ratio'], 
                    yerr=results['ratio_error'],
                    fmt='go', markersize=5, capsize=3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.fill_between(bin_centers, 
                        results['ratio'] - results['ratio_error'],
                        results['ratio'] + results['ratio_error'],
                        alpha=0.2, color='green')
        
        ax2.set_xlabel(f'{variable}', fontsize=12)
        ax2.set_ylabel('Scale Factor\n(Data/MC)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.8, 1.2)
        
        # Add CMS label
        ax1.text(0.02, 0.95, 'CMS', transform=ax1.transAxes, 
                fontsize=16, fontweight='bold')
        ax1.text(0.02, 0.88, 'Simulation Preliminary', transform=ax1.transAxes, 
                fontsize=12, style='italic')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"efficiency_ratio_{variable}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
        
        plt.show()
    
    def _plot_ml_performance(self, variable, save_plot=True):
        """
        Plot ML model performance.
        """
        model_data = self.ratio_predictors[variable]
        results = self.efficiency_results[variable]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top left: True ratios vs bin centers
        bin_centers = model_data['bin_centers']
        true_ratios = model_data['true_ratios']
        
        ax1.plot(bin_centers, true_ratios, 'ro-', markersize=6, linewidth=2)
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax1.set_xlabel(f'{variable}')
        ax1.set_ylabel('Data/MC Ratio')
        ax1.set_title(f'True Efficiency Ratios vs {variable}')
        ax1.grid(True, alpha=0.3)
        
        # Top right: Feature importance
        model = model_data['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = model_data['features']
            
            indices = np.argsort(importances)[::-1]
            ax2.bar(range(len(features)), importances[indices], 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(features)])
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Importance')
            ax2.set_title('ML Model Feature Importance')
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels([features[i] for i in indices], rotation=45)
        
        # Bottom left: Model performance metrics
        ax3.text(0.1, 0.8, f"Model: Gradient Boosting", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.7, f"MSE: {model_data['mse']:.4f}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f"R²: {model_data['r2']:.3f}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f"Features: {len(model_data['features'])}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.4, f"Target: {variable} ratios", fontsize=12, transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Model Performance Summary')
        
        # Bottom right: Ratio distribution
        ax4.hist(true_ratios, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=1, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Data/MC Ratio')
        ax4.set_ylabel('Number of Bins')
        ax4.set_title('Distribution of Scale Factors')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"ml_performance_{variable}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
        
        plt.show()
    
    def save_results(self, filename=None):
        """
        Save all results to files for later use.
        
        Args:
            filename (str): Base filename (optional)
        """
        if filename is None:
            filename = f"efficiency_ratios_{self.version}_{self.run_name}"
        
        # Save efficiency results
        results_file = self.output_dir / f"{filename}_results.npz"
        save_data = {}
        for var, data in self.efficiency_results.items():
            for key, value in data.items():
                save_data[f"{var}_{key}"] = value
        
        np.savez(results_file, **save_data)
        print(f"Results saved to: {results_file}")
        
        # Save summary report
        report_file = self.output_dir / f"{filename}_summary.txt"
        with open(report_file, 'w') as f:
            f.write(f"CMS Trigger Efficiency Ratio Analysis\n")
            f.write(f"Version: {self.version}\n")
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")
            
            if self.data_sample is not None:
                f.write(f"Data events: {len(self.data_sample)}\n")
                f.write(f"MC events: {len(self.mc_sample)}\n")
            
            f.write(f"\nVariables analyzed: {list(self.efficiency_results.keys())}\n")
            f.write(f"ML predictors trained: {list(self.ratio_predictors.keys())}\n")
            
            f.write(f"\nScale Factor Summary:\n")
            for var, results in self.efficiency_results.items():
                mean_sf = np.mean(results['ratio'])
                std_sf = np.std(results['ratio'])
                f.write(f"  {var}: {mean_sf:.3f} ± {std_sf:.3f}\n")
        
        print(f"Summary saved to: {report_file}")
    
    def run_complete_analysis(self):
        """
        Run the complete efficiency ratio analysis workflow.
        """
        print("\n" + "="*60)
        print("CMS Trigger Efficiency Ratio Analysis")
        print("="*60)
        
        # Step 1: Load or generate data
        if self.data_sample is None:
            print("\nStep 1: Loading data...")
            self.generate_demo_data()
        
        # Step 2: Calculate efficiency ratios
        print("\nStep 2: Calculating efficiency ratios...")
        self.calculate_efficiency_ratios()
        
        # Step 3: Train ML predictors
        print("\nStep 3: Training ML ratio predictors...")
        self.train_ratio_predictors()
        
        # Step 4: Apply corrections
        print("\nStep 4: Applying corrections to MC...")
        for method in ['interpolation', 'ml']:
            corrections = self.apply_corrections_to_mc('HighestPt', method)
            print(f"  {method.capitalize()}: Mean correction = {np.mean(corrections):.3f}")
        
        # Step 5: Generate plots
        print("\nStep 5: Creating plots...")
        self.plot_results()
        
        # Step 6: Save results
        print("\nStep 6: Saving results...")
        self.save_results()
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"Results available in: {self.output_dir}")
        print("="*60)


def main():
    """
    Main function demonstrating the integrated ratio analyzer.
    """
    # Create analyzer instance
    analyzer = DataMCRatioAnalyzer(version="v5", run_name="demo")
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Demonstrate individual predictions
    print("\nExample: Predicting ratio for individual events:")
    sample_event = {
        'HighestPt': 400,
        'HT': 800,
        'MET_pt': 50,
        'HighestMass': 80
    }
    
    predicted_ratio = analyzer.predict_ratio_for_event(sample_event, 'HighestPt')
    print(f"Event with pT={sample_event['HighestPt']} GeV: predicted ratio = {predicted_ratio:.3f}")


if __name__ == "__main__":
    main()
