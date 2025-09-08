#!/usr/bin/env python3
"""
Example Integration: CMS Trigger Efficiency Ratio Prediction
===========================================================

This example shows how to integrate the efficiency ratio predictor 
with your existing CMS trigger efficiency analysis workflow.

This demonstrates the complete workflow from loading your ROOT data 
to applying data/MC scale factor corrections.
"""

import sys
import numpy as np
import pandas as pd
import ROOT

# Import your existing modules
from integrated_ratio_predictor_fixed import DataMCRatioAnalyzer

def load_your_data():
    """
    Example of how to load your existing ROOT data.
    Replace this with your actual data loading logic.
    """
    print("Loading CMS data...")
    
    # This would normally be your ROOT DataFrames:
    df_DATA = ROOT.RDataFrame("briar-NewDATA", "data/processed/briar/briar-NewDATA.root")
    df_QCD = ROOT.RDataFrame("briar-NewQCD", "data/processed/briar/briar-NewQCD.root")
    df_ggF = ROOT.RDataFrame("briar-NewggF", "data/processed/briar/briar-NewggF.root")
    df_VBF = ROOT.RDataFrame("briar-NewVBF", "data/processed/briar/briar-NewVBF.root")
    
    # For this demo, we'll use the built-in data generator
    # return None, None, None, None  # DATA, QCD, ggF, VBF
    return df_DATA, df_QCD, df_ggF, df_VBF


def apply_scale_factors_to_analysis(mc_events, scale_factor_analyzer):
    """
    Example of how to apply scale factors to your MC events in a real analysis.
    
    Args:
        mc_events (pd.DataFrame): Your MC events
        scale_factor_analyzer (DataMCRatioAnalyzer): Trained analyzer
    
    Returns:
        pd.DataFrame: MC events with corrected weights
    """
    print("Applying data/MC scale factors to MC events...")
    
    # Method 1: Use interpolated scale factors
    sf_corrections_interp = scale_factor_analyzer.apply_corrections_to_mc(
        variable='HighestPt', 
        method='interpolation'
    )
    
    # Method 2: Use ML-predicted scale factors
    sf_corrections_ml = scale_factor_analyzer.apply_corrections_to_mc(
        variable='HighestPt', 
        method='ml'
    )
    
    # Apply corrections to event weights
    mc_events_corrected = mc_events.copy()
    
    # Choose your preferred method
    if 'weight' in mc_events.columns:
        mc_events_corrected['weight_sf_interp'] = mc_events['weight'] * sf_corrections_interp
        mc_events_corrected['weight_sf_ml'] = mc_events['weight'] * sf_corrections_ml
    else:
        mc_events_corrected['weight_sf_interp'] = sf_corrections_interp
        mc_events_corrected['weight_sf_ml'] = sf_corrections_ml
    
    # Print correction summary
    print(f"Original events: {len(mc_events)}")
    print(f"Mean correction (interpolation): {np.mean(sf_corrections_interp):.3f}")
    print(f"Mean correction (ML): {np.mean(sf_corrections_ml):.3f}")
    print(f"Correction range: [{np.min(sf_corrections_interp):.3f}, {np.max(sf_corrections_interp):.3f}]")
    
    return mc_events_corrected


def event_by_event_prediction(analyzer):
    """
    Demonstrate event-by-event scale factor prediction.
    This is useful for applying corrections in your analysis loop.
    """
    print("\nDemonstrating event-by-event predictions:")
    print("-" * 40)
    
    # Example events with different kinematic properties
    test_events = [
        {'HighestPt': 300, 'HT': 600, 'MET_pt': 30, 'HighestMass': 60},
        {'HighestPt': 500, 'HT': 1000, 'MET_pt': 80, 'HighestMass': 100},
        {'HighestPt': 800, 'HT': 1500, 'MET_pt': 120, 'HighestMass': 150},
    ]
    
    for i, event in enumerate(test_events):
        sf = analyzer.predict_ratio_for_event(event, 'HighestPt')
        print(f"Event {i+1}: pT={event['HighestPt']:.0f} GeV, HT={event['HT']:.0f} GeV → SF = {sf:.3f}")


def create_efficiency_lookup_table(analyzer, variable='HighestPt', n_points=50):
    """
    Create a lookup table for fast scale factor application.
    Useful for large-scale analyses.
    """
    print(f"\nCreating lookup table for {variable}:")
    print("-" * 40)
    
    if variable not in analyzer.efficiency_results:
        print(f"No results available for {variable}")
        return None
    
    # Get variable range
    results = analyzer.efficiency_results[variable]
    var_min = results['bin_centers'].min()
    var_max = results['bin_centers'].max()
    
    # Create lookup points
    lookup_points = np.linspace(var_min, var_max, n_points)
    
    # Get scale factors at these points
    scale_factors = np.interp(
        lookup_points,
        results['bin_centers'],
        results['ratio']
    )
    
    # Create lookup table
    lookup_table = pd.DataFrame({
        variable: lookup_points,
        'scale_factor': scale_factors
    })
    
    print(f"Created lookup table with {n_points} points")
    print(f"Variable range: [{var_min:.1f}, {var_max:.1f}]")
    print(f"Scale factor range: [{scale_factors.min():.3f}, {scale_factors.max():.3f}]")
    
    # Save lookup table
    output_file = f"scale_factor_lookup_{variable}.csv"
    lookup_table.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return lookup_table


def systematic_uncertainty_analysis(analyzer):
    """
    Analyze systematic uncertainties in scale factors.
    """
    print("\nSystematic Uncertainty Analysis:")
    print("-" * 40)
    
    for variable, results in analyzer.efficiency_results.items():
        ratio = results['ratio']
        ratio_err = results['ratio_error']
        
        # Calculate systematic uncertainties
        max_deviation = np.max(np.abs(ratio - 1.0))
        rms_uncertainty = np.sqrt(np.mean(ratio_err**2))
        
        print(f"{variable}:")
        print(f"  Maximum deviation from unity: {max_deviation:.3f}")
        print(f"  RMS uncertainty: {rms_uncertainty:.3f}")
        print(f"  Relative systematic: {rms_uncertainty:.1%}")


def comparison_with_traditional_methods(analyzer):
    """
    Compare ML predictions with traditional binned approach.
    """
    print("\nComparison: ML vs Traditional Methods:")
    print("-" * 40)
    
    for variable in ['HighestPt', 'HT']:
        if variable not in analyzer.ratio_predictors:
            continue
            
        model_data = analyzer.ratio_predictors[variable]
        r2_score = model_data['r2']
        mse = model_data['mse']
        
        print(f"{variable}:")
        print(f"  ML Model R²: {r2_score:.4f}")
        print(f"  ML Model MSE: {mse:.6f}")
        print(f"  → ML explains {r2_score:.1%} of the variance")


def save_for_production_use(analyzer, version="v5"):
    """
    Save results in format suitable for production analysis.
    """
    print(f"\nSaving results for production use (version {version}):")
    print("-" * 50)
    
    # Save models for later use
    import pickle
    
    for variable, model_data in analyzer.ratio_predictors.items():
        model_file = f"sf_predictor_{variable}_{version}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved ML model: {model_file}")
    
    # Save lookup tables for key variables
    for variable in ['HighestPt', 'HT']:
        if variable in analyzer.efficiency_results:
            lookup_table = create_efficiency_lookup_table(analyzer, variable, n_points=100)
    
    print("All production files saved!")


def main():
    """
    Main function demonstrating complete integration workflow.
    """
    print("="*70)
    print("CMS Trigger Efficiency: Data/MC Ratio Prediction Integration")
    print("="*70)
    
    # Step 1: Load your data (replace with actual ROOT data loading)
    print("\n1. Loading Data")
    df_data, df_qcd, df_ggf, df_vbf = load_your_data()
    
    # Step 2: Create and configure the analyzer
    print("\n2. Setting up Efficiency Ratio Analyzer")
    analyzer = DataMCRatioAnalyzer(version="v5", run_name="production")
    
    # Step 3: Run the complete analysis
    print("\n3. Running Analysis")
    # For demo, this will use generated data
    # In production, you'd call: analyzer.load_data_from_root(df_data, df_qcd)
    analyzer.run_complete_analysis()
    
    # Step 4: Demonstrate practical applications
    print("\n4. Practical Applications")
    
    # Event-by-event predictions
    event_by_event_prediction(analyzer)
    
    # Create lookup tables
    create_efficiency_lookup_table(analyzer, 'HighestPt')
    
    # Systematic uncertainty analysis
    systematic_uncertainty_analysis(analyzer)
    
    # Compare methods
    comparison_with_traditional_methods(analyzer)
    
    # Step 5: Apply to your MC sample
    if analyzer.mc_sample is not None:
        print("\n5. Applying Scale Factors")
        corrected_mc = apply_scale_factors_to_analysis(analyzer.mc_sample, analyzer)
        
        # Show impact on your analysis
        original_sum = analyzer.mc_sample['Combo'].sum()  # Original triggered events
        
        # Calculate weighted sums with corrections
        if 'weight_sf_interp' in corrected_mc.columns:
            corrected_sum_interp = np.sum(corrected_mc[corrected_mc['Combo']==1]['weight_sf_interp'])
            corrected_sum_ml = np.sum(corrected_mc[corrected_mc['Combo']==1]['weight_sf_ml'])
            
            print(f"\nImpact on triggered event yield:")
            print(f"  Original MC: {original_sum} events")
            print(f"  With SF (interp): {corrected_sum_interp:.1f} effective events")
            print(f"  With SF (ML): {corrected_sum_ml:.1f} effective events")
            print(f"  Correction factor: {corrected_sum_interp/original_sum:.3f}")
    
    # Step 6: Save for production use
    print("\n6. Saving for Production")
    save_for_production_use(analyzer)
    
    print("\n" + "="*70)
    print("Integration Complete!")
    print("="*70)
    print("\nYou can now:")
    print("• Use the saved models in your production analysis")
    print("• Apply event-by-event corrections using predict_ratio_for_event()")
    print("• Use lookup tables for fast bulk corrections")
    print("• Include systematic uncertainties from ratio_error")
    print("\nExample usage in your analysis:")
    print("  sf = analyzer.predict_ratio_for_event(event_features)")
    print("  corrected_weight = original_weight * sf")


if __name__ == "__main__":
    main()
