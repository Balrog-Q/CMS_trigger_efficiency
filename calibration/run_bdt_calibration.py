#!/usr/bin/env python3
"""
Run BDT Calibration - Simple Integration Script
===============================================

This script provides a simple interface to run the BDT probability calibration workflow
for CMS trigger efficiency analysis. It allows you to easily configure and execute
the complete analysis with QCD-only calibration.

Usage:
    python run_bdt_calibration.py
    
Or with custom parameters:
    python run_bdt_calibration.py --data_version testing --random_state 42
"""

import argparse
import sys
import os
from datetime import datetime

# Add current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from calibration.bdt_calibration_workflow import BDTCalibrationWorkflow
    from calibration.probability_calibration import BDTCalibrator
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required files are in the same directory:")
    print("- bdt_calibration_workflow.py")
    print("- probability_calibration.py")
    print("- library/plotting_tools.py")
    sys.exit(1)


def check_data_availability(data_version):
    """Check if the required data files exist."""
    data_path = f"data/processed/{data_version}/"
    required_files = [
        f"{data_version}-NewQCD.root",
        f"{data_version}-NewggF.root", 
        f"{data_version}-NewVBF.root"
    ]
    
    missing_files = []
    for filename in required_files:
        full_path = data_path + filename
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    
    return missing_files


def list_available_data_versions():
    """List available data versions."""
    processed_dir = "data/processed/"
    if not os.path.exists(processed_dir):
        return []
    
    versions = []
    for item in os.listdir(processed_dir):
        item_path = os.path.join(processed_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains ROOT files
            root_files = [f for f in os.listdir(item_path) if f.endswith('.root')]
            if root_files:
                versions.append(item)
    
    return sorted(versions)


def main():
    """Main function to run the BDT calibration workflow."""
    
    print("="*80)
    print("BDT PROBABILITY CALIBRATION FOR CMS TRIGGER EFFICIENCY")
    print("="*80)
    print("This script will train BDT models on QCD data and apply probability calibration")
    print("using both isotonic regression and Platt scaling methods.\n")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run BDT probability calibration workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bdt_calibration.py
  python run_bdt_calibration.py --data_version azura --random_state 123
  python run_bdt_calibration.py --list_versions
        """
    )
    
    parser.add_argument(
        '--data_version', 
        type=str, 
        default='azura',
        help='Data version to use (default: azura)'
    )
    
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--list_versions',
        action='store_true',
        help='List available data versions and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list versions request
    if args.list_versions:
        print("Available data versions:")
        versions = list_available_data_versions()
        if versions:
            for version in versions:
                print(f"  - {version}")
        else:
            print("  No processed data found in data/processed/")
        sys.exit(0)
    
    # Display configuration
    print(f"Configuration:")
    print(f"  Data version: {args.data_version}")
    print(f"  Random state: {args.random_state}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data availability
    print(f"\nChecking data availability...")
    missing_files = check_data_availability(args.data_version)
    
    if missing_files:
        print("ERROR: Missing required data files:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\nAvailable data versions:")
        versions = list_available_data_versions()
        for version in versions:
            print(f"  - {version}")
        print(f"\nPlease:")
        print(f"1. Make sure you've processed the data using qcd_processing_data.py")
        print(f"2. Use --data_version with one of the available versions")
        print(f"3. Or use --list_versions to see all available options")
        sys.exit(1)
    
    print("✓ All required data files found!")
    
    # Check dependencies
    print(f"\nChecking dependencies...")
    try:
        import ROOT
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✓ All required libraries are available!")
    except ImportError as e:
        print(f"ERROR: Missing required library: {e}")
        print(f"Please install missing dependencies using:")
        print(f"  pip install xgboost lightgbm catboost scikit-learn matplotlib seaborn pandas")
        sys.exit(1)
    
    # Initialize and run workflow
    try:
        print(f"\nInitializing BDT calibration workflow...")
        workflow = BDTCalibrationWorkflow(
            data_version=args.data_version,
            random_state=args.random_state
        )
        
        print(f"Starting workflow execution...")
        workflow.run_complete_workflow()
        
        print(f"\n" + "="*80)
        print(f"SUCCESS: BDT calibration workflow completed!")
        print(f"="*80)
        print(f"Results saved in: {workflow.results_dir}")
        print(f"\nKey outputs:")
        print(f"  - Trained models: models/{{model_name}}_model.pkl")
        print(f"  - Calibrators: calibrators/{{model_name}}_calibrator.pkl")
        print(f"  - Analysis report: calibration_analysis_report.md")
        print(f"  - Calibration plots: *.png files")
        print(f"  - Evaluation results: evaluation_results.pkl")
        
        print(f"\nNext steps:")
        print(f"  1. Review the analysis report for detailed results")
        print(f"  2. Examine calibration plots to understand model behavior")
        print(f"  3. Use calibrated models in your trigger efficiency studies")
        print(f"  4. Consider the best-performing model based on ECE metrics")
        
    except KeyboardInterrupt:
        print(f"\n\nWorkflow interrupted by user.")
        print(f"Partial results may be available in the results directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nERROR during workflow execution:")
        print(f"{str(e)}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Check that data files are not corrupted")
        print(f"2. Ensure sufficient memory is available")
        print(f"3. Verify all dependencies are correctly installed")
        print(f"4. Check file permissions in the current directory")
        
        # Print detailed error information if available
        import traceback
        print(f"\nDetailed error information:")
        traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
