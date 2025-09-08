"""
Data Loading Utility for GBDT Comparison

This module implements the exact data loading methodology described in the research:
- ROOT data loading from azura-NewQCD.root
- Bootstrap sampling with replacement
- Feature extraction using define_parameter function
"""

import os
import numpy as np
import pandas as pd
import ROOT
from typing import Tuple, Optional

# Import from existing library
from library.trigger_efficiency_ML import define_parameter

class QCDDataLoader:
    """
    Data loader for QCD dataset following the research methodology.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data loader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def load_qcd_data(self, 
                      root_file_path: str = "data/processed/azura/azura-NewQCD.root",
                      tree_name: str = "azura-NewQCD",
                      filter_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load QCD data following the exact methodology from the research.
        
        This implements the code snippet:
        ```
        Import ROOT
        df_QCD = ROOT.RDataFrame("azura-NewQCD", "data/processed/azura/azura-NewQCD.root")
        
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, 
        range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("QCD")
        
        df_ref = df_QCD.Filter(filter_all_meas, "")
        npy = df_ref.AsNumpy(columns=names_list_and_signal_trigger)
        df = pd.DataFrame(npy)
        
        X = df.drop('Combo', axis=1).values
        y = df['Combo'].astype('int').values
        
        index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sample = X[index_x]
        y_sample = y[index_x]
        ```
        
        Args:
            root_file_path: Path to the ROOT file
            tree_name: Name of the tree in the ROOT file
            filter_name: Optional filter to apply (e.g., "filter_all_meas")
            
        Returns:
            Tuple of (X_sample, y_sample, metadata)
        """

        print(f"Loading QCD data from {root_file_path}...")
        
        # Check if file exists
        if not os.path.exists(root_file_path):
            raise FileNotFoundError(f"ROOT file not found: {root_file_path}")
             
        # Load ROOT data
        df_QCD = ROOT.RDataFrame(tree_name, root_file_path)

        # Get parameter definitions for QCD
        (variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list) = define_parameter("QCD")

        # Apply filter if provided
        if filter_name:
            try:
                df_ref = df_QCD.Filter(filter_name, "")
                print(f"Applied filter: {filter_name}")
            except Exception as e:
                print(f"Warning: Could not apply filter '{filter_name}': {e}")
                print("Using unfiltered data")
                df_ref = df_QCD
        else:
            df_ref = df_QCD
            print("No filter applied")
        
        # Convert to NumPy array
        print("Converting ROOT data to NumPy array...")
        npy = df_ref.AsNumpy(columns=names_list_and_signal_trigger)
        
        # Create pandas DataFrame
        df = pd.DataFrame(npy)

        print(f"Data shape before sampling: {df.shape}")
        print(f"Available columns: {list(df.columns)}")

        # Prepare features and targets
        X = df.drop('Combo', axis=1).values
        y = df['Combo'].astype('int').values

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")

        # Apply bootstrap sampling with replacement (as in original code)
        print("Applying bootstrap sampling...")
        index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_sample = X[index_x]
        y_sample = y[index_x]
        
        print(f"Sampled data shape: {X_sample.shape}")
        print(f"Sampled class distribution: {np.bincount(y_sample)}")
        
        # Prepare metadata
        metadata = {'feature_names': names_list,
                    'feature_names_plot': names_list_plot,
                    'variable_list': variable_list,
                    'range_min_list': range_min_list,
                    'range_max_list': range_max_list,
                    'num_bins_list': num_bins_list,
                    'y_min_list': y_min_list,
                    'y_max_list': y_max_list,
                    'original_shape': X.shape,
                    'sampled_shape': X_sample.shape
                    }
        
        return X_sample, y_sample, metadata
    
    def get_feature_info(self) -> dict:
        """
            Get feature information for QCD dataset.
            Returns:
                Dictionary containing feature definitions and ranges
        """

        (variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list) = define_parameter("QCD")

        feature_info = {'names': names_list, 
                        'plot_names': names_list_plot, 
                        'variables': variable_list, 
                        'min_ranges': range_min_list, 
                        'max_ranges': range_max_list, 
                        'num_bins': num_bins_list
                        }
        
        return feature_info


# Standalone function for quick data loading (matching the research code exactly)
def load_qcd_data_simple(root_file_path: str = "data/processed/azura/azura-NewQCD.root",filter_all_meas: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple function that exactly implements the code from the research:
                            
    ```python
    Import ROOT
    df_QCD = ROOT.RDataFrame("azura-NewQCD", "data/processed/azura/azura-NewQCD.root")
                            
    variable_list, names_list, names_list_and_signal_trigger, names_list_plot, 
    range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("QCD")
                            
    df_ref = df_QCD.Filter(filter_all_meas, "")
    npy = df_ref.AsNumpy(columns=names_list_and_signal_trigger)
    df = pd.DataFrame(npy)
                            
    X = df.drop('Combo', axis=1).values
    y = df['Combo'].astype('int').values
                            
    index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
    X_sample = X[index_x]
    y_sample = y[index_x]
    ```
                            
    Args:
        root_file_path: Path to the ROOT file
        filter_all_meas: Filter string to apply
                                
    Returns:
        Tuple of (X_sample, y_sample)
    """
    
    # Import ROOT (as in the original code)
    import ROOT
    
    # Load the ROOT DataFrame
    df_QCD = ROOT.RDataFrame("azura-NewQCD", root_file_path)

    # Get parameter definitions
    (variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list) = define_parameter("QCD")
    
    # Apply filter if provided
    if filter_all_meas:
        df_ref = df_QCD.Filter(filter_all_meas, "")
    else:
        df_ref = df_QCD
        
    # Convert to NumPy
    npy = df_ref.AsNumpy(columns=names_list_and_signal_trigger)
    df = pd.DataFrame(npy)
    
    # Prepare features and targets
    X = df.drop('Combo', axis=1).values
    y = df['Combo'].astype('int').values
    
    # Bootstrap sampling
    index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
    X_sample = X[index_x]
    y_sample = y[index_x]
    
    return X_sample, y_sample


if __name__ == "__main__":
    # Example usage
    loader = QCDDataLoader(random_state=42)
    
    try:
        X, y, metadata = loader.load_qcd_data()
        print(f"\nData loaded successfully!")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Feature names: {metadata['feature_names']}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the ROOT file exists and is accessible.")
