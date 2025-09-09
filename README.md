# CMS Trigger Efficiency Analysis Using Gradient Boosting
*A comprehensive machine learning framework for measuring trigger efficiency in CMS HH→bbττ analysis*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ROOT](https://img.shields.io/badge/ROOT-6.24+-red.svg)](https://root.cern/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0+-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.3+-orange.svg)](https://xgboost.readthedocs.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Gradient Boosting Analysis](#running-the-gradient-boosting-analysis)
- [Data Requirements](#data-requirements)
- [Analysis Components](#analysis-components)
- [Output and Results](#output-and-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔬 Overview

This project implements a comprehensive machine learning approach to measure trigger efficiency in CMS (Compact Muon Solenoid) experiments, specifically for the HH→bbττ analysis. The framework uses gradient boosting methods to predict trigger efficiency, providing a data-driven approach to correct for trigger inefficiencies in physics measurements.

### Key Features

- **Multi-algorithm Support**: XGBoost, LightGBM, CatBoost, and scikit-learn GradientBoosting
- **Comprehensive Workflow**: From data processing to final efficiency measurements
- **Probability Calibration**: Ensures reliable probability estimates
- **Cutflow Analysis**: Track event selection efficiency step-by-step
- **Scale Factor Generation**: For systematic uncertainty estimation
- **Model Comparison**: Automated hyperparameter optimization and model selection

### Physics Context

The analysis focuses on:
- **Signal Triggers**: Complex OR combination of HLT paths including jet, HT, MET, and tau triggers
- **Reference Trigger**: HLT_AK8PFJet260 for unbiased efficiency measurement
- **Samples**: QCD background, ggF HH signal, VBF HH signal, and collision data
- **Variables**: Fat jet properties, HT, MET, invariant masses, and kinematic variables

## 📁 Project Structure

```
CMS-trigger-efficiency-ashe/
├── 📊 run_gradient_boosting.ipynb     # Main analysis notebook
├── 📊 data_distribution.ipynb         # Data distribution analysis
├── 📊 composing_plot.ipynb           # Plotting utilities
├── 
├── 📂 library/                       # Core analysis modules
│   ├── trigger_efficiency_ML.py     # Main ML functions and plotting
│   └── processing_data.py            # Data processing utilities
│
├── 📂 data/                          # Data storage
│   ├── HHbbtautau-v1/               # Raw ROOT files (v1)
│   ├── HHbbtautau-v2/               # Raw ROOT files (v2)  
│   └── processed/                   # Processed data files
│       ├── azura/                   # Tau & new processed data
│       ├── ashe/                    # No tau & new processed data
│       ├── briar/                   # Old processed data
│       └── ...
│
├── 📂 model_selection/               # Model comparison framework
│   ├── run_gbdt_comparison.py       # Execute model comparison
│   ├── gbdt_comparison_framework.py # Framework implementation
│   ├── gbdt_training_pipeline.py    # Training pipeline
│   ├── hyperparameter_optimization.py # HPO utilities
│   ├── data_loader.py               # Data loading utilities
│   ├── pipeline_clean.py            # Clean analysis pipeline
│   └── requirements_gbdt.txt        # GBDT-specific requirements
│
├── 📂 calibration/                   # Probability calibration
│   ├── run_bdt_calibration.py       # Execute calibration
│   ├── bdt_calibration_workflow.py  # Workflow implementation
│   ├── probability_calibration.py   # Calibration methods
│   ├── bdt_calibration_example.py   # Usage examples
│   ├── test_gradientboosting_integration.py # Integration tests
│   └── README_BDT_Calibration.md    # Detailed calibration docs
│
├── 📂 cutflow/                       # Selection analysis
│   ├── run_cutflow.py               # Execute cutflow analysis
│   ├── cutflow_analysis.py          # Core cutflow implementation
│   ├── cutflow_example.py           # Advanced cutflow examples
│   └── README_cutflow.md            # Cutflow documentation
│
├── 📂 scale_factor/                  # Systematic uncertainties
│   ├── efficiency_ratio_predictor.py # Ratio prediction methods
│   ├── xgboost_scale_factor_predictor.py # XGBoost-based scale factors
│   ├── integrated_ratio_predictor_fixed.py # Fixed integration method
│   ├── example_integration.py        # Integration examples
│   └── README_Efficiency_Ratio_Prediction.md # Scale factor docs
│
└── 📂 result/                        # Output directory
    └── DD-MM-YYYY-suffix/           # Date-stamped results
        ├── plots/                   # Generated plots
        ├── models/                  # Saved models
        └── reports/                 # Analysis reports
```

## 🚀 Quick Start

### Prerequisites

Ensure you have access to the **CERN CMS environment** with all required modules:

```bash
# Activate CERN CMS environment (recommended)
source /cvmfs/cms.cern.ch/cmsset_default.sh
# or your local CMS environment setup
```
### First time?
To prepare the environment for first time run
```Python
# In CERN CMS environment (recommended)
source /cvmfs/cms.cern.ch/cmsset_default.sh
pip install --user -r requirements.txt

# Or locally
pip install -r requirements.txt
```
For future updates
```Python
# Re-run the extraction anytime
python extract_libraries.py
```
### Run the Main Analysis

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/CMS-trigger-efficiency-ashe
   ```

2. **Open the main analysis notebook:**
   ```bash
   jupyter notebook run_gradient_boosting.ipynb
   ```

3. **Configure your analysis** (in the notebook's second cell):
   ```python
   # Define the run name and version
   run_name = "Run2"
   version = "v2"  # Change to v1, v3, v4, etc. as needed
   
   # Define samples to analyze
   samples = ["QCD", "ggF", "VBF", "DATA"]
   
   # Save trained models?
   save_model_gradu = False  # Set to True to save models
   save_model_data = False
   ```

4. **Execute all cells** to run the complete analysis pipeline

## 🛠 Installation

### Using CERN CMS Environment (Recommended)

The project is designed to work within the CERN CMS software environment where most dependencies are pre-installed:

```bash
# Set up CMS environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

### Local Installation

If running outside CERN, install the required packages:

```bash
# Core ML and data analysis
pip install scikit-learn>=1.7.0
pip install xgboost>=3.0.3
pip install lightgbm>=4.6.0
pip install catboost>=1.2.68
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0

# HEP-specific packages
pip install mplhep
pip install hist

# Additional utilities
pip install tqdm
pip install optuna>=3.0.0  # For hyperparameter optimization
```

### ROOT Installation

ROOT is essential for reading CMS data files. Install from:
- **CERN users**: Usually pre-installed in CMS environment
- **Local users**: Follow [ROOT installation guide](https://root.cern/install/)

## 🔬 Running the Gradient Boosting Analysis

### Main Workflow: `run_gradient_boosting.ipynb`

This Jupyter notebook implements the complete gradient boosting analysis for trigger efficiency measurement.

#### Step-by-Step Execution

1. **Configuration (Cells 1-2)**
   ```python
   # Set run parameters
   run_name = "Run2"
   version = "v2"  # Data version (v1, v2, v3, etc.)
   samples = ["QCD", "ggF", "VBF", "DATA"]
   
   # Configure triggers
   signal_triggers = [
       "HLT_AK8PFHT800_TrimMass50",
       "HLT_AK8PFJet400_TrimMass30", 
       "HLT_AK8PFJet500",
       "HLT_PFJet500",
       "HLT_PFHT1050",
       # ... additional triggers
   ]
   reference_trigger = "HLT_AK8PFJet260"
   ```

2. **Data Loading (Cell 3)**
   - Automatically loads processed ROOT files
   - Maps data versions to file paths
   - Creates RDataFrame objects for analysis

3. **Data Distribution Analysis (Cells 4-6)**
   - Generates comparison plots for all kinematic variables
   - Creates efficiency plots for each sample
   - Outputs: `Distribution_MC_*.png` files

4. **Machine Learning Training (Cells 7-10)**
   - Trains gradient boosting models on QCD data
   - Applies trained model to all samples
   - Generates efficiency predictions

5. **Validation and Comparison (Cells 11-15)**
   - Compares ML predictions with measured efficiencies
   - Creates validation plots
   - Outputs: Efficiency comparison plots

### Key Analysis Variables

The analysis uses these kinematic variables:

```python
analysis_variables = [
    "HighestPt",        # Leading fat jet pT
    "HT",               # Scalar sum of jet pT  
    "MET_pt",           # Missing transverse energy
    "mHH",              # Invariant mass of HH system
    "HighestMass",      # Leading fat jet mass
    "SecondHighestPt",  # Sub-leading fat jet pT
    "SecondHighestMass", # Sub-leading fat jet mass
    "FatHT",            # HT from fat jets only
    "MET_FatJet",       # MET projected on fat jets
    "mHHwithMET",       # HH+MET invariant mass
    "HighestEta",       # Leading fat jet η
    "SecondHighestEta", # Sub-leading fat jet η
    "DeltaEta",         # |η₁ - η₂|
    "DeltaPhi"          # |φ₁ - φ₂|
]
```

### Trigger Configuration

The analysis measures efficiency for this complex trigger combination:

**Signal Triggers (OR combination):**
- `HLT_AK8PFHT800_TrimMass50`
- `HLT_AK8PFJet400_TrimMass30` 
- `HLT_AK8PFJet500`
- `HLT_PFJet500`
- `HLT_PFHT1050`
- `HLT_PFHT500_PFMET100_PFMHT100_IDTight`
- `HLT_PFHT700_PFMET85_PFMHT85_IDTight`
- `HLT_PFHT800_PFMET75_PFMHT75_IDTight`
- `HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg`
- `HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1`

**Reference Trigger:**
- `HLT_AK8PFJet260` (unbiased reference)

## 💾 Data Requirements

### Processed Data Structure

The analysis expects data in this format:

```
data/processed/{version}/
├── {version}-NewQCD.root    # QCD background sample
├── {version}-NewggF.root    # ggF HH signal sample  
├── {version}-NewVBF.root    # VBF HH signal sample
└── {version}-NewDATA.root   # Collision data sample
```

### Data Versions Available

- **v1 (briar)**: Initial data processing
- **v2 (azura)**: Updated selection criteria  
- **v3 (ashe)**: Current recommended version
- **v4 (cypress)**: Alternative processing
- **v5 (azura-v2)**: Refined v2 processing

### Required Branches in ROOT Files

Each ROOT file must contain:

```cpp
// Kinematic variables
Float_t HighestPt, SecondHighestPt;
Float_t HT, FatHT;
Float_t MET_pt, MET_FatJet;
Float_t mHH, mHHwithMET;
Float_t HighestMass, SecondHighestMass;
Float_t HighestEta, SecondHighestEta;
Float_t DeltaEta, DeltaPhi;

// Trigger information
Bool_t Combo;                    // Signal trigger combination
Bool_t HLT_AK8PFJet260;         // Reference trigger
// ... individual trigger branches
```

## 📊 Analysis Components

### 1. Model Selection Framework

Located in `model_selection/`, this component compares different GBDT algorithms:

```bash
# Run comprehensive model comparison
python model_selection/run_gbdt_comparison.py

# With custom settings
python model_selection/run_gbdt_comparison.py --n_trials 100 --cv_folds 5
```

**Features:**
- Automated hyperparameter optimization with Optuna
- Cross-validation evaluation
- Performance comparison across algorithms
- Best model selection and saving

### 2. Probability Calibration

Located in `calibration/`, ensures reliable probability estimates:

```bash
# Run BDT calibration workflow  
python calibration/run_bdt_calibration.py

# List available data versions
python calibration/run_bdt_calibration.py --list_versions
```

**Features:**
- Isotonic regression and Platt scaling
- Cross-validation to prevent overfitting
- Calibration quality metrics (ECE, MCE, Brier score)
- Calibration curve visualization

### 3. Cutflow Analysis

Located in `cutflow/`, tracks selection efficiency:

```bash
# Run cutflow analysis
python cutflow/run_cutflow.py
```

**Features:**
- Step-by-step efficiency tracking
- Event count monitoring
- Selection optimization insights
- Multi-sample comparison

### 4. Scale Factor Generation  

Located in `scale_factor/`, provides systematic uncertainty estimates:

```bash
# Generate efficiency ratio predictions
python scale_factor/efficiency_ratio_predictor.py
```

**Features:**
- Data/MC scale factor calculation
- Uncertainty propagation
- Integration with trigger efficiency
- Ratio prediction methods

## 📈 Output and Results

### Generated Files

After running the main analysis, you'll find results in `result/DD-MM-YYYY-suffix/`:

#### Plots
- `{Sample}_{Variable}_Run2_both.png` - Individual sample distributions
- `Distribution_MC_{Variable}.png` - Combined MC sample comparisons  
- `Efficiency_{Sample}_{Variable}.png` - Trigger efficiency plots
- `Validation_{Method}_{Variable}.png` - ML validation plots

#### Models
- `gb_{suffix}_200et0p2_gradu.sav` - Gradient boosting model (MC training)
- `gb_{suffix}_200et0p2_data.sav` - Gradient boosting model (data training)

#### Reports
- Analysis summary with key metrics
- Model performance comparisons
- Calibration quality assessments

### Typical Results

**Expected Efficiency Values:**
- **QCD Background**: 60-80% (depends on kinematic region)
- **ggF Signal**: 80-95% (higher efficiency due to signal characteristics)
- **VBF Signal**: 70-90% (moderate efficiency)
- **Collision Data**: 65-85% (real trigger performance)

**Key Performance Metrics:**
- **AUC Score**: >0.85 for well-trained models
- **Calibration ECE**: <0.05 for well-calibrated probabilities
- **Validation Agreement**: <5% difference between predicted and measured efficiency

## 🔧 Advanced Usage

### Custom Analysis Configuration

Modify the notebook configuration for specialized analyses:

```python
# Custom trigger selection
signal_triggers = ["HLT_PFHT1050", "HLT_PFJet500"]  # Subset of triggers
reference_trigger = "HLT_AK8PFJet260"

# Custom variable selection  
analysis_variables = ["HighestPt", "HT", "MET_pt"]  # Reduced variable set

# Custom sample selection
samples = ["QCD", "DATA"]  # Only background and data
```

### Integration with Existing Analysis

```python
# Import trigger efficiency functions
from library.trigger_efficiency_ML import *

# Load your RDataFrame
df = ROOT.RDataFrame("tree", "your_file.root")

# Apply trigger efficiency correction
efficiency_weights = predict_trigger_efficiency(df, model_path)
corrected_df = df.Define("trigger_weight", efficiency_weights)
```

### Batch Processing

```python
# Process multiple data versions
versions = ["v1", "v2", "v3"]
for version in versions:
    run_analysis(version=version, save_results=True)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Data File Not Found
```
Error: RDataFrame unable to read file
```
**Solution:**
- Check file paths in the configuration
- Verify data version exists in `data/processed/`
- Ensure ROOT files are accessible

#### 2. Missing Libraries
```
ImportError: No module named 'xgboost'
```
**Solution:**
- Install missing packages: `pip install xgboost lightgbm catboost`
- Use CERN CMS environment where possible
- Check Python environment activation

#### 3. Memory Issues
```
Memory allocation error during training
```
**Solution:**
- Reduce dataset size for testing: `df = df.Range(100000)`
- Use fewer features: modify `analysis_variables` list
- Increase system memory or use batch processing

#### 4. ROOT/Python Integration
```
AttributeError: module 'ROOT' has no attribute 'RDataFrame'
```
**Solution:**
- Ensure compatible ROOT version (≥6.24)
- Check ROOT Python bindings: `import ROOT; print(ROOT.__version__)`
- Re-source CMS environment

#### 5. Poor Model Performance
```
AUC Score < 0.6, Poor efficiency prediction
```
**Solution:**
- Check data quality and preprocessing
- Verify trigger branch consistency
- Increase training data size
- Tune hyperparameters in `model_selection/`

### Performance Optimization

#### For Large Datasets
```python
# Use data sampling for faster development
df_sample = df.Range(50000)  # Use 50k events for testing

# Enable ROOT parallelization
ROOT.ROOT.EnableImplicitMT(4)  # Use 4 threads
```

#### For Production Analysis
```python
# Save intermediate results
df.Snapshot("processed_tree", "intermediate.root")

# Use efficient data formats
df.AsNumpy()  # Convert to NumPy for ML libraries
```

### Getting Help

1. **Check existing documentation** in component README files
2. **Review example scripts** in each subdirectory  
3. **Validate data files** using ROOT TBrowser or similar tools
4. **Test with reduced datasets** to isolate issues

## 🤝 Contributing

### Code Structure

Follow these conventions when contributing:

- **Functions**: Use descriptive names and comprehensive docstrings
- **Variables**: Follow physics naming conventions (e.g., `pt`, `eta`, `phi`)
- **Files**: Include version/date information in output filenames
- **Documentation**: Update README files for significant changes

### Adding New Features

1. **New ML Models**: Add to `model_selection/gbdt_comparison_framework.py`
2. **New Variables**: Update variable lists and plotting functions
3. **New Triggers**: Modify trigger configuration in main notebook
4. **New Samples**: Add sample handling in data loading functions

### Testing Changes

```bash
# Test with minimal dataset
python test_with_small_data.py

# Validate against reference results  
python validate_against_baseline.py

# Check calibration quality
python calibration/test_gradientboosting_integration.py
```

## 📚 References

### Physics Background
- CMS Collaboration, "Search for Higgs boson pair production..."  
- HH→bbττ analysis documentation
- CMS trigger system documentation

### Machine Learning Methods
- **Gradient Boosting**: Friedman, J.H. (2001). "Greedy function approximation: A gradient boosting machine"
- **Probability Calibration**: Platt, J. (1999). "Probabilistic outputs for support vector machines"
- **Model Selection**: Bergstra, J. & Bengio, Y. (2012). "Random search for hyper-parameter optimization"

### Software References
- **ROOT**: https://root.cern/
- **XGBoost**: https://xgboost.readthedocs.io/
- **scikit-learn**: https://scikit-learn.org/
- **CMS Open Data**: http://opendata.cern.ch/

---

**Project Status**: Active development for CMS HH→bbττ trigger efficiency analysis

**Last Updated**: September 2025

**Contact**: trantuankha643@gmail.com

**License**: Follow CMS collaboration guidelines for data and code sharing
