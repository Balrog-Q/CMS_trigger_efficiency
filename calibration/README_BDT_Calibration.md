# BDT Probability Calibration for CMS Trigger Efficiency

This system implements comprehensive probability calibration for Boosted Decision Tree (BDT) models used in CMS trigger efficiency analysis. The calibration is performed using only the QCD dataset, making it suitable for scenarios where you want to avoid signal contamination in the calibration process.

## Overview

**Problem Addressed**: Raw BDT scores often don't represent well-calibrated probabilities. This is crucial for trigger efficiency studies where you need reliable probability estimates.

**Solution**: This implementation provides:
- Training of multiple BDT models (XGBoost, LightGBM, CatBoost, sklearn GradientBoosting) on QCD data
- Probability calibration using isotonic regression and Platt scaling
- Comprehensive evaluation on QCD, ggF, and VBF datasets
- Production-ready calibrated models for trigger efficiency studies

## Key Features

### 🎯 **Calibration Methods**
- **Isotonic Regression**: Non-parametric method that can handle complex calibration curves
- **Platt Scaling**: Parametric sigmoid-based method for simpler cases
- **Cross-validation**: Prevents overfitting during calibration

### 📊 **Evaluation Metrics**
- **Expected Calibration Error (ECE)**: Overall calibration quality
- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Probability scoring rule
- **Log Loss**: Logarithmic loss for probability evaluation
- **Reliability-Resolution**: Decomposition of calibration quality

### 🔍 **Visualization**
- Calibration curves (reliability diagrams)
- ROC curve comparisons
- Probability distribution histograms
- Calibration metrics heatmaps

## Installation

### Prerequisites

Ensure you have the following Python packages installed:

```bash
# Core ML libraries
pip install scikit-learn xgboost lightgbm catboost

# Data manipulation and visualization
pip install pandas numpy matplotlib seaborn

# ROOT (for CMS data handling)
# Follow ROOT installation instructions for your system
```

### Files Required

Make sure these files are in your project directory:
- `probability_calibration.py` - Core calibration module
- `bdt_calibration_workflow.py` - Complete workflow implementation
- `run_bdt_calibration.py` - Simple execution script
- `bdt_calibration_example.py` - Usage examples
- `library/trigger_efficiency_ML.py` - Trigger efficiency measurement

## Quick Start

### 1. Basic Usage

Run the complete calibration workflow with default settings:

```bash
python run_bdt_calibration.py
```

### 2. Custom Configuration

Specify data version and random state:

```bash
python run_bdt_calibration.py --data_version testing --random_state 42
```

### 3. List Available Data

Check what processed data is available:

```bash
python run_bdt_calibration.py --list_versions
```

## Data Requirements

### Expected Data Structure

```
data/processed/
├── testing/
│   ├── testing-NewQCD.root
│   ├── testing-NewggF.root
│   └── testing-NewVBF.root
├── azura/
│   ├── azura-NewQCD.root
│   ├── azura-NewggF.root
│   └── azura-NewVBF.root
└── ... (other versions)
```

### Required Features

The ROOT files should contain these branches:
- `HighestPt`, `HT`, `MET_pt`, `mHH`
- `HighestMass`, `SecondHighestPt`, `SecondHighestMass`
- `FatHT`, `MET_FatJet`, `mHHwithMET`
- `HighestEta`, `SecondHighestEta`, `DeltaEta`, `DeltaPhi`
- `Combo` (target variable)
- `HLT_AK8PFJet260` (reference trigger)

## Workflow Details

### 1. Data Loading and Preparation
```python
# Load ROOT files and convert to pandas
qcd_data = load_qcd_data()
ggf_data = load_ggf_data()
vbf_data = load_vbf_data()

# Split QCD data: 50% training, 30% calibration, 20% testing
X_train, X_cal, X_test = split_qcd_data(qcd_data)
```

### 2. Model Training
```python
# Train multiple BDT models on QCD training data
models = {
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(), 
    "CatBoost": CatBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
```

### 3. Calibration
```python
# Apply calibration using QCD calibration data
calibrator = BDTCalibrator(method='both', cv_folds=5)
calibrator.fit(model, X_cal, y_cal)
```

### 4. Evaluation
```python
# Evaluate on all datasets
for dataset_name, (X_test, y_test) in datasets.items():
    results = calibrator.evaluate_calibration(X_test, y_test)
```

## Output Files

After running the workflow, you'll find these files in the results directory:

### Models and Calibrators
- `models/xgboost_model.pkl` - Trained XGBoost model
- `models/lightgbm_model.pkl` - Trained LightGBM model  
- `models/catboost_model.pkl` - Trained CatBoost model
- `models/gradientboosting_model.pkl` - Trained sklearn GradientBoosting model
- `calibrators/xgboost_calibrator.pkl` - XGBoost calibrator
- `calibrators/lightgbm_calibrator.pkl` - LightGBM calibrator
- `calibrators/catboost_calibrator.pkl` - CatBoost calibrator
- `calibrators/gradientboosting_calibrator.pkl` - sklearn GradientBoosting calibrator

### Analysis Results
- `evaluation_results.pkl` - Complete evaluation results
- `calibration_analysis_report.md` - Comprehensive analysis report

### Visualizations
- `calibration_curve_{model}_{dataset}.png` - Calibration curves
- `probability_hist_{model}_{dataset}.png` - Probability distributions
- `roc_comparison_all.png` - ROC curve comparisons
- `calibration_metrics_summary.png` - Metrics heatmap

## Usage Examples

### Loading Pre-trained Models

```python
import pickle
from probability_calibration import BDTCalibrator

# Load model and calibrator
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('calibrators/xgboost_calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)
```

### Making Calibrated Predictions

```python
# Get calibrated probabilities
calibrated_probs = calibrator.predict_proba(X_new, method='isotonic')

# Calculate trigger efficiency
trigger_efficiency = (calibrated_probs > threshold).mean()
```

### Integration into Existing Workflow

```python
def predict_with_calibration(model, calibrator, X_features):
    """Make calibrated predictions for trigger efficiency."""
    return calibrator.predict_proba(X_features, method='isotonic')

# Usage in your analysis
efficiencies = []
for pt_bin in pt_bins:
    X_bin = features_in_pt_bin(pt_bin)
    cal_probs = predict_with_calibration(model, calibrator, X_bin)
    eff = (cal_probs > 0.5).mean()
    efficiencies.append(eff)
```

## Calibration Quality Interpretation

### Expected Calibration Error (ECE)
- **< 0.05**: Excellent calibration
- **0.05-0.10**: Good calibration
- **0.10-0.15**: Acceptable calibration
- **> 0.15**: Poor calibration, consider re-calibration

### Calibration Curves
- **Perfect calibration**: Points lie on diagonal line
- **Under-confident**: Curve below diagonal
- **Over-confident**: Curve above diagonal

## Advanced Usage

### Custom Calibration

```python
from probability_calibration import BDTCalibrator

# Initialize with custom settings
calibrator = BDTCalibrator(
    method='isotonic',  # or 'sigmoid', 'both'
    cv_folds=10,       # more folds for better calibration
    random_state=42
)

# Fit on your calibration data
calibrator.fit(your_model, X_cal, y_cal)

# Evaluate with custom bins
results = calibrator.evaluate_calibration(
    X_test, y_test, 
    dataset_name="Custom",
    n_bins=20  # more bins for detailed analysis
)
```

### Batch Processing

```python
# Process multiple models
models_to_calibrate = ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']
calibrators = {}

for model_name in models_to_calibrate:
    model = load_model(model_name)
    calibrator = BDTCalibrator(method='both')
    calibrator.fit(model, X_cal, y_cal)
    calibrators[model_name] = calibrator
```

## Troubleshooting

### Common Issues

1. **Missing data files**
   ```bash
   # Check available data versions
   python run_bdt_calibration.py --list_versions
   
   # Use correct version
   python run_bdt_calibration.py --data_version your_version
   ```

2. **Memory issues**
   - Reduce dataset size for initial testing
   - Use fewer cross-validation folds
   - Process one model at a time

3. **Poor calibration**
   - Increase calibration data size
   - Try different calibration methods
   - Check for data distribution shifts

4. **Import errors**
   ```bash
   # Install missing dependencies
   pip install xgboost lightgbm catboost scikit-learn
   ```

### Performance Optimization

```python
# For faster execution with large datasets
calibrator = BDTCalibrator(
    method='sigmoid',  # Faster than isotonic
    cv_folds=3,       # Fewer folds
    random_state=42
)

# Use fewer bins for evaluation
results = calibrator.evaluate_calibration(
    X_test, y_test, n_bins=5
)
```

## Best Practices

### 1. Data Quality
- Ensure consistent preprocessing between training and inference
- Check for data leakage between calibration and test sets
- Monitor data distribution shifts over time

### 2. Model Selection
- Choose models based on ECE, not just AUC
- Consider computational requirements for production
- Validate on multiple datasets (QCD, ggF, VBF)

### 3. Calibration Maintenance
- Re-calibrate periodically with new data
- Monitor calibration quality in production
- Keep separate calibration datasets

### 4. Production Deployment
```python
# Example production wrapper
class CalibratedTriggerPredictor:
    def __init__(self, model_path, calibrator_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.calibrator = pickle.load(open(calibrator_path, 'rb'))
    
    def predict_efficiency(self, features, threshold=0.5):
        probs = self.calibrator.predict_proba(features, method='isotonic')
        return (probs > threshold).mean()
    
    def predict_probabilities(self, features):
        return self.calibrator.predict_proba(features, method='isotonic')
```

## Citation and References

If you use this calibration system in your analysis, please consider citing:

- **Platt Scaling**: Platt, J. (1999). Probabilistic outputs for support vector machines
- **Isotonic Regression**: Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates
- **Calibration Evaluation**: Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning

## Support and Contributing

### Getting Help
1. Check the troubleshooting section above
2. Review the example scripts in `bdt_calibration_example.py`
3. Examine the comprehensive analysis report generated by the workflow

### Contributing
- Report bugs or issues with specific error messages
- Suggest improvements for calibration methods
- Share results from different physics analyses

## License

This calibration system is provided as-is for CMS trigger efficiency analysis. Please follow your collaboration's guidelines for code sharing and publication.
