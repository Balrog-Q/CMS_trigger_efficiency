# Cutflow Analysis for CMS Particle Physics Data

This set of tools implements cutflow analysis for your trigger efficiency study, specifically designed to analyze the complex ParticleNet filters from processing MC sample step by step.

## What is Cutflow Analysis?

Cutflow analysis tracks the number of events remaining after each filtering step in the analysis. It helps to understand:
- How many events we lose at each step
- Which cuts are most restrictive
- The efficiency of the selection criteria
- How different samples behave under the same cuts

## Files Overview
```
cutflow/
├── cutflow_analysis.py      # Core cutflow analysis class and utilities
├── cutflow_example.py       # Advanced example with full CMS HH analysis
├── run_cutflow.py           # Simple script can run immediately on the data
├── README_cutflow.md        # This documentation
└── cutflow_results/         # Folder to store all the results
	└── ...                  # Subfolder for specific run

```

## Quick Start

### Step 1: Install Required Packages
```bash
pip install matplotlib tabulate
```

### Step 2: Update File Paths
Edit the file paths in `run_cutflow.py` to point to the actual ROOT files:

```python
example_files = {
    "QCD": "path/to/your/QCD_file.root",
    "ggF": "path/to/your/ggF_file.root", 
    "DATA": "path/to/your/DATA_file.root"
}
```

### Step 3: Run the Analysis
```bash
python run_cutflow.py
```

## Understanding Complex Filter
### Example 1
The original complex filter was:
```cpp
Sum((FatJet_particleNet_XttVsQCD > 0.1 || FatJet_particleNet_XtmVsQCD > 0.1 || FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 30)) > 0
```

The cutflow analysis breaks this down to show you:
1. How many events pass each individual ParticleNet discriminator
2. How many events pass the mass requirement
3. How many events pass the combined filter
4. The efficiency at each step

## Output Files

After running the analysis, you'll get:

1. **Console Output**: Real-time progress and event counts
2. **`cutflow_[sample].png`**: Visualization plots for each sample
3. **`cutflow_[sample].csv`**: Detailed cutflow data in CSV format
4. **`cutflow_comparison.png`**: Side-by-side comparison of all samples

## Example Output

```
=== Cutflow Analysis for QCD ===
Cut 1: All events
  Events: 1,500,000
  Absolute efficiency: 100.00%
  Relative efficiency: 100.00%
--------------------------------------------------
Cut 2: nFatJet >= 2
  Events: 800,000
  Absolute efficiency: 53.33%
  Relative efficiency: 53.33%
--------------------------------------------------
Cut 3: FatJet_pt[0] > 250 GeV
  Events: 400,000
  Absolute efficiency: 26.67%
  Relative efficiency: 50.00%
--------------------------------------------------
```

## Customization

### Adding Your Own Cuts

To add custom cuts to the analysis, modify the `simple_cutflow_example` function:

```python
# Add your custom cut
df = df.Filter("your_custom_condition", "Custom cut description")
cutflow.add_cut("Custom Cut Name", "your_custom_condition", df)
```

### Analyzing Different Variables

You can analyze different particle physics variables by modifying the filter expressions:

```python
# Examples for different analyses
df = df.Filter("Sum(FatJet_particleNet_XbbVsQCD > 0.8) > 0", "b-tagging requirement")
df = df.Filter("MET_pt > 100", "Missing ET requirement")
df = df.Filter("abs(FatJet_eta[0] - FatJet_eta[1]) > 1.0", "Delta eta requirement")
```

## Advanced Usage

### Detailed Analysis with `cutflow_example.py`

For a more comprehensive analysis that includes:
- Trigger efficiency studies
- Invariant mass requirements
- Multiple discriminator comparisons

Run:
```python
from cutflow_example import cms_higgs_analysis_cutflow
cms_higgs_analysis_cutflow()
```

### Comparing Multiple Samples

```python
from cutflow_analysis import CutflowAnalysis, compare_cutflows

cutflows = []
for sample in ["QCD", "ggF", "VBF", "DATA"]:
    # ... perform analysis for each sample
    cutflows.append(cutflow_result)

compare_cutflows(cutflows, "my_comparison.png")
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Update the file paths in the script
2. **Missing Branches**: Some ROOT files may not have all the branches (e.g., trigger branches in MC)
3. **Memory Issues**: For very large files, consider processing smaller chunks

### Error Messages

- `Error: RDataFrame unable to read file` - Check file path and ROOT file integrity
- `Missing required package` - Install matplotlib and tabulate
- `Branch not found` - Some variables may not exist in all samples

## Understanding the Results

### Efficiency Types

- **Absolute Efficiency**: Events remaining / Initial events
- **Relative Efficiency**: Events remaining / Events from previous step

### Key Insights from Cutflow

1. **Most Restrictive Cuts**: Steps with lowest relative efficiency
2. **Sample Differences**: Comparing efficiency across QCD, signal, and data
3. **Filter Optimization**: Identifying redundant or ineffective cuts

## Integration with Your Existing Code

This cutflow analysis can be integrated into your existing trigger efficiency notebooks. Simply replace your current filtering steps with cutflow-tracked versions:

```python
# Instead of:
df = df.Filter("nFatJet >= 2")

# Use:
cutflow = CutflowAnalysis("MySample")
df = df.Filter("nFatJet >= 2")
cutflow.add_cut("nFatJet >= 2", "nFatJet >= 2", df)
```

This way, you maintain your analysis while gaining insight into each step's efficiency.
