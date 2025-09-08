#!/usr/bin/env python3

import os, time
import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class CutflowAnalysis:
    """
    A class to perform cutflow analysis on particle physics data.
    Tracks event counts and efficiency at each filtering step.
    """
    
    def __init__(self, sample_name="Unknown"):
        self.sample_name = sample_name
        self.cutflow_data = []
        self.initial_count = 0
        
    def add_cut(self, cut_name, cut_expression, dataframe):
        """
        Add a cut to the cutflow analysis.
        
        Args:
            cut_name (str): Human-readable name for the cut
            cut_expression (str): ROOT filter expression
            dataframe: ROOT RDataFrame after applying the cut
        """
        count = dataframe.Count().GetValue()
        
        if len(self.cutflow_data) == 0:
            # First cut - this is our baseline
            self.initial_count = count
            efficiency_abs = 100.0
            efficiency_rel = 100.0
        else:
            # Calculate efficiencies
            efficiency_abs = (count / self.initial_count) * 100.0 if self.initial_count > 0 else 0.0
            efficiency_rel = (count / self.cutflow_data[-1]['count']) * 100.0 if self.cutflow_data[-1]['count'] > 0 else 0.0
        
        cut_info = {
            'step': len(self.cutflow_data) + 1,
            'cut_name': cut_name,
            'cut_expression': cut_expression,
            'count': count,
            'efficiency_absolute': efficiency_abs,
            'efficiency_relative': efficiency_rel
        }
        
        self.cutflow_data.append(cut_info)
        
        print(f"Cut {cut_info['step']}: {cut_name}")
        print(f"  Events: {count:,}")
        print(f"  Absolute efficiency: {efficiency_abs:.2f}%")
        print(f"  Relative efficiency: {efficiency_rel:.2f}%")
        print("-" * 50)
        
        return dataframe
    
    def print_summary(self):
        """Print a formatted summary table of the cutflow."""
        headers = ["Step", "Cut Name", "Events", "Abs. Eff. (%)", "Rel. Eff. (%)"]
        table_data = []
        
        for cut in self.cutflow_data:
            table_data.append([
                cut['step'],
                cut['cut_name'],
                f"{cut['count']:,}",
                f"{cut['efficiency_absolute']:.2f}",
                f"{cut['efficiency_relative']:.2f}"
            ])
        
        print(f"\n=== CUTFLOW SUMMARY FOR {self.sample_name.upper()} ===")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
    def save_to_csv(self, filename, save_path=None):
        """Save cutflow data to CSV file."""
        df = pd.DataFrame(self.cutflow_data)
        
        if save_path:
            save_path = f"cutflow_results/{save_path}"
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

        savefile = f"{save_path}/{filename}"
        df.to_csv(savefile, index=False)
        print(f"Cutflow data saved to {savefile}")
        
    def plot_cutflow(self, save_path=None):
        """Create visualization of the cutflow."""
        steps = [cut['step'] for cut in self.cutflow_data]
        counts = [cut['count'] for cut in self.cutflow_data]
        cut_names = [cut['cut_name'] for cut in self.cutflow_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Event counts
        ax1.plot(steps, counts, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Cut Step')
        ax1.set_ylabel('Number of Events')
        ax1.set_title(f'Cutflow Analysis - {self.sample_name}')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add text annotations
        for i, (step, count, name) in enumerate(zip(steps, counts, cut_names)):
            if i % 2 == 0:  # Alternate positions to avoid overlap
                ax1.annotate(f'{count:,}', (step, count), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, ha='left')
        
        # Plot 2: Relative efficiency
        rel_effs = [cut['efficiency_relative'] for cut in self.cutflow_data]
        ax2.bar(steps, rel_effs, alpha=0.7, color='green')
        ax2.set_xlabel('Cut Step')
        ax2.set_ylabel('Relative Efficiency (%)')
        ax2.set_title('Relative Efficiency at Each Cut')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        # Add cut names as x-axis labels
        ax2.set_xticks(steps)
        ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                            for name in cut_names], rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            path = f"cutflow_results/{save_path}"
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cutflow plot saved to {save_path}")
        
        plt.show()


def analyze_fatjet_particlenet_cutflow(dataframe, sample_name="Sample"):
    """
    Example cutflow analysis for FatJet ParticleNet filtering.
    
    This function demonstrates how to implement the cutflow technique
    for your specific filtering example.
    """
    cutflow = CutflowAnalysis(sample_name)
    
    # Step 1: Initial selection - at least 2 fat jets
    df = dataframe.Filter("nFatJet >= 2", "At least 2 FatJets")
    cutflow.add_cut("Initial: nFatJet >= 2", "nFatJet >= 2", df)
    
    # Step 2: Basic kinematic cuts
    df = df.Filter("FatJet_pt[0] > 250", "Leading jet pT > 250 GeV")
    cutflow.add_cut("Leading jet pT > 250 GeV", "FatJet_pt[0] > 250", df)
    
    # Step 3: Mass requirement
    df = df.Filter("FatJet_mass[0] > 30", "Leading jet mass > 30 GeV")
    cutflow.add_cut("Leading jet mass > 30 GeV", "FatJet_mass[0] > 30", df)
    
    # Step 4: Your specific ParticleNet filter - broken down into parts
    # First, let's see XttVsQCD requirement
    df_xtt = df.Filter("Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0", "At least one jet with XttVsQCD > 0.1")
    cutflow.add_cut("XttVsQCD > 0.1", "Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0", df_xtt)
    
    # Step 5: XtmVsQCD requirement
    df_xtm = df_xtt.Filter("Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0", "At least one jet with XtmVsQCD > 0.1")
    cutflow.add_cut("XtmVsQCD > 0.1", "Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0", df_xtm)
    
    # Step 6: XteVsQCD requirement
    df_xte = df_xtm.Filter("Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0", "At least one jet with XteVsQCD > 0.1")
    cutflow.add_cut("XteVsQCD > 0.1", "Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0", df_xte)
    
    # Step 7: Combined ParticleNet + mass requirement (your original complex filter)
    complex_filter = ("Sum((FatJet_particleNet_XttVsQCD > 0.1 || FatJet_particleNet_XtmVsQCD > 0.1 || "
                     "FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 30)) > 0")
    df_final = df.Filter(complex_filter, "Combined ParticleNet + mass filter")
    cutflow.add_cut("Combined ParticleNet + mass", complex_filter, df_final)
    
    # Print summary and create visualizations
    cutflow.print_summary()
    cutflow.plot_cutflow(f"cutflow_{sample_name}.png")
    cutflow.save_to_csv(f"cutflow_{sample_name}.csv")
    
    return df_final, cutflow


def compare_cutflows(cutflow_list, save_path=None):
    """
    Compare multiple cutflow analyses side by side.
    
    Args:
        cutflow_list: List of CutflowAnalysis objects
        save_path: Optional path to save comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot event counts
    for i, cutflow in enumerate(cutflow_list):
        steps = [cut['step'] for cut in cutflow.cutflow_data]
        counts = [cut['count'] for cut in cutflow.cutflow_data]
        color = colors[i % len(colors)]
        
        ax1.plot(steps, counts, 'o-', color=color, linewidth=2, 
                label=cutflow.sample_name, markersize=6)
    
    ax1.set_xlabel('Cut Step')
    ax1.set_ylabel('Number of Events')
    ax1.set_title('Cutflow Comparison - Event Counts')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot absolute efficiencies
    for i, cutflow in enumerate(cutflow_list):
        steps = [cut['step'] for cut in cutflow.cutflow_data]
        abs_effs = [cut['efficiency_absolute'] for cut in cutflow.cutflow_data]
        color = colors[i % len(colors)]
        
        ax2.plot(steps, abs_effs, 'o-', color=color, linewidth=2, 
                label=cutflow.sample_name, markersize=6)
    
    ax2.set_xlabel('Cut Step')
    ax2.set_ylabel('Absolute Efficiency (%)')
    ax2.set_title('Cutflow Comparison - Absolute Efficiency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


# Example usage function
def main():
    """
    Example of how to use the cutflow analysis with your data.
    """
    # Load your data (replace with actual file path)
    # df = ROOT.RDataFrame("Events", "your_data_file.root")
    
    # For demonstration, let's create mock analysis
    print("=== CUTFLOW ANALYSIS EXAMPLE ===")
    print("This example shows how to implement cutflow analysis")
    print("Replace the mock dataframe with your actual ROOT dataframe")
    
    # Example for multiple samples
    sample_names = ["QCD", "ggF", "VBF", "DATA"]
    cutflows = []
    
    for sample in sample_names:
        print(f"\nAnalyzing {sample} sample...")
        # In real usage, you would load different files here
        # df = ROOT.RDataFrame("Events", f"{sample}_file.root")
        # final_df, cutflow = analyze_fatjet_particlenet_cutflow(df, sample)
        # cutflows.append(cutflow)
    
    # Compare all samples
    # compare_cutflows(cutflows, "cutflow_comparison.png")


if __name__ == "__main__":
    main()
