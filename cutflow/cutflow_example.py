#!/usr/bin/env python3

import ROOT
from cutflow_analysis import CutflowAnalysis, compare_cutflows

#  specific filtering example implemented with cutflow analysis
def cms_higgs_analysis_cutflow():
    """
    Example implementation of cutflow analysis for CMS HH->bbtautau analysis.
    This demonstrates how to break down complex filters into individual steps.
    """
    
    # Initialize ROOT (you may need to adjust paths to  actual data files)
    print("=== CMS HH->bbtautau Cutflow Analysis ===\n")
    
    # Example file paths - replace with the actual files
    file_paths = {
        "QCD": "data/HHbbtautau-v2/QCD_combined.root",
        "ggF": "data/HHbbtautau-v2/GluGluToHHTo2B2Tau_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_2jets_tight.root",
        "VBF": "data/HHbbtautau-v2/VBF_HHTo2B2Tau_CV_1_C2V_0_C3_1_TuneCP5_13TeV-powheg-pythia8_2jets_tight.root",
        "DATA": "data/HHbbtautau-v2/JetHT_Run2018_2jets_tight_v3.root"
    }
    
    all_cutflows = []
    
    for sample_name, file_path in file_paths.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {sample_name} sample")
        print(f"File: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Load the data
            df = ROOT.RDataFrame("Events", file_path)
            
            # Perform cutflow analysis
            final_df, cutflow = analyze_cms_hh_cutflow(df, sample_name)
            all_cutflows.append(cutflow)
            
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            print("Continuing with next sample...")
            continue
    
    # Compare all samples
    if len(all_cutflows) > 1:
        print("\n" + "="*80)
        print("COMPARING ALL SAMPLES")
        print("="*80)
        compare_cutflows(all_cutflows, "cms_hh_cutflow_comparison.png")


def analyze_cms_hh_cutflow(dataframe, sample_name):
    """
    Detailed cutflow analysis for CMS HH->bb tau tau analysis.
    This breaks down  complex filter into understandable steps.
    """
    cutflow = CutflowAnalysis(sample_name)
    
    # Step 1: Initial event count
    cutflow.add_cut("Initial events", "1", dataframe)
    
    # Step 2: At least 2 fat jets
    df = dataframe.Filter("nFatJet >= 2", "At least 2 FatJets")
    cutflow.add_cut("nFatJet >= 2", "nFatJet >= 2", df)
    
    # Step 3: Leading fat jet pT requirement
    df = df.Filter("FatJet_pt[0] > 250", "Leading FatJet pT > 250 GeV")
    cutflow.add_cut("FatJet_pt[0] > 250 GeV", "FatJet_pt[0] > 250", df)
    
    # Step 4: Sub-leading fat jet pT requirement  
    df = df.Filter("FatJet_pt[1] > 200", "Sub-leading FatJet pT > 200 GeV")
    cutflow.add_cut("FatJet_pt[1] > 200 GeV", "FatJet_pt[1] > 200", df)
    
    # Step 5: Eta requirements
    df = df.Filter("abs(FatJet_eta[0]) < 2.4 && abs(FatJet_eta[1]) < 2.4", "Both FatJets |eta| < 2.4")
    cutflow.add_cut("|eta| < 2.4", "abs(FatJet_eta[0]) < 2.4 && abs(FatJet_eta[1]) < 2.4", df)
    
    # Step 6: Basic mass requirements
    df = df.Filter("FatJet_mass[0] > 30 && FatJet_mass[1] > 30", "Both FatJets mass > 30 GeV")
    cutflow.add_cut("Mass > 30 GeV", "FatJet_mass[0] > 30 && FatJet_mass[1] > 30", df)
    
    # Step 7: At least 2 regular jets for HT calculation
    df = df.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")
    df = df.Filter("SelectedJets_pt.size() >= 2", "At least 2 selected jets")
    cutflow.add_cut(">=2 jets (pT>30, |eta|<2.4)", "SelectedJets_pt.size() >= 2", df)
    
    # Step 8: Now let's break down ParticleNet requirements
    # First, check individual discriminators
    
    # XttVsQCD (tau vs QCD discrimination)
    df_xtt = df.Filter("Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0", "XttVsQCD > 0.1")
    count_xtt = df_xtt.Count().GetValue()
    print(f"  Events passing XttVsQCD > 0.1: {count_xtt:,}")
    
    # XtmVsQCD (tau mass vs QCD discrimination)  
    df_xtm = df.Filter("Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0", "XtmVsQCD > 0.1")
    count_xtm = df_xtm.Count().GetValue()
    print(f"  Events passing XtmVsQCD > 0.1: {count_xtm:,}")
    
    # XteVsQCD (tau energy vs QCD discrimination)
    df_xte = df.Filter("Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0", "XteVsQCD > 0.1")
    count_xte = df_xte.Count().GetValue()
    print(f"  Events passing XteVsQCD > 0.1: {count_xte:,}")
    
    # Step 9: The original combined ParticleNet filter
    particlenet_filter = ("Sum((FatJet_particleNet_XttVsQCD > 0.1 || FatJet_particleNet_XtmVsQCD > 0.1 || "
                         "FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 30)) > 0")
    df = df.Filter(particlenet_filter, "Combined ParticleNet filter")
    cutflow.add_cut("ParticleNet OR (Xtt|Xtm|Xte) + mass", particlenet_filter, df)
    
    # Step 10: Additional kinematic requirements (based on  existing code)
    df = df.Define("HT", "Sum(SelectedJets_pt)")
    df = df.Filter("HT > 500", "HT > 500 GeV")
    cutflow.add_cut("HT > 500 GeV", "HT > 500", df)
    
    # Step 11: Invariant mass requirement
    df = df.Define("mHH", "InvariantMass(Take(FatJet_pt, 2), Take(FatJet_eta, 2), Take(FatJet_phi, 2), Take(FatJet_mass, 2))")
    df = df.Filter("mHH > 600 && mHH < 3000", "600 < mHH < 3000 GeV")
    cutflow.add_cut("600 < mHH < 3000 GeV", "mHH > 600 && mHH < 3000", df)
    
    # Step 12: Final trigger requirements (sample-dependent)
    if sample_name == "DATA":
        df = df.Filter("HLT_AK8PFJet260", "Reference trigger")
        cutflow.add_cut("HLT_AK8PFJet260", "HLT_AK8PFJet260", df)
    else:
        # For MC, apply trigger combination if available
        trigger_combo = ("HLT_AK8PFHT800_TrimMass50 || HLT_AK8PFJet400_TrimMass30 || "
                        "HLT_AK8PFJet500 || HLT_PFJet500 || HLT_PFHT1050")
        try:
            df = df.Filter(trigger_combo, "Signal triggers OR")
            cutflow.add_cut("Signal triggers OR", trigger_combo, df)
        except:
            print(f"  Warning: Some triggers not available in {sample_name}")
    
    # Print detailed summary
    cutflow.print_summary()
    
    # Create visualizations
    cutflow.plot_cutflow(f"cutflow_{sample_name}.png")
    cutflow.save_to_csv(f"cutflow_{sample_name}.csv", sample_name)
    
    return df, cutflow


def quick_particlenet_breakdown(dataframe, sample_name):
    """
    Quick analysis to understand specific ParticleNet filter behavior.
    This addresses exact question about the complex filter.
    """
    print(f"\n=== ParticleNet Filter Breakdown for {sample_name} ===")
    
    # Start with basic selections
    df = dataframe.Filter("nFatJet >= 2")
    df = df.Filter("FatJet_pt[0] > 250")
    initial_count = df.Count().GetValue()
    print(f"After basic selections: {initial_count:,} events")
    
    # Test individual components of filter
    filters_to_test = [
        ("XttVsQCD > 0.1", "Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0"),
        ("XtmVsQCD > 0.1", "Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0"), 
        ("XteVsQCD > 0.1", "Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0"),
        ("Mass > 30", "Sum(FatJet_mass > 30) > 0"),
        (" combined filter", "Sum((FatJet_particleNet_XttVsQCD > 0.1 || FatJet_particleNet_XtmVsQCD > 0.1 || FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 30)) > 0")
    ]
    
    print(f"\n{'Filter':<25} {'Events':<12} {'Efficiency (%)':<15}")
    print("-" * 55)
    
    for filter_name, filter_expr in filters_to_test:
        try:
            df_filtered = df.Filter(filter_expr)
            count = df_filtered.Count().GetValue()
            efficiency = (count / initial_count) * 100.0 if initial_count > 0 else 0.0
            print(f"{filter_name:<25} {count:<12,} {efficiency:<15.2f}")
        except Exception as e:
            print(f"{filter_name:<25} {'ERROR':<12} {str(e)[:15]:<15}")


if __name__ == "__main__":
    # Run the full analysis
    cms_higgs_analysis_cutflow()
    
    # If you want to test just the ParticleNet breakdown on a single file:
    # df = ROOT.RDataFrame("Events", "_file.root")
    # quick_particlenet_breakdown(df, "TestSample")
