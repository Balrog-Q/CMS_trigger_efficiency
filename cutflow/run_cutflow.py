#!/usr/bin/env python3

import ROOT
import sys
import os

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cutflow_analysis import CutflowAnalysis

def simple_cutflow_example(file_path, sample_name="Sample"):
    """
    Simple example that can run immediately on the data.
    This directly addresses the specific ParticleNet filter question.
    """
    
    print(f"=== Cutflow Analysis for {sample_name} ===")
    print(f"File: {file_path}")
    print("=" * 60)
    
    try:
        # Load data
        df = ROOT.RDataFrame("Events", file_path)
        
        # Initialize cutflow tracker
        cutflow = CutflowAnalysis(sample_name)
        
        # Specific analysis steps with cutflow tracking
        
        # Step 1: Start with all events
        cutflow.add_cut("All events", "1", df)
        
        # Step 2: At least 2 fat jets 
        df = df.Filter("nFatJet >= 2", "nFatJet >= 2")
        cutflow.add_cut("nFatJet >= 2", "nFatJet >= 2", df)
        
        # Step 3: Leading jet pT 
        df = df.Filter("FatJet_pt[0] > 250", "FatJet_pt[0] > 250")
        cutflow.add_cut("FatJet_pt[0] > 250 GeV", "FatJet_pt[0] > 250", df)

        # Step 4: Brak down complex ParticleNet filter step by step

        # First, just mass requirement

        df_mass = df.Filter("Sum(FatJet_mass > 50) > 0", "Mass requirement")
        count_mass = df_mass.Count().GetValue()
        print(f"\nBreakdown of complex filter:")
        print(f"  Events with FatJet_mass > 50: {count_mass:,}")

        # Then particleNet XbbVsQCD
        df_xbb = df.Filter("Sum(FatJet_particleNet_XbbVsQCD > 0.1) > 0", "XbbVsQCD")
        count_xbb = df_xbb.Count().GetValue()
        print(f"  Events with XbbVsQCD > 0.1: {count_xbb:,}")

        # Now exact complex filter
        filter = ("Sum((FatJet_particleNet_XbbVsQCD > 0.1) && (FatJet_mass > 50)) > 0")

        df = df.Filter(filter, "Complex ParticleNet filter 1")
        cutflow.add_cut("ParticleNet + Mass Filter step1", filter, df)
        
        # Step 5: Break down complex ParticleNet filter step by step
        
        # First, just mass requirement
        df_mass = df.Filter("Sum(FatJet_mass > 50) > 0", "Mass requirement")
        count_mass = df_mass.Count().GetValue()
        print(f"\nBreakdown of complex filter:")
        print(f"  Events with FatJet_mass > 50: {count_mass:,}")

        # ParticleNet XbbVsQCD
        df_xbb = df.Filter("Sum(FatJet_particleNet_XbbVsQCD > 0.1) > 0", "XbbVsQCD")
        count_xbb = df_xbb.Count().GetValue()
        print(f"  Events with XbbVsQCD > 0.1: {count_xbb:,}")
        
        # Then ParticleNet XttVsQCD
        df_xtt = df.Filter("Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0", "XttVsQCD")  
        count_xtt = df_xtt.Count().GetValue()
        print(f"  Events with XttVsQCD > 0.1: {count_xtt:,}")
        
        # ParticleNet XtmVsQCD  
        df_xtm = df.Filter("Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0", "XtmVsQCD")
        count_xtm = df_xtm.Count().GetValue()
        print(f"  Events with XtmVsQCD > 0.1: {count_xtm:,}")
        
        # ParticleNet XteVsQCD
        df_xte = df.Filter("Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0", "XteVsQCD")
        count_xte = df_xte.Count().GetValue()
        print(f"  Events with XteVsQCD > 0.1: {count_xte:,}")
        
        # Now exact complex filter
        filter = ("Sum((FatJet_particleNet_XttVsQCD > 0.1 || " 
                      "FatJet_particleNet_XbbVsQCD > 0.1 || "
                      "FatJet_particleNet_XtmVsQCD > 0.1 || "
                      "FatJet_particleNet_XteVsQCD > 0.1) && "
                      "(FatJet_mass > 50)) > 0")
        
        df = df.Filter(filter, "Complex ParticleNet filter 2")
        cutflow.add_cut("ParticleNet + Mass filter step2", filter, df)
        
        # Additional common cuts from analysis
        
        # Step 6: Jet selection for HT
        df = df.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")
        df = df.Filter("SelectedJets_pt.size() >= 2", "At least 2 jets")
        cutflow.add_cut(">=2 jets (pT>30, |eta|<2.4)", "SelectedJets_pt.size() >= 2", df)
        
        # Step 7: Fat jet selection  
        df = df.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")
        df = df.Filter("SelectedFatJets_pt.size() >= 2", "At least 2 fat jets in acceptance")
        cutflow.add_cut(">=2 FatJets |eta|<2.4", "SelectedFatJets_pt.size() >= 2", df)

        # Step 8: Signal Triggers
        signal_triggers = [
            "HLT_AK8PFHT800_TrimMass50",
            "HLT_AK8PFJet400_TrimMass30",
            "HLT_AK8PFJet500",
            "HLT_PFJet500",
            "HLT_PFHT1050",
            "HLT_PFHT500_PFMET100_PFMHT100_IDTight",
            "HLT_PFHT700_PFMET85_PFMHT85_IDTight",
            "HLT_PFHT800_PFMET75_PFMHT75_IDTight",
            "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg",
            "HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1"
        ]
        available_triggers = [t for t in signal_triggers if df.HasColumn(t)]
        for trigger in available_triggers:
            df_temp = df.Filter(trigger, trigger)
            count_temp = df_temp.Count().GetValue()
            print(f"  Events with {trigger}: {count_temp:,}")

        combo_expr = " or ".join(available_triggers)
        df = df.Define("Combo", combo_expr)
        df = df.Filter("Combo", "Signal Triggers")
        cutflow.add_cut("Signal Trigger", "Combo", df)

        # Step 9: Reference Triggers
        reference_trigger = "HLT_AK8PFJet260"
        df = df.Filter(reference_trigger, "Reference Trigger")
        cutflow.add_cut("Reference Trigger", reference_trigger, df)
        
        # Print the complete cutflow summary
        cutflow.print_summary()
        
        # Create visualizations
        cutflow.plot_cutflow(f"cutflow_{sample_name}.png", sample_name)
        cutflow.save_to_csv(f"cutflow/cutflow_{sample_name}.csv", sample_name)
        
        print(f"\nCutflow analysis complete!")
        print(f"- Summary table printed above")
        print(f"- Plot saved as: cutflow_{sample_name}.png")
        print(f"- Data saved as: cutflow_{sample_name}.csv")
        
        return df, cutflow
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def ggF_cutflow_example(file_path, sample_name="Sample"):
    """
    Simple example for ggF analysis
    """
    print(f"=== Cutflow Analysis for {sample_name} ===")
    print(f"File: {file_path}")
    print("=" * 60)

    try: 
        # Load the data
        df = ROOT.RDataFrame("Events", file_path)

        cutflow = CutflowAnalysis(sample_name)

        # Specific analysis steps with cutflow tracking

        # Step 1: Start with all events
        cutflow.add_cut("All events", "1", df)

        # Step 2: At least 2 fat jets
        df = df.Filter('nFatJet >= 2', "nFatJet >= 2")
        cutflow.add_cut("nFatJet >= 2", "nFatJet >= 2", df)

        # Step 3: Jet selection for HT
        df = df.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")
        df = df.Filter("SelectedJets_pt.size() >= 2", "At least 2 jets")
        cutflow.add_cut(">=2 jets (pT>30, |eta|<2.4)", "SelectedJets_pt.size() >= 2", df)

        # Step 4: Fat Jet selection
        df = df.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")
        df = df.Filter("SelectedFatJets_pt.size() >= 2", "At least 2 fat jets in acceptance")
        cutflow.add_cut(">=2 FatJets |eta|<2.4", "SelectedFatJets_pt.size() >= 2", df)

        # Step 5: Signal Triggers
        signal_triggers = [
            "HLT_AK8PFHT800_TrimMass50",
            "HLT_AK8PFJet400_TrimMass30",
            "HLT_AK8PFJet500",
            "HLT_PFJet500",
            "HLT_PFHT1050",
            "HLT_PFHT500_PFMET100_PFMHT100_IDTight",
            "HLT_PFHT700_PFMET85_PFMHT85_IDTight",
            "HLT_PFHT800_PFMET75_PFMHT75_IDTight",
            "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg",
            "HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1"
        ]
        available_triggers = [signal for signal in signal_triggers if df.HasColumn(signal)]
        print(f"List of all signal trigger used for ggF data:")
        for trigger in available_triggers:
            df_temp = df.Filter(trigger, trigger)
            count_temp = df_temp.Count().GetValue()
            print(f"  Events with {trigger}: {count_temp:,}")

        combo_expr = " or ".join(available_triggers)
        df = df.Define("Combo", combo_expr)
        df = df.Filter("Combo", "Signal Triggers")
        cutflow.add_cut("Signal Trigger", "Combo", df)

        # Step 6: Particle Net filtering
        print(f"\nBreakdown of complex filter:")

        # ParticleNet XbbVsQCD
        df_xbb = df.Filter("FatJet_particleNet_XbbVsQCD[0] > 0.9")
        count_xbb = df_xbb.Count().GetValue()
        print(f"  Events with FatJet_particleNet_XbbVsQCD[0] > 0.9: {count_xbb:,}")

        # ParticleNet XttVsQCD
        df_xtt = df.Filter("FatJet_particleNet_XttVsQCD[0] > 0.9")
        count_xtt = df_xtt.Count().GetValue()
        print(f"  Events with FatJet_particleNet_XttVsQCD[0] > 0.9: {count_xtt:,}")

        # Combination of ParticleNet XbbVsQCD and XttVsQCD
        df_combined = df.Filter("(FatJet_particleNet_XbbVsQCD[0] > 0.9 or FatJet_particleNet_XttVsQCD[0] > 0.9)")
        count_combined = df_combined.Count().GetValue()
        print(f"  -> Events with FatJet_particleNet_XbbVsQCD[0] > 0.9 or FatJet_particleNet_XttVsQCD[0] > 0.9: {count_combined:,}")

        # ParticleNet XbbVsQCD
        df_xbb = df.Filter("FatJet_particleNet_XbbVsQCD[1] > 0.9")
        count_xbb = df_xbb.Count().GetValue()
        print(f"  Events with FatJet_particleNet_XbbVsQCD[1] > 0.9: {count_xbb:,}")

        # ParticleNet XttVsQCD
        df_xtt = df.Filter("FatJet_particleNet_XttVsQCD[1] > 0.9")
        count_xtt = df_xtt.Count().GetValue()
        print(f"  Events with FatJet_particleNet_XttVsQCD[1] > 0.9: {count_xtt:,}")

        # Combination of ParticleNet XbbVsQCD and XttVsQCD
        df_combined = df.Filter("(FatJet_particleNet_XbbVsQCD[1] > 0.9 or FatJet_particleNet_XttVsQCD[1] > 0.9)")
        count_combined = df_combined.Count().GetValue()
        print(f"  -> Events with FatJet_particleNet_XbbVsQCD[1] > 0.9 or FatJet_particleNet_XttVsQCD[1] > 0.9: {count_combined:,}")

        # Final Combination
        filter = ("(FatJet_particleNet_XbbVsQCD[0] > 0.9 or FatJet_particleNet_XttVsQCD[0] > 0.9) and"
                  "(FatJet_particleNet_XbbVsQCD[1] > 0.9 or FatJet_particleNet_XttVsQCD[1] > 0.9)")
        df = df.Filter(filter, "Complex ParticleNet filter")
        cutflow.add_cut("ParticleNet filter", filter, df)

        # Step 7: Optional filtering
        df = df.Define("HighestPt", "FatJet_pt[0]")
        df = df.Define("SecondHighestPt", "FatJet_pt[1]")
        df = df.Define("HighestMass", "FatJet_mass[0]")
        df = df.Define("SecondHighestMass", "FatJet_mass[1]")

        df = df.Filter("HighestPt > 300", "1st pT > 300")
        cutflow.add_cut("1st pT > 300", "HighestPt > 300", df)

        df = df.Filter("SecondHighestPt > 500", "2nd pT > 250")
        cutflow.add_cut("2nd pT > 250", "SecondHighestPt > 250", df)

        df = df.Filter("SecondHighestMass > 30", "2nd mass > 30")
        cutflow.add_cut("2nd mass > 30", "SecondHighestMass > 30", df)

        df = df.Filter("HighestMass > 30", "1st mass > 30")
        cutflow.add_cut("1st mass > 30", "HighestMass > 30", df)

        # Print the complete cutflow summary
        cutflow.print_summary()

        # Create visualizations
        cutflow.plot_cutflow(f"cutlfow_{sample_name}.png", sample_name)
        cutflow.save_to_csv(f"cutflow_{sample_name}.csv", sample_name)

        print(f"\nCutflow analysis complete!")
        print(f"- Summary table printed above")
        print(f"- Plot saved as: cutflow_{sample_name}.png")
        print(f"- Data saved as: cutlfow_{sample_name}.csv")

        return df, cutflow

    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
def new_QCD_cutflow(file_path, sample_name="Sample"):
    """
    Testing new cuts from 28.8.2025
    """
    
    print(f"=== Cutflow Analysis for {sample_name} ===")
    print(f"File: {file_path}")
    print("=" * 60)
    
    try:
        # Load  data
        df = ROOT.RDataFrame("Events", file_path)
        
        # Initialize cutflow tracker
        cutflow = CutflowAnalysis(sample_name)
        
        #  specific analysis steps with cutflow tracking
        
        # Step 1: Start with all events
        cutflow.add_cut("All events", "1", df)
        
        # Step 2: At least 2 fat jets
        df = df.Filter("nFatJet >= 2", "nFatJet >= 2")
        cutflow.add_cut("nFatJet >= 2", "nFatJet >= 2", df)
        
        # Step 3: Leading jet pT
        df = df.Filter("FatJet_pt[0] > 250", "FatJet_pt[0] > 250")
        cutflow.add_cut("FatJet_pt[0] > 250 GeV", "FatJet_pt[0] > 250", df)

        # Step 4: Break down complex ParticleNet filter step by step

        print(f"Breakdown of complex filter:")

        # # First, just mass requirement
        # df_mass = df.Filter("Sum(FatJet_mass > 30) > 0", "Mass requirement")
        # count_mass = df_mass.Count().GetValue()
        # print(f"  Events with FatJet_mass > 30: {count_mass:,}")

        # ParticleN et XttVsQCD
        # df_xtt = df.Filter("Sum(FatJet_particleNet_XttVsQCD > 0.1) > 0", "XttVsQCD")
        # count_xtt = df_xtt.Count().GetValue()
        # print(f"  Events with XttVsQCD > 0.1: {count_xtt:,}")
        
        # # Then ParticleNet XtmVsQCD
        # df_xtm = df.Filter("Sum(FatJet_particleNet_XtmVsQCD > 0.1) > 0", "XtmVsQCD")  
        # count_xtm = df_xtm.Count().GetValue()
        # print(f"  Events with XtmVsQCD > 0.1: {count_xtm:,}")
        
        # # ParticleNet XteVsQCD  
        # df_xte = df.Filter("Sum(FatJet_particleNet_XteVsQCD > 0.1) > 0", "XteVsQCD")
        # count_xte = df_xte.Count().GetValue()
        # print(f"  Events with XteVsQCD > 0.1: {count_xte:,}")
        
        # # Now exact complex filter
        # filter = ("Sum(((FatJet_particleNet_XttVsQCD > 0.1) || (FatJet_particleNet_XtmVsQCD > 0.1) || (FatJet_particleNet_XteVsQCD > 0.1))) > 0")
        
        # print("\n")
        # df = df.Filter(filter, "Complex ParticleNet filter 1")
        # cutflow.add_cut("ParticleNet Xtt & Xtm & Xte + Mass > 30 filter", filter, df)
        
        # Step 5: Break down complex ParticleNet filter step by step

        print(f"Breakdown of complex filter:")

        # First, particleNet XbbVsQCD
        df_xbb = df.Filter("Sum(FatJet_particleNet_XbbVsQCD > 0.1) > 0", "XbbVsQCD")
        count_xbb = df_xbb.Count().GetValue()
        print(f"  Events with XbbVsQCD > 0.1: {count_xbb:,}")

        # Then, just mass requirement
        df_mass = df.Filter("Sum(FatJet_mass > 50) > 0", "Mass requirement")
        count_mass = df_mass.Count().GetValue()
        print(f"  Events with FatJet_mass > 30: {count_mass:,}")

        # Now exact complex filter
        filter = ("Sum((FatJet_particleNet_XbbVsQCD > 0.1) && (FatJet_mass > 50)) > 0")

        print("\n")
        df = df.Filter(filter, "Complex ParticleNet filter 2")
        cutflow.add_cut("ParticleNet Xbb + Mass > 50 Filter", filter, df)

        # Step 5.1: Cut Sum(mass > 50) > 1
        df = df.Filter("Sum(FatJet_mass > 50) > 1", ">=2 jets with mass > 50")
        cutflow.add_cut(">=2 jets (m>50)", "Sum(FatJet_mass > 50) > 1", df)
        
        # Additional common cuts from  analysis
        
        # Step 6: Jet selection for HT
        df = df.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")
        df = df.Filter("SelectedJets_pt.size() >= 2", "At least 2 jets")
        cutflow.add_cut(">=2 jets (pT>30, |eta|<2.4)", "SelectedJets_pt.size() >= 2", df)
        
        # Step 7: Fat jet selection  
        df = df.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")
        df = df.Filter("SelectedFatJets_pt.size() >= 2", "At least 2 fat jets in acceptance")
        cutflow.add_cut(">=2 FatJets |eta|<2.4", "SelectedFatJets_pt.size() >= 2", df)

        # Step 8: Signal Triggers
        signal_triggers = [
            "HLT_AK8PFHT800_TrimMass50",
            "HLT_AK8PFJet400_TrimMass30",
            "HLT_AK8PFJet500",
            "HLT_PFJet500",
            "HLT_PFHT1050",
            "HLT_PFHT500_PFMET100_PFMHT100_IDTight",
            "HLT_PFHT700_PFMET85_PFMHT85_IDTight",
            "HLT_PFHT800_PFMET75_PFMHT75_IDTight",
            "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg",
            "HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1"
        ]
        available_triggers = [t for t in signal_triggers if df.HasColumn(t)]
        # for trigger in available_triggers:
        #     df_temp = df.Filter(trigger, trigger)
        #     count_temp = df_temp.Count().GetValue()
        #     print(f"  Events with {trigger}: {count_temp:,}")

        combo_expr = " or ".join(available_triggers)
        df = df.Define("Combo", combo_expr)
        df = df.Filter("Combo", "Signal Triggers")
        cutflow.add_cut("Signal Trigger", "Combo", df)

        # Step 9: Reference Triggers
        reference_trigger = "HLT_AK8PFJet260"
        df = df.Filter(reference_trigger, "Reference Trigger")
        cutflow.add_cut("Reference Trigger", reference_trigger, df)
        
        # Print the complete cutflow summary
        cutflow.print_summary()
        
        # Create visualizations
        cutflow.plot_cutflow(f"cutflow_{sample_name}.png", sample_name)
        cutflow.save_to_csv(f"cutflow_{sample_name}.csv", sample_name)
        
        print(f"\nCutflow analysis complete!")
        print(f"- Summary table printed above")
        print(f"- Plot saved as: cutflow_{sample_name}.png")
        print(f"- Data saved as: cutflow_{sample_name}.csv")
        
        return df, cutflow
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    """
    Main function - modify the file paths below to point to  actual data files
    """
    
    # Example file paths - MODIFY THESE to point to  actual files
    # example_files = {
    #     "QCD": "/Users/khatran/Documents/CERN/cern-source/project/CMS-trigger-efficiency-azura/HHbbtautau/QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
    #     "ggF": "/Users/khatran/Documents/CERN/cern-source/project/CMS-trigger-efficiency-azura/HHbbtautau/GluGluToHHTo2B2Tau_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_2jets_tight.root",
    #     "DATA": "/Users/khatran/Documents/CERN/cern-source/project/CMS-trigger-efficiency-azura/HHbbtautau/JetHT_Run2018_2jets_tight_v3.root"
    # }

    example_files = {
        "$H_{T}: 200-300$": "data/HHbbtautau-v2/QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 300-500$": "data/HHbbtautau-v2/QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 500-700$": "data/HHbbtautau-v2/QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 700-1000$": "data/HHbbtautau-v2/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 1000-1500$": "data/HHbbtautau-v2/QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 1500-2000$": "data/HHbbtautau-v2/QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
        "$H_{T}: 2000-Inf$": "data/HHbbtautau-v2/QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_2jets_tight.root",
    }
    
    # If you want to test with just one file, uncomment and modify this:
    single_file = "data/HHbbtautau-v2/QCD_combined.root"
    # single_file = "data/HHbbtautau-v2/GluGluToHHTo2B2Tau_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_2jets_tight.root"
    # simple_cutflow_example(single_file, "TestSample")
    # ggF_cutflow_example(single_file, "ggFSampleAnalysis")
    new_QCD_cutflow(single_file, "newQCDTest")
    return
    
    # Run cutflow analysis on multiple samples
    all_cutflows = []
    
    for sample_name, file_path in example_files.items():
        if os.path.exists(file_path):
            print(f"\n{'='*80}")
            final_df, cutflow = simple_cutflow_example(file_path, sample_name)
            if cutflow:
                all_cutflows.append(cutflow)
        else:
            print(f"\nFile not found: {file_path}")
            print("Please update the file path in the script.")
    
    # If we have multiple samples, create a comparison
    if len(all_cutflows) > 1:
        print(f"\n{'='*80}")
        print("CREATING COMPARISON PLOT")
        print("="*80)
        
        from cutflow_analysis import compare_cutflows
        compare_cutflows(all_cutflows, "cutflow_comparison.png")
        print("Comparison plot saved as: cutflow_comparison.png")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import matplotlib.pyplot as plt
        from tabulate import tabulate
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install matplotlib tabulate")
        sys.exit(1)
    
    main()
