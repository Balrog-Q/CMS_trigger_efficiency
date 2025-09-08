from library.trigger_efficiency_ML import *
import ROOT
import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

def process_data(file_path, run_name, suffix, output_tree_name, output_file_name, signal_triggers, reference_triggers, pass_ref, plotting):
    """
    Process the data from the given file path.
    
    Parameters:
    file_path (str): The path to the data file.
    
    Returns:
    pd.DataFrame: Processed data as a DataFrame.
    """
    
    # Define the signal trigger and filter names
    signal_trigger   = "Combo"

    reference_trigger = reference_triggers[0]

    filter_pass_real = signal_trigger

    filter_pass_meas = signal_trigger + "&&" + reference_trigger

    filter_all_meas  = reference_trigger

    all_columns      = ["HighestPt", "HT", "MET_pt", "mHH","HighestMass", "SecondHighestPt", 
                        "SecondHighestMass", "FatHT", "MET_FatJet", "mHHwithMET", "HighestEta", 
                        "SecondHighestEta", "DeltaEta", "DeltaPhi", 
                        "nJet_pt30", "maxFatJetMass", "FatJetBalance", "minDeltaPhiJetMET", "TauPt1st", "TauPt2nd",
                        "Combo", "HLT_AK8PFJet260"]

    ROOT.gInterpreter.Declare("""
                              float mass_3(float pt1, float eta1, float phi1, float m1,
                                           float pt2, float eta2, float phi2, float m2,
                                           float pt3, float eta3, float phi3, float m3) {
                                typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> PtEtaPhiMVector;
                                PtEtaPhiMVector p41(pt1, eta1, phi1, m1);
                                PtEtaPhiMVector p42(pt2, eta2, phi2, m2);
                                PtEtaPhiMVector p43(pt3, eta3, phi3, m3);
                                return (p41 + p42 + p43).M();
                              }
                              
                              float computeMinDeltaPhiJetMET(ROOT::VecOps::RVec<float> jet_phi, float met_phi) {
                                if (jet_phi.size() == 0) return -1.0;
                                float min_dphi = 999.0;
                                for (auto jphi : jet_phi) {
                                  float dphi = TMath::Abs(ROOT::VecOps::DeltaPhi(jphi, met_phi));
                                  if (dphi < min_dphi) min_dphi = dphi;
                                }
                                return min_dphi;
                              }
                              """)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Originally the QCD sample was in several different files each with a different HT ranges
    df_QCD1 = ROOT.RDataFrame("Events", file_path[0])
    df_QCD2 = ROOT.RDataFrame("Events", file_path[1])
    df_QCD3 = ROOT.RDataFrame("Events", file_path[2])
    df_QCD4 = ROOT.RDataFrame("Events", file_path[3])
    df_QCD5 = ROOT.RDataFrame("Events", file_path[4])
    df_QCD6 = ROOT.RDataFrame("Events", file_path[5])
    df_QCD7 = ROOT.RDataFrame("Events", file_path[6])

    df_list = [df_QCD1, df_QCD2, df_QCD3, df_QCD4, df_QCD5, df_QCD6, df_QCD7]
    df_QCD_list = []

    for df in df_list:
        df = df.Filter("nFatJet >= 2")
        df = df.Filter("FatJet_pt[0] > 250")
        # df = df.Filter("Sum((FatJet_particleNet_XbbVsQCD > 0.1) && (FatJet_mass > 50)) > 0")
        # df = df.Filter("Sum((FatJet_particleNet_XttVsQCD > 0.1) || (FatJet_particleNet_XtmVsQCD > 0.1) || (FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 30)) > 0")
        # df = df.Filter("Sum((FatJet_particleNet_XbbVsQCD > 0.1) || (FatJet_particleNet_XttVsQCD > 0.1) || (FatJet_particleNet_XtmVsQCD > 0.1) || (FatJet_particleNet_XteVsQCD > 0.1) && (FatJet_mass > 50)) > 0")
        # df = df.Filter("Sum(((FatJet_particleNet_XttVsQCD > 0.1) || (FatJet_particleNet_XtmVsQCD > 0.1) || (FatJet_particleNet_XteVsQCD > 0.1)) && (FatJet_mass > 30)) > 0")
        df = df.Filter("Sum((FatJet_particleNet_XbbVsQCD > 0.1) && (FatJet_mass > 50)) > 0")
        df = df.Filter("Sum(FatJet_mass > 50) > 1")

        # Define the necessary variables for the analysis
        df = df.Define("HighestPt", "FatJet_pt[0]")
        df = df.Define("SecondHighestPt", "FatJet_pt[1]")
        df = df.Define("SelectedMass", "FatJet_mass[abs(FatJet_eta) < 2.4]")
        df = df.Define("HighestMass", "SelectedMass[0]")
        df = df.Define("SecondHighestMass", "SelectedMass[1]")
        df = df.Define("mHH", "InvariantMass(Take(FatJet_pt, 2),Take(FatJet_eta, 2),Take(FatJet_phi, 2),Take(FatJet_mass, 2))") 
        df = df.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")
        df = df.Filter("SelectedJets_pt.size() >= 2")
        df = df.Define("HT", "Sum(SelectedJets_pt)")
        df = df.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")
        df = df.Filter("SelectedFatJets_pt.size() >= 2")
        df = df.Define("FatHT", "Sum(SelectedFatJets_pt)")
        df = df.Define("MET_FatJet", "(FatJet_pt[0]+FatJet_pt[1]+MET_pt)")
        df = df.Define("mHHwithMET", "mass_3(FatJet_pt[0],FatJet_eta[0],FatJet_phi[0],FatJet_mass[0],FatJet_pt[1],FatJet_eta[1],FatJet_phi[1],FatJet_mass[1],MET_pt,0.0,MET_phi,0.0)")

        available_triggers = [t for t in signal_triggers if trigger_exists(df, t)]
        print("List of all signal trigger used for QCD data:")
        for trigger in available_triggers:
            print("-",trigger)
        combo_expr = " or ".join(available_triggers)
        # New definition of Combo to include tau triggers
        df = df.Define("Combo", combo_expr)

        df = df.Define("SelectedEta", "FatJet_eta[abs(FatJet_eta) < 2.4]")
        df = df.Define("HighestEta", "SelectedEta[0]")
        df = df.Define("SecondHighestEta" , "SelectedEta[1]")
        df = df.Define("SelectedPhi", "FatJet_phi[abs(FatJet_eta) < 2.4]")
        df = df.Define("DeltaPhi", "SelectedPhi[0]-SelectedPhi[1]")
        df = df.Define("DeltaEta", "abs(HighestEta-SecondHighestEta)")

        # New Tau variables
        df = df.Define("nJet_pt30", "Sum(Jet_pt > 30)")
        df = df.Define("maxFatJetMass", "FatJet_mass.size() > 0 ? Max(FatJet_mass) : -1")
        df = df.Define("FatJetBalance", "(FatJet_pt[0] - FatJet_pt[1])/(FatJet_pt[0] + FatJet_pt[1])")
        df = df.Define("minDeltaPhiJetMET", "computeMinDeltaPhiJetMET(Jet_phi, MET_phi)")
        df = df.Define("TauPt1st", "Tau_pt.size() > 0 ? Tau_pt[0] : 0")
        df = df.Define("TauPt2nd", "Tau_pt.size() > 1 ? Tau_pt[1] : 0")
            
        df_QCD_list.append(df)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # The ggF sample is a specific di-Higgs production process, while the QCD sample 
    # is a background process that includes various QCD events.
    df_ggF = ROOT.RDataFrame("Events", file_path[7])
    # Filter to ensure at least two fat jets are present
    df_ggF = df_ggF.Filter("nFatJet >= 2", "All events")

    df_ggF = df_ggF.Define("HighestPt", "FatJet_pt[0]")
    df_ggF = df_ggF.Define("SecondHighestPt", "FatJet_pt[1]")
    df_ggF = df_ggF.Define("HighestMass", "FatJet_mass[0]")
    df_ggF = df_ggF.Define("SecondHighestMass", "FatJet_mass[1]")
    df_ggF = df_ggF.Define("mHH", "InvariantMass(Take(FatJet_pt, 2), Take(FatJet_eta, 2), Take(FatJet_phi, 2), Take(FatJet_mass, 2))") 
    df_ggF = df_ggF.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")

    # Filter to ensure at least two selected jets
    df_ggF = df_ggF.Filter("SelectedJets_pt.size() >= 2")

    df_ggF = df_ggF.Define("HT", "Sum(SelectedJets_pt)")
    df_ggF = df_ggF.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")

    # Filter to ensure at least two selected fat jets
    df_ggF = df_ggF.Filter("SelectedFatJets_pt.size() >= 2")

    df_ggF = df_ggF.Define("FatHT", "Sum(SelectedFatJets_pt)")
    df_ggF = df_ggF.Define("MET_FatJet", "(FatJet_pt[0] + FatJet_pt[1] + MET_pt)")
    df_ggF = df_ggF.Define("mHHwithMET", "mass_3(FatJet_pt[0], FatJet_eta[0], FatJet_phi[0], FatJet_mass[0], FatJet_pt[1], FatJet_eta[1], FatJet_phi[1], FatJet_mass[1], MET_pt, 0.0, MET_phi, 0.0)")

    available_triggers = [t for t in signal_triggers if trigger_exists(df_ggF, t)]
    print("List of all signal trigger used for ggF data:")
    for trigger in available_triggers:
        print("-",trigger)
    combo_expr = " or ".join(available_triggers)
    # New definition of Combo to include tau triggers
    df_ggF = df_ggF.Define("Combo", combo_expr)

    df_ggF = df_ggF.Define("SelectedEta", "FatJet_eta[abs(FatJet_eta) < 2.4]")
    df_ggF = df_ggF.Define("HighestEta", "SelectedEta[0]")
    df_ggF = df_ggF.Define("SecondHighestEta" , "SelectedEta[1]")
    df_ggF = df_ggF.Define("SelectedPhi", "FatJet_phi[abs(FatJet_eta) < 2.4]")
    df_ggF = df_ggF.Define("DeltaPhi", "SelectedPhi[0] - SelectedPhi[1]")
    df_ggF = df_ggF.Define("DeltaEta", "abs(HighestEta - SecondHighestEta)")
    # New Tau variables
    df_ggF = df_ggF.Define("nJet_pt30", "Sum(Jet_pt > 30)")
    df_ggF = df_ggF.Define("maxFatJetMass", "FatJet_mass.size() > 0 ? Max(FatJet_mass) : -1")
    df_ggF = df_ggF.Define("FatJetBalance", "(FatJet_pt[0] - FatJet_pt[1])/(FatJet_pt[0] + FatJet_pt[1])")
    df_ggF = df_ggF.Define("minDeltaPhiJetMET", "computeMinDeltaPhiJetMET(Jet_phi, MET_phi)")
    df_ggF = df_ggF.Define("TauPt1st", "Tau_pt.size() > 0 ? Tau_pt[0] : 0")
    df_ggF = df_ggF.Define("TauPt2nd", "Tau_pt.size() > 1 ? Tau_pt[1] : 0")

    df_ggF = df_ggF.Filter("(FatJet_particleNet_XbbVsQCD[0] > 0.9 or FatJet_particleNet_XttVsQCD[0] > 0.9) and (FatJet_particleNet_XbbVsQCD[1] > 0.9 or FatJet_particleNet_XttVsQCD[1] > 0.9)")

    # Filter the ggF dataframes based on the defined criteria
    df_ggF_filtered = df_ggF.Filter("HighestPt > 300")
    df_ggF_filtered = df_ggF_filtered.Filter("SecondHighestPt > 250")
    df_ggF_filtered = df_ggF_filtered.Filter("SecondHighestMass > 30")
    df_ggF_filtered = df_ggF_filtered.Filter("HighestMass > 30")

    df_ggF.Snapshot(output_tree_name[1], output_file_name[1])

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # The VBF sample is used for the VBF category in the analysis.
    df_VBF = ROOT.RDataFrame("Events", file_path[8])
    # Filter to ensure at least two fat jets are present
    df_VBF = df_VBF.Filter("nFatJet >= 2", "All events")

    df_VBF = df_VBF.Define("HighestPt", "FatJet_pt[0]")
    df_VBF = df_VBF.Define("SecondHighestPt", "FatJet_pt[1]")
    df_VBF = df_VBF.Define("HighestMass", "FatJet_mass[0]")
    df_VBF = df_VBF.Define("SecondHighestMass", "FatJet_mass[1]")
    df_VBF = df_VBF.Define("mHH", "InvariantMass(Take(FatJet_pt, 2), Take(FatJet_eta, 2), Take(FatJet_phi, 2), Take(FatJet_mass, 2))") 
    df_VBF = df_VBF.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")

    # Filter to ensure at least two selected jets
    df_VBF = df_VBF.Filter("SelectedJets_pt.size() >= 2")

    df_VBF = df_VBF.Define("HT", "Sum(SelectedJets_pt)")
    df_VBF = df_VBF.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")

    # Filter to ensure at least two selected fat jets
    df_VBF = df_VBF.Filter("SelectedFatJets_pt.size() >= 2")

    df_VBF = df_VBF.Define("FatHT", "Sum(SelectedFatJets_pt)")
    df_VBF = df_VBF.Define("MET_FatJet", "(FatJet_pt[0] + FatJet_pt[1] + MET_pt)")
    df_VBF = df_VBF.Define("mHHwithMET","mass_3(FatJet_pt[0], FatJet_eta[0], FatJet_phi[0], FatJet_mass[0], FatJet_pt[1], FatJet_eta[1], FatJet_phi[1], FatJet_mass[1], MET_pt, 0.0, MET_phi, 0.0)")

    available_triggers = [t for t in signal_triggers if trigger_exists(df_VBF, t)]
    print("List of all signal trigger used for VBF data:")
    for trigger in available_triggers:
        print("-",trigger)
    combo_expr = " or ".join(available_triggers)
    # New definition of Combo to include tau triggers
    df_VBF = df_VBF.Define("Combo", combo_expr)

    df_VBF = df_VBF.Define("SelectedEta", "FatJet_eta[abs(FatJet_eta) < 2.4]")
    df_VBF = df_VBF.Define("HighestEta", "SelectedEta[0]")
    df_VBF = df_VBF.Define("SecondHighestEta" , "SelectedEta[1]")
    df_VBF = df_VBF.Define("SelectedPhi", "FatJet_phi[abs(FatJet_eta) < 2.4]")
    df_VBF = df_VBF.Define("DeltaPhi", "SelectedPhi[0] - SelectedPhi[1]")
    df_VBF = df_VBF.Define("DeltaEta", "abs(HighestEta - SecondHighestEta)")
    # New Tau variables
    df_VBF = df_VBF.Define("nJet_pt30", "Sum(Jet_pt > 30)")
    df_VBF = df_VBF.Define("maxFatJetMass", "FatJet_mass.size() > 0 ? Max(FatJet_mass) : -1")
    df_VBF = df_VBF.Define("FatJetBalance", "(FatJet_pt[0] - FatJet_pt[1])/(FatJet_pt[0] + FatJet_pt[1])")
    df_VBF = df_VBF.Define("minDeltaPhiJetMET", "computeMinDeltaPhiJetMET(Jet_phi, MET_phi)")
    df_VBF = df_VBF.Define("TauPt1st", "Tau_pt.size() > 0 ? Tau_pt[0] : 0")
    df_VBF = df_VBF.Define("TauPt2nd", "Tau_pt.size() > 1 ? Tau_pt[1] : 0")

    # Filter to ensure the fat jets pass the particleNet selection criteria
    df_VBF = df_VBF.Filter("(FatJet_particleNet_XbbVsQCD[0] > 0.9 or FatJet_particleNet_XttVsQCD[0] > 0.9) and (FatJet_particleNet_XbbVsQCD[1] > 0.9 or FatJet_particleNet_XttVsQCD[1] > 0.9)")

    # Filter the VBF dataframes based on the defined criteria
    df_VBF_filtered = df_VBF.Filter("HighestPt > 300")
    df_VBF_filtered = df_VBF_filtered.Filter("SecondHighestPt > 250")
    df_VBF_filtered = df_VBF_filtered.Filter("SecondHighestMass > 30")
    df_VBF_filtered = df_VBF_filtered.Filter("HighestMass > 30")

    df_VBF.Snapshot(output_tree_name[2], output_file_name[2])

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Loading the real DATA from CMS 2018 data
    df_DATA = ROOT.RDataFrame("Events", file_path[9])
    # Filter to ensure at least two fat jets are present
    df_DATA = df_DATA.Filter("nFatJet >= 2", "All events")
    # df_DATA = df_DATA.Filter("FatJet_pt[0] > 250", "All events")

    df_DATA = df_DATA.Define("HighestPt", "FatJet_pt[0]")
    df_DATA = df_DATA.Define("SecondHighestPt", "FatJet_pt[1]")
    df_DATA = df_DATA.Define("HighestMass", "FatJet_mass[0]")
    df_DATA = df_DATA.Define("SecondHighestMass", "FatJet_mass[1]")
    df_DATA = df_DATA.Define("mHH", "InvariantMass(Take(FatJet_pt, 2), Take(FatJet_eta, 2), Take(FatJet_phi, 2), Take(FatJet_mass, 2))")
    df_DATA = df_DATA.Define("SelectedJets_pt", "Jet_pt[Jet_pt > 30 && abs(Jet_eta) < 2.4]")

    # Filter to ensure at least two selected jets
    df_DATA = df_DATA.Filter("SelectedJets_pt.size() >= 2")

    df_DATA = df_DATA.Define("HT", "Sum(SelectedJets_pt)")
    df_DATA = df_DATA.Define("SelectedFatJets_pt", "FatJet_pt[abs(FatJet_eta) < 2.4]")

    # Filter to ensure at least two selected fat jets
    df_DATA = df_DATA.Filter("SelectedFatJets_pt.size() >= 2")

    df_DATA = df_DATA.Define("FatHT", "Sum(SelectedFatJets_pt)")
    df_DATA = df_DATA.Define("MET_FatJet", "(FatJet_pt[0] + FatJet_pt[1] + MET_pt)")
    df_DATA = df_DATA.Define("mHHwithMET", "mass_3(FatJet_pt[0], FatJet_eta[0], FatJet_phi[0], FatJet_mass[0], FatJet_pt[1], FatJet_eta[1], FatJet_phi[1], FatJet_mass[1], MET_pt, 0.0, MET_phi, 0.0)")

    available_triggers = [t for t in signal_triggers if trigger_exists(df_DATA, t)]
    print("List of all signal trigger used for Real DATA:")
    for trigger in available_triggers:
        print("-",trigger)
    combo_expr = " or ".join(available_triggers)
    # New definition of Combo to include tau triggers
    df_DATA = df_DATA.Define("Combo", combo_expr)

    # Combo for the triggers of di-Higgs events
    # df = df.Define("Combo", "HLT_AK8PFHT800_TrimMass50 or HLT_AK8PFJet400_TrimMass30 or HLT_AK8PFJet500 or HLT_PFJet500 or HLT_PFHT1050 or HLT_PFHT500_PFMET100_PFMHT100_IDTight or HLT_PFHT700_PFMET85_PFMHT85_IDTight or HLT_PFHT800_PFMET75_PFMHT75_IDTight")

    df_DATA = df_DATA.Define("DeltaPhi", "(FatJet_phi[0] - FatJet_phi[1])")
    df_DATA = df_DATA.Define("DeltaEta", "abs(FatJet_eta[0] - FatJet_eta[1])")
    df_DATA = df_DATA.Define("HighestEta","FatJet_eta[0]")
    df_DATA = df_DATA.Define("SecondHighestEta" , "FatJet_eta[1]")
    # New Tau variables
    df_DATA = df_DATA.Define("nJet_pt30", "Sum(Jet_pt > 30)")
    df_DATA = df_DATA.Define("maxFatJetMass", "FatJet_mass.size() > 0 ? Max(FatJet_mass) : -1")
    df_DATA = df_DATA.Define("FatJetBalance", "(FatJet_pt[0] - FatJet_pt[1])/(FatJet_pt[0] + FatJet_pt[1])")
    df_DATA = df_DATA.Define("minDeltaPhiJetMET", "computeMinDeltaPhiJetMET(Jet_phi, MET_phi)")
    df_DATA = df_DATA.Define("TauPt1st", "Tau_pt.size() > 0 ? Tau_pt[0] : 0")
    df_DATA = df_DATA.Define("TauPt2nd", "Tau_pt.size() > 1 ? Tau_pt[1] : 0")

    # Filter to ensure at least two selected fat jets
    df_DATA = df_DATA.Filter("HLT_AK8PFJet260")

    df_DATA.Snapshot(output_tree_name[3], output_file_name[3])

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    all_pass_ref_trigger = pass_ref

    pddf_list = []

    if all_pass_ref_trigger == True:
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("QCD")
        for df in df_QCD_list:
            pddf, _ = RD_to_pandas(df, reference_triggers[0], all_columns, names_list)
            pddf_list.append(pddf)

    elif all_pass_ref_trigger == False:
        for df in df_QCD_list:
            pddf = pd.DataFrame(df.AsNumpy(all_columns))
            pddf_list.append(pddf)

    result = pd.concat(pddf_list, axis=0, ignore_index=True)

    # HT distribution of the data sample
    hist = df_DATA.Histo1D(("HT", "; HT (GeV);Events", 165, 450, 2100), "HT")
    if plotting:
        draw_histogram(hist, "", "HT_DATA.png", suffix=suffix)
    
    # Get values per interval 
    values_per_interval = [hist.GetBinContent(hist.FindBin(interval)) for interval in np.arange(450, 2110, 10)]

    # Select values for each interval for all columns
    selected_values = pd.DataFrame()

    for interval, num_values in zip(np.arange(450, 2110, 10), values_per_interval):
        subset = result.loc[(result['HT'] >= interval) & (result['HT'] < interval + 10)]
        selected_subset = pd.DataFrame()
        
        for column in result.columns:
            # replace=True allows the sampling function to pick the same event more than once. This is a standard technique (boostrapping with replacement) and is perfectly acceptable for this kind of background modeling. It ensures your `HT` perfectly match the data's
            # shape, although some simulated events will be reused.
            selected_subset[column] = subset[column].sample(int(num_values), replace=True, random_state=42).values

            # Determine the number of events to sample: either the number from data or the number available in the subset, whichever is smaller.
            # If you want to avoid re-using events, you can simply take the maximum number of events available in the simulation for that bin. The downside is that your final QCD `HT` shape will not perfectly match the data's shape in the bins where you have a deficit.
            # n_samples = min(int(num_values), len(subset))
            # selected_subset[column] = subset[column].sample(n_samples, random_state=42).values
        
        selected_values = pd.concat([selected_values, selected_subset])
        
    # Convert the pandas array columns into cpp vectors so they can be converted into an RDataframe
    numpy_array = selected_values.values
    cpp_vector_list = []

    for i in range(0,len(all_columns)):
        cpp_vector = numpy_array[:,i].astype(dtype=np.float64)
        cpp_vector_list.append(cpp_vector)


    df_QCD = ROOT.RDF.FromNumpy({"HighestPt": cpp_vector_list[0], "HT" : cpp_vector_list[1], "MET_pt" : cpp_vector_list[2], "mHH" : cpp_vector_list[3],
                                "HighestMass" : cpp_vector_list[4], "SecondHighestPt" : cpp_vector_list[5], "SecondHighestMass" : cpp_vector_list[6], 
                                "FatHT" : cpp_vector_list[7], "MET_FatJet" : cpp_vector_list[8], "mHHwithMET" : cpp_vector_list[9],
                                "HighestEta" : cpp_vector_list[10],"SecondHighestEta" : cpp_vector_list[11],  "DeltaEta" : cpp_vector_list[12], 
                                "DeltaPhi" : cpp_vector_list[13], 
                                "nJet_pt30": cpp_vector_list[14], "maxFatJetMass": cpp_vector_list[15], "FatJetBalance": cpp_vector_list[16],
                                "minDeltaPhiJetMET": cpp_vector_list[17], "TauPt1st": cpp_vector_list[18], 'TauPt2nd': cpp_vector_list[19],
                                "Combo" : cpp_vector_list[20], "HLT_AK8PFJet260" : cpp_vector_list[21]})
    
    df_QCD.Snapshot(output_tree_name[0], output_file_name[0])

    # ----------------------------------------------------------------------------------------------------------
    if plotting:
        # Comparison plots
        # Create lists for distributions of events passing the signal trigger and all events in the simulation samples

        num_QCD_list = []
        denom_QCD_list = []
        num_ggF_list = []
        denom_ggF_list = []
        num_VBF_list = []
        denom_VBF_list = []
        all_nums_and_denoms = []

        y_range_list = [3000, 3000, 4000, 5500, 15000, 3000, 15000, 3000, 5000, 4500, 30000, 30000, 25000, 20000]

        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("dist")

        for j in range(len(range_min_list)):
            num_QCD, denom_QCD = numerator_and_denominator(df_QCD, "QCD", df_DATA, filter_pass_real, filter_all_meas, variable_list[j], names_list[j], names_list_plot[j], y_range_list[j], filter_pass_meas, filter_all_meas, run_name, suffix=suffix)
            num_QCD_list.append(num_QCD)
            denom_QCD_list.append(denom_QCD)

            num_ggF, denom_ggF = numerator_and_denominator(df_ggF, "ggF", df_DATA, filter_pass_real, filter_all_meas, variable_list[j], names_list[j], names_list_plot[j], y_range_list[j], filter_pass_meas, filter_all_meas, run_name, suffix=suffix)
            num_ggF_list.append(num_ggF)
            denom_ggF_list.append(denom_ggF)

            num_VBF, denom_VBF = numerator_and_denominator(df_VBF, "VBF", df_DATA, filter_pass_real, filter_all_meas, variable_list[j], names_list[j], names_list_plot[j], y_range_list[j], filter_pass_meas, filter_all_meas, run_name, suffix=suffix)
            num_VBF_list.append(num_VBF)
            denom_VBF_list.append(denom_VBF)

            # Combine all nums and denoms for this iteration into a single list and append to all_numsa_and_denoms
            combined_list = [num_QCD, denom_QCD, num_ggF, denom_ggF, num_VBF, denom_VBF]
            all_nums_and_denoms.append(combined_list)

        # ----------------------------------------------------------------------------------------------------------
        # Distributions of all the simulation samples in the same plot
        # variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("dist")

        y_range_list = [3000, 3000, 4000, 5500, 15000, 3000, 15000, 3000, 5000, 4500, 30000, 30000, 25000, 20000]

        color_list   = [ROOT.kViolet, ROOT.kAzure-5, ROOT.kRed+1, ROOT.kOrange+2, ROOT.kGray+3, ROOT.kGreen+2]

        legend_list  = ["#splitline{QCD events passing the}{signal trigger}", "QCD all events", "#splitline{ggF events passing the}{signal trigger}",
                    "ggF all events", "#splitline{VBF events passing the}{signal trigger}", "VBF all events"]

        for i in range(len(num_QCD_list)):
            draw_histograms_same(all_nums_and_denoms[i], color_list, y_range_list[i], legend_list, names_list_plot[i], "Events", "Distribution_MC_" + names_list[i] + ".png", "QCD", suffix=suffix)

        # ----------------------------------------------------------------------------------------------------------
        # Distributions of event passing the reference trigger and signal and reference trigger for the data sample
        variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list = define_parameter("dist-DATA")

        y_range_list    = [2000, 2000, 3000, 3500, 8000, 2000, 6000, 2000, 4000, 4000, 10000, 10000, 7000, 11000]
        num_DATA_list   = []
        denom_DATA_list = []

        for j in range(len(range_min_list)):
            num_DATA, denom_DATA = numerator_and_denominator(df_DATA, "DATA", df_DATA, filter_pass_real, filter_all_meas, variable_list[j], names_list[j], names_list_plot[j], [], filter_pass_meas, filter_all_meas, run_name, suffix=suffix)
            num_DATA_list.append(num_DATA)
            denom_DATA_list.append(denom_DATA)

    return

if __name__ == '__main__':
    version = input("Enter the version you want (e.g. v1): ")
    run_name = input("Enter the run name: ")

    if version == "v1":
        suffix = "briar"
    elif version == "v2":
        suffix = "azura"
    elif version == "v3":
        suffix = "ashe"
    elif version == "v4":
        suffix = "cypress"
    elif version == "v5":
        suffix = "azura-v2"
    elif version == "v6":
        suffix = "newQCD"

    if version == "v1":
        tree_path = "data/HHbbtautau-v1/"
        root = "tree.root"
        realdata = "JetHT_tree.root"
    else:
        tree_path = "data/HHbbtautau-v2/"
        root = "2jets_tight.root"
        realdata = "JetHT_Run2018_2jets_tight_v3.root"

    simulation = ["QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_",
                  "GluGluToHHTo2B2Tau_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_",
                  "VBF_HHTo2B2Tau_CV_1_C2V_0_C3_1_TuneCP5_13TeV-powheg-pythia8_"]
    
    file_path = [tree_path + t + root for t in simulation]
    file_path.append(tree_path + realdata)
    
    samples = ["QCD", "ggF", "VBF", "DATA"]
    output_tree_name = [suffix + "-" + "New" + t for t in samples]
    path = f"data/processed/{suffix}"
    if not os.path.exists(path):
        os.makedirs(path)

    output_file_name = [path + "/" + t + ".root" for t in output_tree_name]
    
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

    reference_triggers = ["HLT_AK8PFJet260"]

    pass_ref = False
    plotting = False

    print(f"\nList of data files ({len(file_path)} files):")
    for file in file_path:
        print(f"- {file}")
    print("-" * 30)

    start_time = time.time()
    process_data(file_path, run_name, suffix, output_tree_name, output_file_name, signal_triggers, reference_triggers, pass_ref, plotting)
    end_time = time.time() - start_time
    print("-" * 30)
    print("End processing, all data saved. Files are saved at:")
    for j in range(len(output_file_name)):
        print(f"- {output_file_name[j]} with tree name {output_tree_name[j]}")
    print(f"Processing time: {end_time:.2f} seconds")