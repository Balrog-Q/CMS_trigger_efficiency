import os, ROOT # Import ROOT for HEP analysis

ROOT.ROOT.EnableImplicitMT(1)  # Enable multi-threading in ROOT

from ROOT import TEfficiency, TLegend, gPad, Math, VecOps # Import ROOT classes for efficiency calculations and vector operations
from ROOT.VecOps import Concatenate # type: ignore # Import VecOps for efficient vector operations
from ROOT import TH1F, TH1D # Import ROOT histograms for data visualization

import xgboost as xgb

# Install and import common HEP (High Energy Physics) analysis packages
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier # type: ignore # Import scikit-learn for machine learning algorithms
from sklearn.neural_network import MLPClassifier # type: ignore # Import scikit-learn for machine learning algorithms

import sklearn.metrics as metrics # type: ignore # Import scikit-learn for machine learning and evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, log_loss, mean_squared_error, roc_curve, roc_auc_score, auc # type: ignore # Import scikit-learn for machine learning and evaluation metrics
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_score, RepeatedKFold, GridSearchCV # type: ignore # Import scikit-learn for machine learning and model evaluation
from sklearn.preprocessing import StandardScaler # type: ignore # Import scikit-learn for machine learning and data preprocessing

import numpy as np # Import NumPy for numerical operations
import pandas as pd # Import NumPy for numerical operations and Pandas for data manipulation
import matplotlib.pyplot as plt # Import NumPy for numerical operations

import scipy # type: ignore # Import SciPy for scientific computing
from scipy.stats import binned_statistic # Import SciPy for scientific computing and statistics
from scipy.sparse import csr_matrix # Import SciPy for scientific computing and statistics

import tensorflow as tf # Import TensorFlow for machine learning
from tensorflow import keras # Import TensorFlow and Keras for deep learning
from tensorflow.keras.models import Sequential # type: ignore # Import Keras for building neural networks
from tensorflow.keras.layers import Dense, Flatten # type: ignore # Import Keras layers for building neural networks

import keras_tuner as kt # Import Keras Tuner for hyperparameter tuning

import hist # Import hist for histogramming
from hist import Hist # Import hist for histogramming

import pickle # Import pickle for saving/loading models and data
import mplhep as hep # Import mplhep for HEP-style plotting

from array import array # Import array for efficient array handling

from typing import Dict, List, Tuple, Any

import os
from datetime import datetime
from pathlib import Path

def trigger_exists(df, trigger_name):
    return df.HasColumn(trigger_name)

def get_plot_directory(suffix):
    """
    Checks for a directory named with the current date in 'dd-mm-yyyy' format.
    If the directory does not exist, it creates it.

    Returns:
        str: The path of the directory for today's plots.
    """

    # Get the current date and format it as requested.
    dirc = "result/" + datetime.now().strftime("%d-%m-%Y") + suffix
    out_dir = Path(dirc).resolve()
    

    # Check if a directory with this name already exists
    if not os.path.exists(out_dir):
        print(f"Directory '{out_dir}' not found. Creating new directory for plots.")
        # os.makedirs(dirc)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Return the directory name
    return out_dir

def efficiency_plot(df1, df2, define, variable, label, name, suffix, save_plot=True):
    """
    Plots the trigger efficiency as a function of the given variable and saves it as a file/

    Args:
        df1 (RDataframe): Dataframe containing the points that pass the trigger (numerator).
        df2 (RDataframe): Dataframe containing all the data points (denominator).
        define: A list containing the axis names, the number of bins and the ranges, e.g. ("FatJet_pt[0]", "; Highest FatJet pt (GeV); Efficiency", range_min_pt, range_min_pt, range_max_pt).
        variable (string): The variable. to use to plot the efficiency, e.g. "HighestPt".
        label (string): Title for the plot.
        name (string): Name for the file to save the plot in.

    Returns:
        eff: The TEfficiency object.

    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)

    h1 = df1.Histo1D(define, variable) # Create a histogram for the numerator (points that pass the trigger)
    h2 = df2.Histo1D(define, variable) # Create a histogram for the denominator (all data points)

    h1_c = h1.Clone() # Clone the histogram for the numerator
    h2_c = h2.Clone() # Clone the histogram for the denominator
    c = ROOT.TCanvas("c", "", 700, 700) # Create a canvas to draw the histograms
    
    eff = TEfficiency(h1_c, h2_c) # Create a TEfficiency object from the histograms
    eff.Draw() # Draw the efficiency plot
    eff.SetTitle(label) # Set the title of the efficiency plot

    # Construct the full path and save the file inside the dated folder
    if save_plot == True:
        full_path = os.path.join(plot_dir, name)
        c.SaveAs(full_path) # Save the efficiency plot as a file

    return eff

def efficiency_plot_stds(eff1, eff2, h1, h2, h3, h4, legend1, legend2, legend3, legend4, label, ymin, ymax, xmin, xmax, xlabel, name, sample, suffix):
    """
    Plots two or three efficiency plots with uncertainties on top of each other and saves it as a file.

    Args:
        eff1: The first efficiency object to plot (true efficiency)
        eff2: The second efficiency object to plot (measured efficiency)
        h1: The histogram containing the trigger efficiency predicted by a ML algorithm
        h2: Upper limit uncertainties
        h3: Lower limit uncertainties
        h4: Another architecture uncertainties
        legend1, legend2, legend3, legend4: Legends for the histograms (true efficiency, measured, ML1, ML2)
        label (string): Title for the plot
        ymin, ymax: Min and max ranges for y axis.
        xmin, xmax: Min and max ranges for x axis.
        name (string): Name for the file to save the plot in.

    Returns:
        c: The canvas object.

    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)

    ROOT.gStyle.SetOptStat(0) # Disable the statistics box
    ROOT.gStyle.SetTextFont(42) # Set the text font to Helvetica
    c = ROOT.TCanvas("c", "", 900, 700) # Create a canvas to draw the histograms

    h1.SetMarkerColor(ROOT.kBlue) # Set the marker color of the histogram
    h2.SetLineColor(ROOT.kViolet+1) # Set the line color of the upper limit uncertainties histogram
    h2.SetMarkerColor(ROOT.kViolet+1) # Set the marker color of the upper limit uncertainties histogram
    h3.SetLineColor(ROOT.kBlue-7) # Set the line color of the lower limit uncertainties histogram
    h2.SetMarkerColor(ROOT.kBlue-7) # Set the marker color of the lower limit uncertainties histogram

    eff1.SetTitle(label) # Set the title of the efficiency plot
    eff1.SetLineColor(ROOT.kBlack) # Set the line color of the first efficiency object

    h1.GetYaxis().SetRangeUser(ymin, ymax) # Set the y-axis range of the histogram
    h1.GetYaxis().SetTitle("Efficiency") # Set the y-axis title of the histogram
    h1.GetXaxis().SetTitle(xlabel) # Set the x-axis title of the histogram
    h1.GetXaxis().SetRangeUser(xmin, xmax) # Set the x-axis range of the histogram

    c.SetBottomMargin(0.12) # Set the bottom margin of the canvas

    h1.GetXaxis().SetTitleSize(0.055) # Set the x-axis title size of the histogram
    h1.GetYaxis().SetTitleSize(0.055) # Set the y-axis title size of the histogram
    h1.GetXaxis().SetLabelSize(0.05) # Set the x-axis label size of the histogram
    h1.GetYaxis().SetLabelSize(0.05) # Set the y-axis label size of the histogram
    h1.GetYaxis().SetTitleOffset(0.9) # Set the y-axis title offset of the histogram
    h1.GetXaxis().SetTitleOffset(0.98) # Set the x-axis title offset of the histogram

    h1.Draw("P*") # Draw the histogram with points
    h2.Draw("C same") # Draw the upper limit uncertainties histogram on top of the histogram
    h3.Draw("C same") # Draw the lower limit uncertainties histogram on top of the histogram

    if eff2 != None: # If the second efficiency object is not None, draw it
        eff2.SetLineColor(ROOT.kRed) # Set the line color of the second efficiency object
        eff2.Draw("same") # Draw the second efficiency object on top of the histogram
    
    eff1.Draw("same") # Draw the first efficiency object on top of the histogram

    if h4 != None: # If the fourth histogram is not None, draw it
        h4.SetLineColor(ROOT.kGreen) # Set the line color of the fourth histogram
        h4.SetMarkerColor(ROOT.kGreen) # Set the marker color of the fourth histogram
        h4.Draw("P* same") # Draw the fourth histogram with points on top of the histogram

    legend = TLegend(0.48, 0.13, 0.9, 0.63) # Create a legend to display the efficiency objects and histograms
    le1 = legend.AddEntry(eff1, legend1, "l") # Add the first efficiency object to the legend
    le3 = legend.AddEntry(h1, legend3, "l") # Add the histogram to the legend
    le5 = legend.AddEntry(h2, "GB means + STD", "l") # Add the upper limit uncertainties histogram to the legend
    le6 = legend.AddEntry(h3, "GB means - STD", "l") # Add the lower limit uncertainties histogram to the legend

    if eff2 != None: # If the second efficiency object is not None, add it to the legend
        le2 = legend.AddEntry(eff2, legend2, "l")
        le2.SetTextSize(0.045)

    if h4 != None: # If the fourth histogram is not None, add it to the legend
        le4 = legend.AddEntry(h4, legend4, "l")
        le4.SetTextSize(0.045)
    
    # Set text sizes for the legend entries
    le1.SetTextSize(0.045) 
    le3.SetTextSize(0.045)
    le5.SetTextSize(0.045)
    le6.SetTextSize(0.045)

    legend.SetBorderSize(0) # Set the border size of the legend to 0 (no border)

    legend.Draw("same") # Draw the legend on the canvas

    # Redraw CMS text elements to ensure they are visible
    latex = ROOT.TLatex() # Create a TLatex object for drawing text on the canvas
    latex.SetNDC() # Set the NDC (normalized device coordinates) for the TLatex object

    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(42) # Helvetica
    latex.SetTextSize(0.045) # Base size

    if sample == "DATA": # If the sample is "DATA", draw specific text
        latex.DrawLatex(0.5, 0.69, "2018 data")
    else:
        latex.DrawLatex(0.5, 0.69, sample)

    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(42) # Helvetica
    latex.SetTextSize(0.045) # Base size

    if sample == "DATA": # If the sample is "DATA", draw specific text
        latex.DrawLatex(0.5, 0.64, "Trained with 2018 data")
        latex.DrawLatex(0.66, 0.915, "59.9 fb^{-1}")
    else:
        latex.DrawLatex(0.5, 0.64, "Trained with QCD")

    latex.SetTextFont(61) # Set the text font to Helvetica Bold
    latex.SetTextSize(0.065) # Set the text size of the CMS label
    latex.DrawLatex(0.1, 0.915, "CMS") # Draw the CMS label on the canvas

    latex.SetTextFont(52) # Set the text font to Helvetica Italic
    latex.SetTextSize(0.047) # Set the text size of the preliminary label

    if sample == "DATA": # If the sample is "DATA", draw specific text
        latex.DrawLatex(0.21, 0.915, "Preliminary")
    else:
        latex.DrawLatex(0.21, 0.915, "Simulation Preliminary")

    latex.SetTextFont(42) # Set the text font to Helvetica
    latex.SetTextSize(0.047) # Set the text size of the (13 TeV) label
    latex.DrawLatex(0.78, 0.915, "(13 TeV)") # Draw the (13 TeV) label on the canvas

    # Construct the full path and save the file inside the dated folder
    full_path = os.path.join(plot_dir, name)
    c.SaveAs(full_path) # Save the canvas as a file with the specified name
    
    return c

def mean_efficiency(X_test, y_prob, bin_edges, range_min, range_max):
    """
    Calculates the mean efficiency within specified bins and saves the result into a histogram.

    Args:
        X_test (array-like): The test portion from a machine learning algorithm.
        y_prob (array-like): The probability for every test data point to pass the trigger.
        num_bins (integer): The number of bins.
        range_min (integer): The starting point of the x-axis range.
        range_max (integer): The end point of the x-axis range.

    Returns:
        h_test (TH1F): A histogram containing the resulting mean efficiencies.
    """

    num_bins = len(bin_edges) - 1 # Calculate the number of bins from the bin edges
    bin_width = (range_max - range_min) / num_bins # Calculate the width of each bin
    bin_centers = [edge + bin_width / 2 for edge in bin_edges[:-1]] # Calculate the center of each bin

    # If num_bins is a list and not an integer, use this. Otherwise uncomment commented lines
    bin_means, bin_edges, bin_number = binned_statistic(X_test, y_prob, statistic='mean', bins=bin_edges) #range = (range_min, range_max)
    bin_means = np.nan_to_num(bin_means) # Replace NaN values with 0 in the bin means

    # If num_bins is not a list but an integer, use h3 = TH1F("", "", num_bins, range_min, range_max)
    h3 = TH1F("", "", len(bin_edges) - 1, array('d', bin_edges)) # Create a histogram with the bin edges

    for i in range(1, num_bins + 1): # Loop through each bin
        h3.SetBinContent(i, bin_means[i-1]) # Set the bin content to the mean value for that bin

    h_test = h3.Clone() # Clone the histogram to create a new histogram object

    return h_test

def efficiency_plot_DATA(df, df_name, signal_trigger, reference_trigger, variable, var_name_plot, var_name, label, range_min, range_max, num_bins, y_min, y_max, file_name, suffix):
    """
    Creates efficiency plots for data based on user-defined signal and reference triggers.
    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)

    df_measured_pass = df.Filter(signal_trigger + " && " + reference_trigger, "") # Filter the dataframe for points that pass the signal trigger and reference trigger
    
    df_measured_all = df.Filter(reference_trigger, "") # Filter the dataframe for all points that pass the reference trigger

    eff_measured = efficiency_plot(df_measured_pass, df_measured_all, variable, var_name, "", "", suffix, save_plot=False) # Create the efficiency plot for the measured data

    ROOT.gStyle.SetOptStat(0) # Disable the statistics box
    ROOT.gStyle.SetTextFont(42) # Set the text font to Helvetica

    c = ROOT.TCanvas(" ", "", 900, 700) # Create a canvas to draw the histograms
    c.SetBottomMargin(0.12) # Set the bottom margin of the canvas

    eff_measured.SetLineColor(ROOT.kBlack) # Set the line color of the efficiency plot
    eff_measured.SetMarkerColor(ROOT.kBlack) # Set the marker color of the efficiency plot
    eff_measured.SetMarkerStyle(ROOT.kDot) # Set the marker style of the efficiency plot

    # Draw an empty histogram to set up the frame
    h_empty = ROOT.TH1F("h_empty", " ", len(num_bins) - 1, array('d', num_bins)) # Create an empty histogram with the specified number of bins
    h_empty.SetStats(0) # Disable the statistics box for the empty histogram
    h_empty.GetXaxis().SetTitle(var_name_plot) # Set the x-axis title of the empty histogram
    h_empty.GetYaxis().SetTitle("Efficiency") # Set the y-axis title of the empty histogram
    h_empty.GetXaxis().SetTitleSize(0.055) # Set the x-axis title size of the empty histogram
    h_empty.GetYaxis().SetTitleSize(0.055) # Set the y-axis title size of the empty histogram
    h_empty.GetXaxis().SetLabelSize(0.05) # Set the x-axis label size of the empty histogram
    h_empty.GetYaxis().SetLabelSize(0.05) # Set the y-axis label size of the empty histogram
    h_empty.GetYaxis().SetTitleOffset(0.9) # Set the y-axis title offset of the empty histogram
    h_empty.GetXaxis().SetTitleOffset(0.98) # Set the x-axis title offset of the empty histogram
    h_empty.Draw() # Draw the empty histogram to set up the frame
    eff_measured.Draw("same") # Draw the efficiency plot on top of the empty histogram

    # Redraw CMS text elements to ensure they are visible
    latex = ROOT.TLatex() # Create a TLatex object for drawing text on the canvas
    latex.SetNDC() # Set the NDC (normalized device coordinates) for the TLatex object

    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(42) # Helvetica
    latex.SetTextSize(0.05) # Base size
    latex.DrawLatex(0.6, 0.3, "HH 2018 data") # Draw the label text on the canvas

    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(42) # Helvetica
    latex.SetTextSize(0.045) # Base size
    latex.DrawLatex(0.66, 0.915, "59.8 fb^{-1}") # Draw the integrated luminosity text on the canvas

    latex.SetTextFont(61) # Set the text font to Helvetica Bold
    latex.SetTextSize(0.065) # Set the text size of the CMS label
    latex.DrawLatex(0.1, 0.915, "CMS") # Draw the CMS label on the canvas
    
    latex.SetTextFont(52) # Set the text font to Helvetica Italic
    latex.SetTextSize(0.047) # Set the text size of the preliminary label
    latex.DrawLatex(0.21, 0.915, "Preliminary") # Draw the preliminary label on the canvas

    latex.SetTextFont(42) # Set the text font to Helvetica
    latex.SetTextSize(0.047) # Set the text size of the (13 TeV) label
    latex.DrawLatex(0.78, 0.915, "(13 TeV)") # Draw the (13 TeV) label on the canvas

    c.SaveAs(os.path.join(plot_dir, file_name)) # Save the canvas as a file with the specified name

    return eff_measured

def draw_histogram(h, label, name, suffix):
    """
    Draws a histogram and saves it into a file.
    
    Args:
        h: A histogram object.
        label: Title for the plot.
        name: Filename 
        
    Returns:
        c: The canvas object that can be saved into a file.
    
    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)

    # Produce plot
    ROOT.gStyle.SetOptStat(0) # Disable the statistics box
    ROOT.gStyle.SetTextFont(42) # Set the text font to Helvetica
    c = ROOT.TCanvas("c", "", 800, 700) # Create a canvas to draw the histogram

    h.GetXaxis().SetTitleSize(0.04) # Set the x-axis title size of the histogram
    h.GetYaxis().SetTitleSize(0.04) # Set the y-axis title size of the histogram
    
    # Error bars can be drawn with Draw("E")
    h.Draw("E") # Draw the histogram with error bars
    h.SetTitle(label) # Set the title of the histogram

    c.SaveAs(os.path.join(plot_dir, name)) # Save the canvas as a file with the specified name
    
    return c

def draw_histograms_same(h_list, color_list, yaxis_range, legend_list, xtitle, ytitle, name, sample, suffix):
    """
    Draws user defined number of histograms into the same plot and saves it into a file.
    
    Args:
        h_list: Histogram list
        color_list: Colors for the histograms
        yaxis_range: Rnage for the y axis
        legend_list: List of the legends for the histograms
        xtitle: x axis name
        ytitle: y axis name
        name: The filename
        
    Returns:
        c: The canvas object.
    
    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)
    
    # Produce plot
    ROOT.gStyle.SetOptStat(0) # Disable the statistics box
    ROOT.gStyle.SetTextFont(42) # Set the text font to Helvetica

    c = ROOT.TCanvas("c", "", 900, 700) # Create a canvas to draw the histograms
    c.SetLeftMargin(0.15) # Set the left margin of the canvas
    c.SetBottomMargin(0.12) # Set the bottom margin of the canvas

    h_list[0].GetXaxis().SetTitleSize(0.055) # Set the x-axis title size of the first histogram
    h_list[0].GetYaxis().SetTitleSize(0.055) # Set the y-axis title size of the first histogram
    h_list[0].GetXaxis().SetLabelSize(0.05) # Set the x-axis label size of the first histogram
    h_list[0].GetYaxis().SetLabelSize(0.05) # Set the y-axis label size of the first histogram
    h_list[0].GetYaxis().SetTitleOffset(1.4) # Set the y-axis title offset of the first histogram
    h_list[0].GetXaxis().SetTitleOffset(0.98) # Set the x-axis title offset of the first histogram

    y_range_set = yaxis_range
    for i in range(len(h_list)):
        if y_range_set <= h_list[i].GetMaximum() + 500:
            y_range_set = h_list[i].GetMaximum() + 500

    # print(f"Y-axis range for {xtitle} feature is {y_range_set:.2f}")
    # print("-" * 40)
    h_list[0].GetYaxis().SetRangeUser(0, y_range_set) # Set the y-axis range of the first histogram

    h_list[0].GetXaxis().SetTitle(xtitle) # Set the x-axis title of the first histogram
    h_list[0].GetYaxis().SetTitle(ytitle) # Set the y-axis title of the first histogram

    for i in range(len(h_list)): # Loop through each histogram in the list
        h_list[i].SetLineColor(color_list[i]) # Set the line color of the histogram

    h_list[0].Draw("E") # Draw the first histogram with error bars

    for i in range(len(h_list)): # Loop through each histogram in the list
        h_list[i].Draw("E same") # Draw the histogram on top of the first histogram with error bars

    if xtitle == "#Delta#phi": # If the x-axis title is "#Delta#phi", adjust the x-axis range
         legend = TLegend(0.35, 0.44, 0.75, 0.86) # Create a legend to display the histograms
    else: # Otherwise, use a different legend position
        legend = TLegend(0.48, 0.44, 0.9, 0.86) # 0.1,0.7,0.3,0.9 # Create a legend to display the histograms

    for i in range(len(legend_list)): # Loop through each histogram in the legend list
        le=legend.AddEntry(h_list[i], legend_list[i], "l") # Add the histogram to the legend with the corresponding label
        legend.AddEntry(0, "", "") # Add an empty entry to the legend for spacing
        le.SetTextSize(0.04) # Set the text size of the legend entry

    legend.SetBorderSize(0) # Set the border size of the legend to 0 (no border)
    #legend.SetMargin(0.3) # Set the margin of the legend to 0.3 (optional, can be adjusted)
    #legend.SetEntrySeparation(0.5) # Set the entry separation of the legend to 0.5 (optional, can be adjusted)
    legend.Draw("same") # Draw the legend on the canvas
    
    # Redraw CMS text elements to ensure they are visible
    latex = ROOT.TLatex() # Create a TLatex object for drawing text on the canvas
    latex.SetNDC() # Set the NDC (normalized device coordinates) for the TLatex object
    
    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(61) # Helvetica Bold
    latex.SetTextSize(0.065) # Set the text size of the CMS label
    latex.DrawLatex(0.15, 0.915, "CMS") # Draw the CMS label on the canvas

    latex.SetTextFont(52) # Set the text font to Helvetica Italic
    latex.SetTextSize(0.047) # Set the text size of the preliminary label

    if sample == "DATA": # If the sample is "DATA", draw specific text
        latex.DrawLatex(0.26, 0.915, "Preliminary") 
    else: # Otherwise, use a different legend position
        latex.DrawLatex(0.26, 0.915, "Simulation Preliminary")

    latex.SetTextFont(42) # Set the text font to Helvetica
    latex.SetTextSize(0.047) # Set the text size of the (13 TeV) label
    latex.DrawLatex(0.78, 0.915, "(13 TeV)") # Draw the (13 TeV) label on the canvas
    
    c.SaveAs(os.path.join(plot_dir, name)) # Save the canvas as a file with the specified name
    
    return c

def numerator_and_denominator(df, df_name, filter_pass_real, filter_pass_meas, filter_all_meas, variable, var_name, xtitle, yaxis_range, run_name, suffix):
    """
    Creates plots of events in the trigger efficiency numerator and denominator based on user-defined signal and reference triggers.
    
    Args
        df: RDataFrame to be used
        df_name (string): Name of the dataframe
        signal_trigger (string): Signal trigger we use
        variable: 
        var_name (string): Name of the variable to plot
        xtitle: X-axis title
        yaxis_range: Y-axis range
        
    Returns
        num_real, denom_real: The histogram objects for events in the numerator and denominator
    
    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)
    
    if df_name == "DATA": # If the dataframe is for data, we need to filter the events based on the measurement triggers
        df_real_pass = df.Filter(filter_pass_meas, "") # Filter the dataframe for points that pass the measurement trigger
        df_real_all = df.Filter(filter_all_meas, "") # Filter the dataframe for all points that pass the measurement trigger

        N_signal = df_real_pass.Histo1D(variable,var_name) # Create a histogram for the numerator events
        N_signal = N_signal.GetValue() # Get the value of the histogram for the numerator events
        N_all = df_real_all.Histo1D(variable,var_name) # Create a histogram for the denominator events
        N_all = N_all.GetValue() # Get the value of the histogram for the denominator events
        
    else: # If the dataframe is not for data, we can directly filter the events based on the signal trigger
        df_real_pass = df.Filter(filter_pass_real, "") # Filter the dataframe for points that pass the signal trigger
        df_meas_all  = df.Filter(filter_all_meas, "") # Filter the dataframe for points that pass the reference trigger
        
        N_signal = df_real_pass.Histo1D(variable,var_name).GetValue() # Create a histogram for the numerator events
        N_all    = df.Histo1D(variable,var_name).GetValue() # Create a histogram for the denominator events
        # N_reference = df_meas_all.Histo1D(variable,var_name).GetValue()

    c2 = ROOT.TCanvas("c", "", 900, 700) # Create a canvas to draw the histograms
    c2.SetLeftMargin(0.15) # Set the left margin of the canvas
    c2.SetBottomMargin(0.12) # Set the bottom margin of the canvas

    N_all.GetXaxis().SetTitleSize(0.055) # Set the x-axis title size of the denominator histogram
    N_all.GetYaxis().SetTitleSize(0.055) # Set the y-axis title size of the denominator histogram
    N_all.GetXaxis().SetLabelSize(0.05) # Set the x-axis label size of the denominator histogram
    N_all.GetYaxis().SetLabelSize(0.05) # Set the y-axis label size of the denominator histogram
    N_all.GetXaxis().SetTitleOffset(0.98) # Set the x-axis title offset of the denominator histogram
    N_all.GetYaxis().SetTitleOffset(1.2) # Set the y-axis title offset of the denominator histogram

    # N_all.SetLineColor(ROOT.kAzure-5)
    N_signal.SetLineColor(ROOT.kRed)
    # N_all.SetLineColor(ROOT.kGreen+2)
    # N_signal.SetLineColor(ROOT.kOrange+2) # Set the line color of the numerator histogram    
    # if df_name != "DATA":
    #     N_reference.SetLineColor(ROOT.kViolet)

    all_c = N_all.Clone() # Clone the denominator histogram to create a new histogram object
    signal_c = N_signal.Clone() # Clone the numerator histogram to create a new histogram object
    # if df_name != "DATA":
    #     reference_c = N_reference.Clone()

    N_all.GetYaxis().SetTitle("Events") # Set the y-axis title of the denominator histogram
    N_all.GetXaxis().SetTitle(xtitle) # Set the x-axis title of the denominator histogram
    if yaxis_range <= N_all.GetMaximum():
        y_range_set = N_all.GetMaximum() + 500
    else:
        y_range_set = yaxis_range
    
    # print(f"Y-axis range for {df_name} data with {xtitle} feature is {y_range_set:.2f}")
    # print("-" * 40)
    N_all.GetYaxis().SetRangeUser(0, y_range_set) # Set the y-axis range of the denominator histogram

    N_all.Draw("E") # Draw the denominator histogram with error bars
    N_signal.Draw("E same") # Draw the numerator histogram on top of the denominator histogram with error bars
    # if df_name != "DATA":
    #     N_reference.Draw("E same")

    if xtitle == "#Delta#phi": # If the x-axis title is "#Delta#phi", adjust the legend position
        legend1 = TLegend(0.35, 0.44, 0.75, 0.86) # Create a legend to display the histograms
    else: # Otherwise, use a different legend position
        legend1 = TLegend(0.45, 0.6, 0.89, 0.89) # Create a legend to display the histograms

    if df_name == "DATA": # If the dataframe is for data, we need to add specific entries to the legend
        leg1=legend1.AddEntry(all_c,"#splitline{Events passing the}{reference trigger}","l") # Add the denominator histogram to the legend with a specific label
        leg2=legend1.AddEntry(signal_c,"#splitline{Events passing the signal}{& reference trigger}","l") # Add the numerator histogram to the legend with a specific label
    else: # If the dataframe is not for data, we can add different entries to the legend
        leg1=legend1.AddEntry(all_c,"All events","l") # Add the denominator histogram to the legend with a specific label
        leg2=legend1.AddEntry(signal_c,"#splitline{Events passing the}{signal trigger}","l") # Add the numerator histogram to the legend with a specific label
        # leg3=legend1.AddEntry(reference_c,"#splitline{Events passing the}{reference trigger}","l")
    
    leg1.SetTextSize(0.04) # Set the text size of the first legend entry
    leg2.SetTextSize(0.04) # Set the text size of the second legend entry
    # if df_name != "DATA":
    #     leg3.SetTextSize(0.04)

    legend1.SetBorderSize(0) # Set the border size of the legend to 0 (no border)
    legend1.Draw() # Draw the legend on the canvas

    N_all.SetStats(0) # Disable the statistics box for the denominator histogram
    
    # Redraw CMS text elements to ensure they are visible
    latex = ROOT.TLatex() # Create a TLatex object for drawing text on the canvas
    latex.SetNDC() # Set the NDC (normalized device coordinates) for the TLatex object
    
    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(61) # Helvetica Bold
    latex.SetTextSize(0.065) # Set the text size of the CMS label
    latex.DrawLatex(0.15, 0.915, "CMS") # Draw the CMS label on the canvas

    latex.SetTextFont(52) # Set the text font to Helvetica Italic
    latex.SetTextSize(0.047) # Set the text size of the preliminary label

    if df_name == "DATA": # If the dataframe is for data, we need to add specific entries to the legend
        latex.DrawLatex(0.26, 0.915, "Preliminary") # Draw the preliminary label on the canvas
        latex.SetTextFont(42)  # Helvetica
        latex.SetTextSize(0.045)  # Base size
        latex.DrawLatex(0.66, 0.915, "59.8 fb^{-1}") # Draw the integrated luminosity text on the canvas
    else: # If the dataframe is not for data, we can add different entries to the legend
        latex.DrawLatex(0.26, 0.915, "Simulation Preliminary") # Draw the simulation preliminary label on the canvas

    latex.SetTextFont(42) # Helvetica
    latex.SetTextSize(0.047) # Base size
    latex.DrawLatex(0.78, 0.915, "(13 TeV)") # Draw the (13 TeV) label on the canvas
    
    # c2.Update() # Update the canvas to reflect the changes

    file_name = f"{df_name}_{var_name}_{run_name}_both.png"
    c2.SaveAs(os.path.join(plot_dir, file_name)) # Save the canvas as a file with the specified name
    
    return N_signal, N_all

# This is for Santeri Laurila's peace of mind
def comparing_plot(df, sample, filter_pass_real, filter_pass_meas, filter_all_meas, variable, var_name, xtitle, run_name, suffix):
    # Get the directory path for today's plot
    plot_dir = get_plot_directory(suffix)

    # Filter the events based on the signal and reference trigger
    df_real_pass = df.Filter(filter_pass_real, "") # Signal trigger
    df_real_all  = df                              # All events
    df_meas_pass = df.Filter(filter_pass_meas, "") # Signal & Reference triggers
    df_meas_all  = df.Filter(filter_all_meas, "")  # Reference trigger

    # Create histogram for the different type of events
    N_signal    = df_real_pass.Histo1D(variable, var_name).GetValue() # No. Signal trigger
    N_all       = df_real_all.Histo1D(variable, var_name).GetValue()  # No. All events
    N_sigref    = df_meas_pass.Histo1D(variable, var_name).GetValue() # No. Signal & Referecne triggers
    N_reference = df_meas_all.Histo1D(variable, var_name).GetValue()  # No. Reference trigger

    # Comparison for measured efficiency
    c1 = ROOT.TCanvas("c1", "", 900, 700)
    c1.SetLeftMargin(0.15)
    c1.SetBottomMargin(0.12)

    N_reference.GetXaxis().SetTitleSize(0.055); N_reference.GetYaxis().SetTitleSize(0.055)
    N_reference.GetXaxis().SetLabelSize(0.05); N_reference.GetYaxis().SetLabelSize(0.05)
    N_reference.GetXaxis().SetTitleOffset(0.98); N_reference.GetYaxis().SetTitleOffset(1.2)

    N_reference.SetLineColor(ROOT.kGreen+2)
    N_sigref.SetLineColor(ROOT.kOrange+2)

    reference_c = N_reference.Clone()
    sigref_c = N_sigref.Clone()

    N_reference.GetYaxis().SetTitle("Events"); N_reference.GetXaxis().SetTitle(xtitle)
    N_reference.GetYaxis().SetRangeUser(0, N_reference.GetMaximum() + 300)

    N_reference.Draw("E")
    N_sigref.Draw("E same")

    if xtitle == "#Delta#phi":
        legend1 = TLegend(0.35, 0.44, 0.75, 0.86)
    else:
        legend1 = TLegend(0.45, 0.6, 0.89, 0.89)
    
    leg_ref = legend1.AddEntry(reference_c, "#splitline{Events passing the}{reference trigger}", "l")
    leg_sar = legend1.AddEntry(sigref_c, "#splitline{Events passing the signal}{& reference trigger}", "l")

    leg_ref.SetTextSize(0.04); leg_sar.SetTextSize(0.04)
    legend1.SetBorderSize(0)
    legend1.Draw()

    N_reference.SetStats(0)

    latex = ROOT.TLatex(); latex.SetNDC(); 
    latex.SetTextFont(61); latex.SetTextSize(0.065); latex.DrawLatex(0.15, 0.915, "CMS")
    latex.SetTextFont(52); latex.SetTextSize(0.047); latex.DrawLatex(0.26, 0.915, "Simulation Preliminary")
    latex.SetTextFont(42); latex.SetTextSize(0.047); latex.DrawLatex(0.78, 0.915, "(13 TeV)")

    c1.Update()

    # Comparison for real efficiency
    c2 = ROOT.TCanvas("c2", "", 900, 700)
    c2.SetLeftMargin(0.15)
    c2.SetBottomMargin(0.12)

    N_all.GetXaxis().SetTitleSize(0.055); N_all.GetYaxis().SetTitleSize(0.055)
    N_all.GetXaxis().SetLabelSize(0.05); N_all.GetYaxis().SetLabelSize(0.05)
    N_all.GetXaxis().SetTitleOffset(0.98); N_all.GetYaxis().SetTitleOffset(1.2)

    N_all.SetLineColor(ROOT.kAzure-5)
    N_signal.SetLineColor(ROOT.kRed+1)

    all_c = N_all.Clone()
    signal_c = N_signal.Clone()

    N_all.GetYaxis().SetTitle("Events"); N_all.GetXaxis().SetTitle(xtitle)
    N_all.GetYaxis().SetRangeUser(0, N_all.GetMaximum() + 300)

    N_all.Draw("E")
    N_signal.Draw("E same")

    if xtitle == "#Delta#phi":
        legend2 = TLegend(0.35, 0.44, 0.75, 0.86)
    else:
        legend2 = TLegend(0.45, 0.6, 0.89, 0.89)
    
    leg_all = legend2.AddEntry(all_c, "All events", "l")
    leg_sig = legend2.AddEntry(signal_c, "#splitline{Events passing the}{signal trigger}", "l")

    leg_all.SetTextSize(0.04); leg_sig.SetTextSize(0.04)
    legend2.SetBorderSize(0)
    legend2.Draw()

    N_all.SetStats(0)

    latex = ROOT.TLatex(); latex.SetNDC(); 
    latex.SetTextFont(61); latex.SetTextSize(0.065); latex.DrawLatex(0.15, 0.915, "CMS")
    latex.SetTextFont(52); latex.SetTextSize(0.047); latex.DrawLatex(0.26, 0.915, "Simulation Preliminary")
    latex.SetTextFont(42); latex.SetTextSize(0.047); latex.DrawLatex(0.78, 0.915, "(13 TeV)")

    c2.Update()

    file_name = f"{sample}_{var_name}_{run_name}_"
    c1.SaveAs(os.path.join(plot_dir, "meas_" + file_name + ".png"))
    c2.SaveAs(os.path.join(plot_dir, "real_" + file_name + ".png"))

    return N_signal, N_all, N_sigref, N_reference


def RD_to_pandas(df, trigger, columns, feature_names):
    """
    Creates a pandas dataframe out of RDataframe, takes only the events passing the reference trigger and returns the new dataframe and 
    a part of the dataframe containing only the feature columns.
    
    Args:
        df: RDataframe
        trigger: The reference trigger 
        columns: Columns from the original RDataframe we want to use in the new pandas dataframe
        feature_names: Names of the features as defined in the RDataframe
        
    Returns:
        pddf: The new pandas dataframe, where every event passes the reference trigger
        features: Pandas dataframe containing only the feature columns
    
    """

    # Filter the dataframe using the best reference trigger
    df_ref = df.Filter(trigger, "") # Filter the dataframe for points that pass the reference trigger

    # choose only the columns that are needed, since the datasets are so large that converting the whole dataframe will take a long time
    npy = df_ref.AsNumpy(columns=columns) # Convert the filtered dataframe to a NumPy array with the specified columns
    
    pddf = pd.DataFrame(npy) # Create a pandas DataFrame from the NumPy array
    features = pddf[feature_names] # Extract the feature columns from the DataFrame

    return pddf, features

# Function for performing the mean and std deviation calculation for the gradient boosting predicted efficiencies
def mean_and_standard_deviation(models, X, y, num_bins_list, range_min_list, range_max_list):
    """
    Calculates the mean and standard deviation of the predictions made by the bootstrapped gradient boosting model
    
    Args:
        models (list): The models from the bootstrapping
        X (array): Features to be used as input for the model 
        y (array): Labels to be used in calculating the accuracies and AUC-ROC scores
        num_bins_list (list): List of the number of bins for each variable to plot
        range_min_list: List of the minimum x axis range for each variable to plot
        range_max_list: List of the maximum x axis range for each variable to plot
    
    Returns:
        mean_list: List of the histograms for each variable to plot containing the mean values from the predictions made by the bootstrapping ensemble 
        mean_plus_list: List of the histograms for each variable to plot containing the mean + std values from the predictions made by the bootstrapping ensemble  
        mean_minus_list: List of the histograms for each variable to plot containing the mean - std values from the predictions made by the bootstrapping ensemble 
    
    """
    
    # Calculate test accuracies and AUC scores for the ensemble having n models
    auc_scores = []
    test_accuracies = []

    for i in range(len(models)): # Loop through each model in the ensemble
        test_predictions = models[i].predict(X) # Make predictions for the test data using the current model
        test_accuracy = accuracy_score(y, test_predictions) # Calculate the accuracy of the model on the test data
        test_accuracies.append(test_accuracy) # Append the accuracy to the list of test accuracies

        # Calculate AUC-ROC score for each model
        y_scores = models[i].predict_proba(X)[:, 1]  # Probability of positive class
        auc_score = roc_auc_score(y, y_scores) # Calculate the Area Under the Curve (AUC) for the ROC curve using the true labels and predicted probabilities
        auc_scores.append(auc_score) # Append the AUC score to the list of AUC scores

        # Print the test accuracy and AUC-ROC score for each model
        print(f"Model {i + 1} - Test Accuracy: {test_accuracy:.4f} - AUC-ROC Score: {auc_score:.4f}")

    avg_test_accuracy = np.mean(test_accuracies) # Calculate the average test accuracy across all models
    avg_auc_score = np.mean(auc_scores) # Calculate the average AUC-ROC score across all models
    print(f"\nAverage Test Accuracy across all models: {avg_test_accuracy:.4f}")
    print(f"Average AUC-ROC Score across all models: {avg_auc_score:.4f}")
    
    # Draw ROC curve for the ensemble
    plt.figure(figsize=(8, 6))

    # Calculate ROC curve for each model
    mean_fpr = np.linspace(0, 1, 100) # Create a mean false positive rate for the ROC curve
    tprs = [] # List to store true positive rates for each model

    for i in range(len(models)): # Loop through each model in the ensemble
        y_scores = models[i].predict_proba(X)[:, 1] # Probability of positive class
        fpr, tpr, _ = roc_curve(y, y_scores) # Calculate the false positive rate and true positive rate for the ROC curve using the true labels and predicted probabilities
        interp_tpr = np.interp(mean_fpr, fpr, tpr) # Interpolate the true positive rate to match the mean false positive rate
        interp_tpr[0] = 0.0 # Ensure the first value is 0.0 for the ROC curve
        tprs.append(interp_tpr) # Append the interpolated true positive rate to the list of true positive rates

    # Calculate mean and standard deviation of ROC curve
    mean_tpr = np.mean(tprs, axis=0) # Calculate the mean true positive rate across all models
    mean_tpr[-1] = 1.0 # Ensure the last value is 1.0 for the ROC curve
    std_tpr = np.std(tprs, axis=0) # Calculate the standard deviation of the true positive rate across all models

    plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC (AUC = {avg_auc_score:.2f})', lw=2, alpha=0.8) # Plot the mean ROC curve with the average AUC score

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='darkorange', alpha=0.8, label = 'Line for random guesses')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    
    # Calculate the means and standard deviations for each bin
    std_deviations = []
    mean_values = []
    pred_list = []

    for i in range(0,len(X)): 
        # Create an empty list to store predictions from each classifier for each data point
        individual_predictions = []

        # Loop through each Gradient Boosting classifier
        for model in models:
            # Make predictions for the single data point using the current classifier
            predictions = model.predict_proba([X[i]])[:,1] # Get the probability of the positive class (signal trigger) for the single data point
            individual_predictions.append(predictions) # Append the predictions to the list of individual predictions
        
        std_deviation = np.std(individual_predictions) # Calculate the standard deviation of the predictions for the single data point across all classifiers
        mean = np.mean(individual_predictions) # Calculate the mean of the predictions for the single data point across all classifiers
        std_deviations.append(std_deviation) # Append the standard deviation to the list of standard deviations
        mean_values.append(mean) # Append the mean to the list of mean values
        pred_list.append(individual_predictions) # Append the list of individual predictions for the single data point to the list of predictions

    
    # Create lists to store the mean + STD and mean - STD values
    means_plus = []
    means_minus =[]

    for i in range(0,len(mean_values)):
        mean_plus_std = mean_values[i] + std_deviations[i] # Calculate the mean + standard deviation for the single data point
        
        if mean_plus_std > 1.0: # Ensure the mean + standard deviation does not exceed 1.0
            mean_plus_std = 1.0
    
        means_plus.append(mean_plus_std) # Append the mean + standard deviation to the list of means plus
        mean_minus_std = mean_values[i] - std_deviations[i] # Calculate the mean - standard deviation for the single data point
        
        if mean_minus_std < 0.0: # Ensure the mean - standard deviation does not go below 0.0
            mean_minus_std = 0.0
    
        means_minus.append(mean_minus_std) # Append the mean - standard deviation to the list of means minus
    
    
    # Plot the means and STDs for all variables & add the histograms into lists    
    mean_list = []
    mean_plus_list = []
    mean_minus_list = []

    for i in range(0,len(num_bins_list)):    
        # Calculate the mean efficiency for each variable using the mean values and the specified number of bins and range
        h_gb_means = mean_efficiency(X[:,i], mean_values, num_bins_list[i], range_min_list[i], range_max_list[i])
        # Calculate the mean efficiency + standard deviation for each variable
        h_gb_means_plus = mean_efficiency(X[:,i], means_plus, num_bins_list[i], range_min_list[i], range_max_list[i]) 
        # Calculate the mean efficiency - standard deviation for each variable
        h_gb_means_minus = mean_efficiency(X[:,i], means_minus, num_bins_list[i], range_min_list[i], range_max_list[i]) 
        mean_list.append(h_gb_means)
        mean_plus_list.append(h_gb_means_plus)
        mean_minus_list.append(h_gb_means_minus)
    
    return mean_list, mean_plus_list, mean_minus_list

def open_models(filename, load_model=False, save_model=False, models=None):
    if os.path.exists(filename) and load_model:
        # If the file exists, load the models from it.
        print(f"File '{filename}' found. Loading existing model...")
        try:
            with open(filename, 'rb') as f:
                models = pickle.load(f)
            print("Model loaded successfully!")
            # You can now inspect the loaded model
            # print("Model details:", models)

            return models

        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading pickle file: {e}")
            print("The file might be corrupted or empty. Please consider regenerating it.")
            models = None # Set models to None to indicate failure

    elif save_model:
        # If the file does not exist, create the models and save them.
        print(f"File '{filename}' not found. Training a new model and saving it...")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the newly trained model to the file
        print(f"Saving new model to '{filename}'...")
        with open(filename, 'wb') as f:
            pickle.dump(models, f)
        print("New model saved successfully!")

# Function to be used when choosing a reference trigger
def real_and_measured_efficiency_plots(df, df_name, sample, signal_trigger, reference_trigger, variable,var_name_plot, var_name, label, range_min, range_max, num_bins, y_min, y_max, file_name, suffix):
    """
    Creates real and measured efficiency plots based on user-defined signal and reference triggers.
    """

    # Get the directory path for today's plots
    plot_dir = get_plot_directory(suffix)
    
    df_real_pass = df.Filter(signal_trigger, "") # Filter the dataframe for points that pass the signal trigger
    df_measured_pass = df.Filter(signal_trigger + " && " + reference_trigger, "") # Filter the dataframe for points that pass both the signal and reference triggers
    df_measured_all = df.Filter(reference_trigger, "") # Filter the dataframe for all points that pass the reference trigger
    eff_real = efficiency_plot(df_real_pass, df, variable, var_name, "", "", suffix, save_plot=False) # Create the real efficiency plot using the filtered data for the signal trigger and the original dataframe
    eff_measured = efficiency_plot(df_measured_pass, df_measured_all, variable, var_name, "", "", suffix, save_plot=False) # Create the measured efficiency plot using the filtered data for both the signal and reference triggers and the filtered data for all points that pass the reference trigger

    ROOT.gStyle.SetOptStat(0) # Disable the statistics box
    ROOT.gStyle.SetTextFont(42) # Set the text font to Helvetica
    c = ROOT.TCanvas(" ", "", 900, 700) # Create a canvas to draw the efficiency plots
    c.SetBottomMargin(0.12) # Set the bottom margin of the canvas

    eff_measured.SetLineColor(ROOT.kRed) # Set the line color of the measured efficiency plot to red
    eff_measured.SetMarkerColor(ROOT.kRed) # Set the marker color of the measured efficiency plot to red
    eff_measured.SetMarkerStyle(ROOT.kOpenCircle) # Set the marker style of the measured efficiency plot to open circle
    eff_real.SetLineColor(ROOT.kBlack) # Set the line color of the real efficiency plot to black
    eff_real.SetMarkerColor(ROOT.kBlack) # Set the marker color of the real efficiency plot to black
    eff_real.SetMarkerStyle(ROOT.kDot) # Set the marker style of the real efficiency plot to dot

    # Draw an empty histogram to set up the frame
    h_empty = ROOT.TH1F("h_empty", " ", len(num_bins) - 1, array('d', num_bins)) # Create an empty histogram with the specified number of bins
    h_empty.SetStats(0)  # Hide the statistics box
    h_empty.GetXaxis().SetTitle(var_name_plot) # Set the x-axis title of the empty histogram
    h_empty.GetYaxis().SetTitle("Efficiency") # Set the y-axis title of the empty histogram
    h_empty.GetYaxis().SetRangeUser(y_min,y_max) # Set the y-axis range of the empty histogram
    h_empty.GetXaxis().SetTitleSize(0.055) # Set the x-axis title size of the empty histogram
    h_empty.GetYaxis().SetTitleSize(0.055) # Set the y-axis title size of the empty histogram
    h_empty.GetXaxis().SetLabelSize(0.05) # Set the x-axis label size of the empty histogram
    h_empty.GetYaxis().SetLabelSize(0.05) # Set the y-axis label size of the empty histogram
    h_empty.GetYaxis().SetTitleOffset(0.9) # Set the y-axis title offset of the empty histogram
    h_empty.GetXaxis().SetTitleOffset(0.98) # Set the x-axis title offset of the empty histogram
    h_empty.Draw()

    # eff_real.SetTitle(label) # Set the title of the real efficiency plot
    eff_measured.Draw("same") # Draw the measured efficiency plot on top of the empty histogram
    eff_real.Draw("same") # Draw the real efficiency plot on top of the measured efficiency plot
    # eff_measured.Draw("same") # Draw the measured efficiency plot on top of the real efficiency plot

    legend = TLegend(0.45, 0.2, 0.88, 0.5)  # 0.1,0.7,0.3,0.9 x0, y0, x1, y1 # Create a legend to display the efficiency plots
    le1 = legend.AddEntry(eff_real, "True efficiency", "l") # Add the real efficiency plot to the legend with a specific label
    le2 = legend.AddEntry(eff_measured, "Measured efficiency", "l") # Add the measured efficiency plot to the legend with a specific label
    le1.SetTextSize(0.05) # Set the text size of the first legend entry
    le2.SetTextSize(0.05) # Set the text size of the second legend entry
    legend.SetBorderSize(0) # Set the border size of the legend to 0 (no border)
    legend.Draw() # Draw the legend on the canvas
   
    # Redraw CMS text elements to ensure they are visible
    latex = ROOT.TLatex() # Create a TLatex object for drawing text on the canvas
    latex.SetNDC() # Set the NDC (normalized device coordinates) for the TLatex object
    
    # Draw text inside the plot at specific coordinates
    latex.SetTextFont(42)  # Helvetica
    latex.SetTextSize(0.05)  # Base size

    if sample == "DATA": # If the sample is "DATA", we need to add specific entries to the legend
        latex.DrawLatex(0.45, 0.65, "2018 data")
    else: # If the sample is not "DATA", we can add different entries to the legend
        latex.DrawLatex(0.46, 0.49, sample)
    
    latex.SetTextFont(61) # Set the text font to Helvetica Bold
    latex.SetTextSize(0.065) # Set the text size of the CMS label
    latex.DrawLatex(0.1, 0.915, "CMS") # Draw the CMS label on the canvas

    latex.SetTextFont(52) # Set the text font to Helvetica Italic
    latex.SetTextSize(0.047) # Set the text size of the preliminary label

    if sample == "DATA": # If the sample is "DATA", we need to add specific entries to the legend
        latex.DrawLatex(0.21, 0.915, "Preliminary")
    else: # If the sample is not "DATA", we can add different entries to the legend
        latex.DrawLatex(0.21, 0.915, "Simulation Preliminary")
  
    latex.SetTextFont(42) # Set the text font to Helvetica
    latex.SetTextSize(0.047) # Set the text size of the (13 TeV) label
    latex.DrawLatex(0.78, 0.915, "(13 TeV)") # Draw the (13 TeV) label on the canvas
    
    c.SaveAs(os.path.join(plot_dir, file_name)) # Save the canvas as a file with the specified name
    
    return eff_real, eff_measured

def define_parameter(sample):
    # Ranges used for the plots of every variable in every sample
    if sample == "QCD":
        range_min_pt = 250
        range_max_pt = 1100
        num_bins_pt  = [250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,
                        420,430,440,450,460,470,480,490,500,520,540,560,580,600,620,640,660,
                        680,700,750,800,850,900,1000,1100]
        # num_bins_pt = np.arange(range_min_pt, range_max_pt + 10, 10)

        range_min_pt1 = 170
        range_max_pt1 = 1000
        num_bins_pt1 = [170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,
                        340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,
                        520,540,560,580,600,620,640,660,680,700,750,800,850,900,1000]
        # num_bins_pt1 = np.arange(range_min_pt1, range_max_pt1 + 10, 10)

        range_min_HT = 450
        range_max_HT = 2000
        num_bins_HT  = [440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,
                        780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,
                        1100,1120,1140,1160,1180,1200,1250,1300,1350,1400,1450,1500,1600,1700,
                        1800,1900,2000]
        # num_bins_HT = np.arange(range_min_HT, range_max_HT + 10, 10)

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,
                        200,210,220,230,240,250,260,270,280,290,300,320,340,360,380,400,450,500]
        # num_bins_MET = np.arange(range_min_MET, range_max_MET + 10, 10)

        range_min_mHH = 400
        range_max_mHH = 4000
        num_bins_mHH = [400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,
                        1200,1250,1300,1350,1400,1450,1500, 1600,1700,1800,1900,2000,2200,
                        2400,2600,2800,3000,3500,4000]
        # num_bins_mHH = np.arange(range_min_mHH, range_max_mHH + 10, 10)

        range_min_mass = 20
        range_max_mass = 300
        num_bins_mass  = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
                         210,220,230,240,250,260,270,280,290,300]
        # num_bins_mass = np.arange(range_min_mass, range_max_mass + 10, 10)

        range_min_METfj = 400
        range_max_METfj = 2500
        num_bins_METfj = [400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,
                          1200,1250,1300,1350,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
        # num_bins_METfj = np.arange(range_min_METfj, range_max_METfj + 10, 10)

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta  = [-3,-2,-1,0,1,2,3]

        range_min_delta = 0
        range_max_delta = 10
        num_bins_delta  = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        
        range_min_phi = -5
        range_max_phi = 5
        num_bins_phi  = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        
        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.5,-0.5,-0.6,-0.5]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,0.95,0.95,0.95,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]

    elif sample == "ggF":
        range_min_pt = 270
        range_max_pt = 1100
        num_bins_pt = [270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,
                       440,450,460,470,480,490,500,520,540,560,580,600,620,640,660,680,700,
                       750,800,850,900,1000,1100]
        # num_bins_pt = np.arange(range_min_pt, range_max_pt + 10, 10)

        range_min_pt1 = 250
        range_max_pt1 = 900
        num_bins_pt1 = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,
                        370,380,390,400,410,420,430,440,450,460,470,480,490,500,520,540,560,
                        580,600,620,640,660,680,700,750,800,850,900]
        # num_bins_pt1 = np.arange(range_min_pt1, range_max_pt1 + 10, 10)

        range_min_HT = 450
        range_max_HT = 2000
        num_bins_HT = [440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,
                       800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,
                       1120,1140,1160,1180,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,
                       1900,2000]
        # num_bins_HT = np.arange(range_min_HT, range_max_HT + 10, 10)

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
                        210,220,230,240,250,260,270,280,290,300,320,340,360,380,400,450,500]
        # num_bins_MET = np.arange(range_min_MET, range_max_MET + 10, 10)

        range_min_mHH = 550
        range_max_mHH = 2400
        num_bins_mHH = [600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                        1400,1450,1500,1600,1700,1800,1900,2000,2200,2400]
        # num_bins_mHH = np.arange(range_min_mHH, range_max_mHH + 10, 10)

        range_min_mass = 30
        range_max_mass = 220
        num_bins_mass = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230]
        # num_bins_mass = np.arange(range_min_mass, range_max_mass + 10, 10)

        range_min_METfj = 600
        range_max_METfj = 2500
        num_bins_METfj = [600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                          1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
        # num_bins_METfj = np.arange(range_min_METfj, range_max_METfj + 10, 10)

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta  = [-3,-2,-1,0,1,2,3]

        range_min_delta = 0
        range_max_delta = 3
        num_bins_delta  = [0,0.5,1,1.5,2,2.5,3]
        
        range_min_phi = -5
        range_max_phi = 8
        num_bins_phi  = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        
        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.5,-0.5,-0.6,-0.5]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,0.95,0.95,0.95,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]
        
    elif sample == "VBF":
        range_min_pt = 270
        range_max_pt = 1100
        num_bins_pt = [270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,
                       460,470,480,490,500,520,540,560,580,600,620,640,660,680,700,750,800,850,900,
                       1000,1100]
        # num_bins_pt = np.arange(range_min_pt, range_max_pt + 10, 10)

        range_min_pt1 = 250
        range_max_pt1 = 900
        num_bins_pt1  = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,
                         390,400,410,420,430,440,450,460,470,480,490,500,520,540,560,580,600,620,640,
                         660,680,700,750,800,850,900]
        # num_bins_pt1 = np.arange(range_min_pt1, range_max_pt1 + 10, 10)

        range_min_HT = 530
        range_max_HT = 2000
        num_bins_HT  = [520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,820,840,860,880,900,
                        920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,1200,1250,1300,
                        1350,1400,1450,1500,1600,1700,1800,1900,2000]
        # num_bins_HT = np.arange(range_min_HT, range_max_HT + 10, 10)

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,
                         220,230,240,250,260,270,280,290,300,320,340,360,380,400,450,500]
        # num_bins_MET = np.arange(range_min_MET, range_max_MET + 10, 10)

        range_min_mHH = 600
        range_max_mHH = 3000
        num_bins_mHH  = [550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,
                         1450,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000]
        # num_bins_mHH = np.arange(range_min_mHH, range_max_mHH + 10, 10)

        range_min_mass = 30
        range_max_mass = 300
        num_bins_mass  = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,
                          240,250,260,270,280,290,300]
        # num_bins_mass = np.arange(range_min_mass, range_max_mass + 10, 10)

        range_min_METfj = 550
        range_max_METfj = 2500
        num_bins_METfj  = [500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                           1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
        # num_bins_METfj = np.arange(range_min_METfj, range_max_METfj + 10, 10)

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta  = [-3,-2,-1,0,1,2,3]

        range_min_delta = 0
        range_max_delta = 4.5
        num_bins_delta  = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5]
        
        range_min_phi = -5
        range_max_phi = 5
        num_bins_phi  = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        
        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.5,-0.5,-0.6,-0.5]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,0.95,0.95,0.95,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]
        
    elif sample == "DATA":
        range_min_pt = 250
        range_max_pt = 1100
        num_bins_pt  = [250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,
                        430,440,450,460,470,480,490,500,520,540,560,580,600,620,640,660,680,700,
                        750,800,850,900,1000,1100]
        # num_bins_pt = np.arange(range_min_pt, range_max_pt + 10, 10)

        range_min_pt1 = 160
        range_max_pt1 = 1000
        num_bins_pt1  = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,
                         380,390,400,410,420,430,440,450,460,470,480,490,500,520,540,560,580,600,
                         620,640,660,680,700,750,800,850,900,1000]
        # num_bins_pt1 = np.arange(range_min_pt1, range_max_pt1 + 10, 10)

        range_min_HT = 380
        range_max_HT = 2000
        num_bins_HT  = [360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,
                        740,760,780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,
                        1100,1120,1140,1160,1180,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,1900,2000]
        # num_bins_HT = np.arange(range_min_HT, range_max_HT + 10, 10)

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,
                         220,230,240,250,260,270,280,290,300,320,340,360,380,400,450,500]
        # num_bins_MET = np.arange(range_min_MET, range_max_MET + 10, 10)

        range_min_mHH = 400
        range_max_mHH = 4000
        num_bins_mHH  = [400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,
                         1300,1350,1400,1450,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000,3500,4000]
        # num_bins_mHH = np.arange(range_min_mHH, range_max_mHH + 10, 10)

        range_min_mass = 20
        range_max_mass = 300
        num_bins_mass  = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,
                          240,250,260,270,280,290,300]
        # num_bins_mass = np.arange(range_min_mass, range_max_mass + 10, 10)

        range_min_METfj = 400
        range_max_METfj = 2500
        num_bins_METfj  = [500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                           1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
        # num_bins_METfj = np.arange(range_min_METfj, range_max_METfj + 10, 10)

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta  = [-3,-2,-1,0,1,2,3]

        range_min_delta = 0
        range_max_delta = 5
        num_bins_delta  = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        
        range_min_phi = -5
        range_max_phi = 5
        num_bins_phi  = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        
        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.5,-0.5,-0.6,-0.5]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,0.95,0.95,0.95,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]
        
    elif sample == "dist":
        range_min_pt = 250
        range_max_pt = 1600
        num_bins_pt  = [250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,
                       460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,
                       670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,
                       880,890,900,910,920,930,940,950,960,970,980,1000,1010,1020,1030,1040,1050,1060,1070,
                       1080,1090,1100,1110,1120,1130,1140,1150,1160,1170,1180,1190,1200,1210,1220,1230,1240,
                       1250,1260,1270,1280,1290,1300,1310,1320,1330,1340,1350,1360,1370,1380,1390,1400,1410,
                       1420,1430,1440,1450,1460,1470,1480,1490,1500,1510,1520,1530,1540,1550,1560,1570,1580,
                       1590,1600]

        range_min_pt1 = 150
        range_max_pt1 = 900
        num_bins_pt1  = [150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,
                         360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,
                         570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,
                         780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,
                         1000,1010,1020,1030,1040,1050,1060,1070,1080,1090,1100]

        range_min_HT = 400
        range_max_HT = 2400
        num_bins_HT  = [400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,
                        820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,
                        1200,1220,1240,1260,1280,1300,1320,1340,1360,1380,1400,1420,1440,1460,1480,1500,1520,
                        1540,1560,1580,1600,1620,1640,1660,1680,1700,1720,1740,1760,1780,1800,1820,1840,1860,
                        1880,1900,1920,1940,1960,1980,2000,2020,2040,2060,2080,2100,2120,2140,2160,2180,2200,
                        2220,2240,2260,2280,2300,2320,2340,2360,2380,2400,2420,2440,2460,2480,2500,2520,2540,
                        2560,2580,2600,2620,2640,2660,2680,2700]

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,
                         240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,
                         450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,
                         660,670,680,690,700,710,720,730,740,750,760,780,790,800]

        range_min_mHH = 400
        range_max_mHH = 3500
        num_bins_mHH  = [450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                         1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,
                         2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000,3050,
                         3100,3150,3200,3250,3300,3350,3400,3450,3500]

        range_min_mass = 30
        range_max_mass = 350
        num_bins_mass  = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,
                          250,260,270,280,290,300,310,320,330,340,350]

        range_min_METfj = 420
        range_max_METfj = 3000
        num_bins_METfj  = [420,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,
                           1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,
                           2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,
                           2950,3000]

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta  = [-3,-2,-1,0,1,2,3,4,5,6,7]

        range_min_delta = 0
        range_max_delta = 5
        num_bins_delta  = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        
        range_min_phi = -5
        range_max_phi = 5
        num_bins_phi  = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.3,-0.4,-0.6,-0.17]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]

    elif sample == "dist-DATA":
        range_min_pt = 250
        range_max_pt = 1500
        num_bins_pt  = [250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,
                        460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,
                        670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,
                        880,890,900,910,920,930,940,950,960,970,980,1000,1010,1020,1030,1040,1050,1060,1070,
                        1080,1090,1100,1110,1120,1130,1140,1150,1160,1170,1180,1190,1200,1210,1220,1230,1240,
                        1250,1260,1270,1280,1290,1300,1310,1320,1330,1340,1350,1360,1370,1380,1390,1400,1410,
                        1420,1430,1440,1450,1460,1470,1480,1490,1500]

        range_min_pt1 = 150
        range_max_pt1 = 900
        num_bins_pt1  = [150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,
                         360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,
                         570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,
                         780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,
                         1000,1010,1020,1030,1040,1050,1060,1070,1080,1090,1100]

        range_min_HT = 400
        range_max_HT = 2400
        num_bins_HT  = [400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,
                        820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,
                        1200,1220,1240,1260,1280,1300,1320,1340,1360,1380,1400,1420,1440,1460,1480,1500,1520,
                        1540,1560,1580,1600,1620,1640,1660,1680,1700,1720,1740,1760,1780,1800,1820,1840,1860,
                        1880,1900,1920,1940,1960,1980,2000,2020,2040,2060,2080,2100,2120,2140,2160,2180,2200,
                        2220,2240,2260,2280,2300,2320,2340,2360,2380,2400,2420,2440,2460,2480,2500]

        range_min_MET = 0
        range_max_MET = 500
        num_bins_MET  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,
                         240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400]

        range_min_mHH = 400
        range_max_mHH = 3500
        num_bins_mHH  = [450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,
                         1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,
                         2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000,3050,
                         3100,3150,3200,3250,3300,3350,3400,3450,3500]

        range_min_mass = 30
        range_max_mass = 350
        num_bins_mass  = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,
                          250,260,270,280,290,300]

        range_min_METfj = 420
        range_max_METfj = 3000
        num_bins_METfj = [420,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,
                          1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,
                          2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000]

        range_min_eta = -3
        range_max_eta = 3
        num_bins_eta = [-3,-2,-1,0,1,2,3,4,5,6,7]

        range_min_delta = 0
        range_max_delta = 5
        num_bins_delta = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        
        range_min_phi = -5
        range_max_phi = 5
        num_bins_phi = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

        # y_min_list = [0,0,0,0,0,0,0,0,0,0,-0.3,-0.4,-0.6,-0.17]
        # y_max_list = [1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05]

        # y_min_list = [-0.05 for t in range(20)]
        # y_max_list = [1.05 for t in range(20)]
        y_min_list = [-0.05 for t in range(14)]
        y_max_list = [1.05 for t in range(14)]
    
    # # Tau triggers
    # range_min_njet = 0
    # range_max_njet = 12
    # num_bins_njet = np.arange(range_min_njet, range_max_njet + 1, 1)

    # range_min_jetbal = 0
    # range_max_jetbal = 0.6
    # num_bins_jetbal = np.arange(range_min_jetbal, range_max_jetbal + 0.05, 0.05)

    # range_min_deltaMET = 0
    # range_max_deltaMET = 2
    # num_bins_deltaMET = np.arange(range_min_deltaMET, range_max_deltaMET + 0.05, 0.05)

    # Default values used in histogram plotting    
    pt      = ("FatJet_pt[0]", "; p_{T0} [GeV] ;Efficiency", len(num_bins_pt) - 1, array('d', num_bins_pt))

    HT      = ("HT", ";HT [GeV] ;Efficiency", len(num_bins_HT) - 1, array('d', num_bins_HT))

    MET     = ("MET", "; p_{T}^{miss} [GeV] ;Efficiency", len(num_bins_MET) - 1, array('d', num_bins_MET))

    mHH     = ("mHH", ";m_{HH} [GeV];Efficiency", len(num_bins_mHH) - 1, array('d', num_bins_mHH))

    mass    = ("Mass[0]", ";m_{0} [GeV] ;Efficiency", len(num_bins_mass) - 1, array('d', num_bins_mass))

    pt1     = ("FatJet_pt[1]", ";p_{T1} [GeV] ;Efficiency", len(num_bins_pt1) - 1, array('d', num_bins_pt1))

    mass1   = ("Mass[1]", ";m_{1} [GeV] ;Efficiency", len(num_bins_mass) - 1, array('d', num_bins_mass))

    FJHT    = ("FatJet HT", ";LR HT [GeV] ;Efficiency", len(num_bins_HT) - 1, array('d', num_bins_HT))

    METfj   = ("MET_FatJet", ";p_{T}^{miss}+p_{T0}+p_{T1} [GeV] ;Efficiency", len(num_bins_METfj) - 1, array('d', num_bins_METfj))

    mHHwMET = ("mHHwithMET", ";m_{HH}+p_{T}^{miss} [GeV];Efficiency", len(num_bins_mHH) - 1, array('d', num_bins_mHH))

    eta     = ("Eta[0]", ";η_{0} ;Efficiency", len(num_bins_eta) - 1, array('d', num_bins_eta))

    eta1    = ("Eta[1]", ";η_{1} ;Efficiency", len(num_bins_eta) - 1, array('d', num_bins_eta))

    deltaeta = ("DeltaEta", ";Δη ;Efficiency", len(num_bins_delta) - 1, array('d', num_bins_delta))

    deltaphi = ("DeltaPhi", ";Δφ ;Efficiency", len(num_bins_phi) - 1, array('d', num_bins_phi))

    # jetpt30 = ("nJet_pt30", ";p_{T30}; Efficiency", len(num_bins_njet) - 1, array('d', num_bins_njet))

    # maxmass = ("maxFatJetMass", ";m_{max} [GeV]; Efficiency", len(num_bins_mass) - 1, array('d', num_bins_mass))

    # jetbalance = ("FatJetBalance", ";p_{mid} [GeV]; Efficiency", len(num_bins_jetbal) - 1, array('d', num_bins_jetbal))

    # mindeltaphi = ("minDeltaPhiJetMET", ";∆φ + p_{T}^{miss} [GeV]; Efficiency", len(num_bins_deltaMET) - 1, array('d', num_bins_deltaMET))

    # taupt1 = ("TauPtLead", ";p_{τT1} [GeV]; Efficiency", len(num_bins_pt) - 1, array('d', num_bins_pt))

    # taupt2 = ("TauPtSubl", ";p_{τT2} [GeV]; Efficiency", len(num_bins_pt1) - 1, array('d', num_bins_pt1))

    # Define the variable list and names for the histograms
    variable_list = [pt, HT, MET, mHH, mass, pt1, mass1, FJHT, METfj, mHHwMET, eta, eta1, deltaeta, deltaphi, 
                    #  jetpt30, maxmass, jetbalance, mindeltaphi, taupt1, taupt2
                     ]

    names_list = ["HighestPt", "HT", "MET_pt", "mHH", "HighestMass", "SecondHighestPt", "SecondHighestMass", "FatHT", "MET_FatJet", "mHHwithMET", "HighestEta", "SecondHighestEta", "DeltaEta", "DeltaPhi", 
                #   "nJet_pt30", "maxFatJetMass", "FatJetBalance", "minDeltaPhiJetMET", "TauPt1st", "TauPt2nd"
                  ] 

    names_list_and_signal_trigger = ["HighestPt", "HT", "MET_pt", "mHH", "HighestMass", "SecondHighestPt", "SecondHighestMass", "FatHT", "MET_FatJet", "mHHwithMET", "HighestEta", "SecondHighestEta",
                                    "DeltaEta", "DeltaPhi", 
                                    # "nJet_pt30", "maxFatJetMass", "FatJetBalance", "minDeltaPhiJetMET", "TauPt1st", "TauPt2nd",
                                    "Combo"]

    names_list_plot = ["p_{T0} [GeV]", "H_{T} [GeV]", "p_{T}^{miss} [GeV]", "m_{HH} [GeV]", "m_{0} [GeV]", "p_{T1} [GeV]", "m_{1} [GeV]",
                    "H_{T}^{LR} [GeV]", "p_{T}^{miss}+p_{T0}+p_{T1} [GeV]", "m_{HH}+p_{T}^{miss} [GeV]", "#eta_{0}", "#eta_{1}", "#Delta#eta", "#Delta#phi", 
                    # "p_{T30}", "m_{max}", "p_{mid}", "#Delta#phi + p_{T}^{miss}", "p_{#tau 1}" , "p_{#tau 2}"
                    ]
    
    # Define the ranges and number of bins for each variable
    range_min_list = [range_min_pt, range_min_HT, range_min_MET, range_min_mHH, range_min_mass, range_min_pt1,
                    range_min_mass, range_min_HT, range_min_METfj, range_min_mHH, range_min_eta, range_min_eta, range_min_delta, range_min_phi,
                    # range_min_njet, range_min_mass, range_min_jetbal, range_min_deltaMET, range_min_pt, range_min_pt1
                    ]
    range_max_list = [range_max_pt, range_max_HT, range_max_MET, range_max_mHH, range_max_mass, range_max_pt1, range_max_mass, range_max_HT,
                    range_max_METfj, range_max_mHH, range_max_eta, range_max_eta, range_max_delta, range_max_phi,
                    # range_max_njet, range_max_mass, range_max_jetbal, range_max_deltaMET, range_max_pt, range_max_pt1
                    ]
    num_bins_list = [num_bins_pt, num_bins_HT, num_bins_MET, num_bins_mHH, num_bins_mass, num_bins_pt1, num_bins_mass, num_bins_HT,
                    num_bins_METfj, num_bins_mHH, num_bins_eta, num_bins_eta, num_bins_delta, num_bins_phi,
                    # num_bins_njet, num_bins_mass, num_bins_jetbal, num_bins_deltaMET, num_bins_pt, num_bins_pt1
                    ]
    
    return variable_list, names_list, names_list_and_signal_trigger, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list

def choosing_reference_trigger(sample, data, variable_list, names_list, names_list_plot, range_min_list, range_max_list, num_bins_list, y_min_list, y_max_list, signal_trigger, reference_trigger, run_name, suffix):
    if sample != "DATA":
        eff_real_list = []
        eff_meas_list = []

        for j in range(0, len(variable_list)):
            eff_real, eff_measured = real_and_measured_efficiency_plots(data, sample, sample, signal_trigger, reference_trigger, variable_list[j], 
                                                                                    names_list_plot[j], names_list[j], "", range_min_list[j], range_max_list[j], 
                                                                                    num_bins_list[j], y_min_list[j], y_max_list[j],
                                                                                    sample + "_TEfficiency_" + names_list[j] + "_" + reference_trigger + "_" + run_name + ".png", suffix)
                
            eff_real_list.append(eff_real)
            eff_meas_list.append(eff_measured)

        return eff_real_list, eff_meas_list
    
    elif sample == "DATA":
        eff_list_DATA = []

        for j in range(0,len(variable_list)):
            eff_DATA = efficiency_plot_DATA(data, sample, signal_trigger, reference_trigger, 
                                            variable_list[j], names_list_plot[j], names_list[j], "", 
                                            range_min_list[j], range_max_list[j], num_bins_list[j], 
                                            y_min_list[j], y_max_list[j],
                                            sample + "_TEfficiency_" + names_list[j] + "_" + reference_trigger + "_" + run_name + ".png", suffix)
            
            eff_list_DATA.append(eff_DATA)
        
        return eff_list_DATA


def train_method(data, names_list_and_signal_trigger, names_list, n_estimators, learning_rate, depth, random_state, filter_all_meas, n_samples):
    # Initial model training stage
    pddf, features = RD_to_pandas(data, filter_all_meas, names_list_and_signal_trigger, names_list)

    X = features.values
    y = pddf['Combo'].astype('int').values

    models   = []
    indice_x = []

    for i in range(n_samples):
        index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        indice_x.append(index_x)
        
        X_sample = X[index_x, :]
        y_sample = y[index_x]

        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_sample, y_sample, test_size=0.2, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=depth, random_state=random_state).fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_test)

        # Monitor the model's performance on the validation set for early stopping
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Iteration {i + 1} - Validation Accuracy: {val_accuracy:.4f}")
        if (i + 1 == 3) and val_accuracy <= 0.85:
            print("Accuracy low, try new models...")
            return models

    # Evaluation the final model(s) on the test set
    for i, model in enumerate(models):
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Model {i + 1} - Test Accuracy: {test_accuracy:.4f}")

    return models

def xgboost_train(data, names_list_and_signal_trigger, names_list, params: Dict[str, Any], config: Dict[str, Any], random_state, filter_all_meas, n_samples):
    # Initial model training stage
    pddf, features = RD_to_pandas(data, filter_all_meas, names_list_and_signal_trigger, names_list)

    X = features.values
    y = pddf['Combo'].astype('int').values

    models = []
    indice_x = []

    for i in range(n_samples):
        index_x = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        indice_x.append(index_x)

        X_sample = X[index_x, :]
        y_sample = y[index_x]

        # Split the data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X_sample, y_sample, test_size=0.5, random_state=random_state, stratify=y)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        print("\nStage 2: MODEL EVALUATION")
        print("=" * 50)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        models = []
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                max_depth=params.get('max_depth', 6),
                min_child_weight=params.get('min_child_weight', 1e-5),
                reg_lambda=params.get('reg_lambda', 1.0),
                subsample=params.get('subsample', 1.0),
                learning_rate=config['learning_rate'],
                n_estimators=config['max_iterations'],
                early_stopping_rounds=config['early_stopping'],
                random_state=random_state,
                eval_metric='logloss',
                verbosity=0
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            accuracy = accuracy_score(y_val, y_pred)
            auc_score = roc_auc_score(y_val, y_pred_proba)

            models.append(model)
            fold_scores.append({'accuracy': accuracy, 'auc': auc_score})

            print(f"Iteration {i + 1} - Validation Accuracy: {accuracy:.4f} & AUC: {auc_score:.4f}")

        test_results = {}

        

        

def applying_method(data, models, names_list_and_signal_trigger, names_list, num_bins_list, range_min_list, range_max_list, filter_all_meas, signal_trigger, n_samples):
    pddf, features = RD_to_pandas(data, filter_all_meas, names_list_and_signal_trigger, names_list)

    X = features.values
    y = pddf['Combo'].astype('int').values

    # Get the gradient boosting predicted efficiencies and uncertainties
    mean_list, mean_plus_list, mean_minus_list = mean_and_standard_deviation(models, X, y, num_bins_list, range_min_list, range_max_list)

    auc_scores = []
    test_accuracies = []

    for i in range(n_samples):
        test_predictions = models[i].predict(X)
        test_accuracy = accuracy_score(y, test_predictions)
        test_accuracies.append(test_accuracy)

        # Calculate AUC-ROC score for each model
        y_scores = models[i].predict_proba(X)[:, 1] # Probability of positive class
        auc_score = roc_auc_score(y, y_scores)
        auc_scores.append(auc_score)

    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc_score = np.mean(auc_scores)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i in range(n_samples):
        y_scores = models[i].predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_scores)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Calculate mean and standard deviation of ROC curve
    mean_tpr = np.mean(tprs, axis= 0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    return mean_list, mean_plus_list, mean_minus_list, avg_test_accuracy, avg_auc_score, mean_fpr, mean_tpr

def feature_ranking(sample, data, models, filter_all_meas, names_list_and_signal_trigger, names_list, suffix):
    plot_dir = get_plot_directory(suffix)

    # Initial model training stage
    pddf, features = RD_to_pandas(data, filter_all_meas, names_list_and_signal_trigger, names_list)

    X = features.values
    y = pddf['Combo'].astype('int').values

    # Draw the feature importance ranking plot
    custom_feature_names = [f'Feature {i}' for i in range(1, 11)]

    new_names = [r'$p_{T0}$', r'$H_{T}$', r'$p_{T}^{miss}$', r'$m_{HH}$', r'$m_{0}$',
                r'$p_{T1}$', r'$m_{1}$', r'$H_{T}^{LR}$', r'$p_{T}^{miss} + p_{T0} + p_{T1}$',
                r'$m_{HH} + p_{T}^{miss}$', r'$\eta_{0}$', r'$\eta_{1}$', r'$\Delta\eta$', r'$\Delta\phi$',
                # r'$p_{T30}$', r'$m_{mass}$', r'$p_{mid}$', r'$\Delta\phi + p_{T}^{miss}$', r'$p_{\tau 1}$', r'$p_{\tau 2}$'
                ]

    # Feature Importance Ranking
    feature_importance = np.zeros(X.shape[1])

    for model in models:
        feature_importance += model.feature_importances_

    average_feature_importance = feature_importance / len(models)

    # Sort features by importance
    sorted_indices = np.argsort(average_feature_importance)[::-1]
    sorted_features = [new_names[i] for i in sorted_indices]
    sorted_importance = average_feature_importance[sorted_indices]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), sorted_importance, align='center')
    plt.xticks(range(X.shape[1]), sorted_features, rotation=75, size=15)

    # Add text to the plot similar to ROOT
    plt.text(0, 1.02, "CMS", fontdict={'fontsize': 26, 'fontweight': 'bold'}, transform=plt.gca().transAxes)

    if sample == "DATA":
        plt.text(0.11, 1.02, "Preliminary", fontdict={'fontsize': 20, 'style': 'italic'}, transform=plt.gca().transAxes)
        plt.text(0.72, 1.02, "59.8 fb$^{-1}$", fontdict={'fontsize': 20, 'style': 'italic'}, transform=plt.gca().transAxes)
    else:
        plt.text(0.11, 1.02, "Simulation Preliminary", fontdict={'fontsize': 20, 'style': 'italic'}, transform=plt.gca().transAxes)

    plt.text(0.87, 1.0-2, "(13 TeV)", fontdict={'fontsize': 20}, transform=plt.gca().transAxes)
    plt.yticks(size=15)
    plt.xlabel('Features', size=15)
    plt.ylabel('Feature Importance', size=15)
    # plt.title('Feature Importance Ranking')!
    plt.tight_layout

    if sample == "DATA":
        file_name = "realdata-feature.png"
    else:
        file_name = "simulation-feature.png"
    plt.savefig(os.path.join(plot_dir, file_name))
    
    plt.show()
    
def ROC_curve_plotting(sample, mean_fpr_list, mean_tpr_list, avg_auc_score_list, suffix):
    plot_dir = get_plot_directory(suffix)

    # Draw ROC curve for the ensemble
    plt.figure(figsize=(8, 6))

    if sample == "DATA":
        plt.plot(mean_fpr_list["DATA"], mean_tpr_list["DATA"], color='b', label=f'ROC (AUC = {avg_auc_score_list["DATA"]:.2f})', lw=2, alpha=0.8)
    else:
        # Plot ROC curve for the ensemble
        plt.plot(mean_fpr_list["QCD"], mean_tpr_list["QCD"], color='b', label=f'QCD ROC (AUC = {avg_auc_score_list["QCD"]:.2f})', lw=2, alpha=0.8)

        # Plot ROC curve for model 1
        plt.plot(mean_fpr_list["ggF"], mean_tpr_list["ggF"], color='g', label=f'ggF ROC (AUC = {avg_auc_score_list["ggF"]:.2f})', lw=2, alpha=0.8)
        if 'VBF' in mean_fpr_list:
            # Plot ROC curve for model 2
            plt.plot(mean_fpr_list["VBF"], mean_tpr_list["VBF"], color='r', label=f'VBF ROC (AUC = {avg_auc_score_list["VBF"]:.2f})', lw=2, alpha=0.8)

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='darkorange', alpha=0.8, label='Line for random guesses')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Add text to the plot similar to ROOT
    plt.text(0, 1.02, "CMS", fontdict={'fontsize': 23, 'fontweight': 'bold'}, transform=plt.gca().transAxes)
    if sample == "DATA":
        plt.text(0.15, 1.02, "Preliminary", fontdict={'fontsize': 20, 'style': 'italic'}, transform=plt.gca().transAxes)
        plt.text(0.61, 1.02, "59.8 fb$^{-1}$", fontdict={'fontsize': 20}, transform=plt.gca().transAxes)
    else:
        plt.text(0.14, 1.02, "Simulation Preliminary", fontdict={'fontsize': 18, 'style': 'italic'}, transform=plt.gca().transAxes)
        plt.text(0.42,0.32, "Model trained with QCD applied \n to QCD, ggF and VBF",fontdict={'fontsize': 15})

    plt.text(0.83, 1.02, "(13 TeV)", fontdict={'fontsize': 18}, transform=plt.gca().transAxes)
    plt.legend(loc='lower right', fontsize=15)

    if sample == "DATA":
        file_name = "realdata-roc.png"
    else:
        file_name = "simulation-roc.png"
    plt.savefig(os.path.join(plot_dir, file_name))

    plt.show()
    