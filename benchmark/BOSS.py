#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import ContractableBOSSDilation

from sktime.benchmarking.data import UEADataset, make_datasets

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ### Settings ###

# In[9]:


fast_datasets = [
    # 'ACSF1',
    # 'Adiac',
    # 'AllGestureWiimoteX',
    # 'AllGestureWiimoteY',
    # 'AllGestureWiimoteZ',
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    # 'BME',
    "Car",
    "CBF",
    # 'Chinatown',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    "Coffee",
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'Crop',
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    # 'DodgerLoopDay',
    # 'DodgerLoopGame',
    # 'DodgerLoopWeekend',
    # 'Earthquakes',
    "ECG200",
    # 'ECG5000',
    "ECGFiveDays",
    # 'ElectricDevices',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    # 'FaceAll',
    # "FaceFour",
    # 'FacesUCR',
    # 'FiftyWords',
    # 'Fish',
    # 'FordA',
    # 'FordB',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'Fungi',
    # 'GestureMidAirD1',
    # 'GestureMidAirD2',
    # 'GestureMidAirD3',
    # 'GesturePebbleZ1',
    # 'GesturePebbleZ2',
    #####"Gun_Point",
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'Ham',
    # 'HandOutlines',
    # 'Haptics',
    # 'Herring',
    # 'HouseTwenty',
    # 'InlineSkate',
    # 'InsectEPGRegularTrain',
    # 'InsectEPGSmallTrain',
    # 'InsectWingbeatSound',
    "ItalyPowerDemand",
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MelbournePedestrian',
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    # 'Missing_value_and_variable_length_datasets_adjusted',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    "OliveOil",
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'PickupGestureWiimoteZ',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PLAID',
    "Plane",
    # 'PowerCons',
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    # 'RefrigerationDevices',
    # 'Rock',
    # 'ScreenType',
    # 'SemgHandGenderCh2',
    # 'SemgHandMovementCh2',
    # 'SemgHandSubjectCh2',
    # 'ShakeGestureWiimoteZ',
    # 'ShapeletSim',
    # 'ShapesAll',
    # 'SmallKitchenAppliances',
    # 'SmoothSubspace',
    ####"SonyAIBORobot Surface",
    ####"SonyAIBORobot SurfaceII",
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    "SyntheticControl",
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    "TwoLeadECG",
    # 'TwoPatterns',
    # 'UMD',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    "Wine",
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga'

    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "ECG200",
    # "ECGFiveDays",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxTW",
]


# In[10]:


all_datasets = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    'BME',
    "Car",
    "CBF",
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    "Coffee",
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    # 'DodgerLoopDay',
    # 'DodgerLoopGame',
    # 'DodgerLoopWeekend',
    'Earthquakes',
    "ECG200",
    'ECG5000',
    "ECGFiveDays",
    'ElectricDevices',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FaceAll',
    "FaceFour",
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    #'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    ####"Gun_Point",
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    "ItalyPowerDemand",
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    "OliveOil",
    'OSULeaf',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PLAID',
    "Plane",
    'PowerCons',
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    ###"SonyAIBORobot Surface",
    ###"SonyAIBORobot SurfaceII",
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    "SyntheticControl",
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    "TwoLeadECG",
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    "Wine",
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga'
]


# In[11]:


from typing import Dict, List


DATA_PATH = "./Univariate_ts"
datasets = make_datasets(
    path=DATA_PATH, dataset_cls=UEADataset, names=fast_datasets
)
# ["ArrowHead", "Car", "CBF", "Coffee"]
# ["ArrowHead"]
# fast_datasets
# all_datasets

# CBOSS_Dilation Hyperparameter:
# cboss_num_of_random_dilations = 100
# cboss_win_lengths = [32, 28, 24, 20, 16] #  picked randomly
# cboss_norm_options = [True, False]
# cboss_word_lengths = [8] 
# cboss_alphabet_size = 2
# cboss_feature_selection = "none" # {"chi2", "none", "random"} Sets the feature selections strategy to be used. Chi2 reduces the number of words significantly and is thus much faster (preferred). Random also reduces the number significantly. None applies not feature selection and yields large bag of words, e.g. much memory may be needed.
# cboss_max_feature_count = 256 # default=256, If feature_selection=random is chosen, this parameter defines the number of randomly chosen unique words used. 

# REMINDER: feature_count = num_of_random_dilations * _transformed_data.shape[1]
def generate_parameters():
    parameters = [
        [cboss_num_of_random_dilations_per_win_size, cboss_win_lengths, cboss_norm_options, cboss_word_lengths, cboss_alphabet_size, cboss_feature_selection, cboss_max_feature_count]
        for cboss_num_of_random_dilations_per_win_size in range(1, 6, 4) # default: Parameter existiert nicht
        for w, cboss_win_lengths in enumerate([[1]]) # default: Parameter existiert nicht
        for n, cboss_norm_options in enumerate([[True,False]]) # default: [True, False]
        for g, cboss_word_lengths in enumerate([[16, 14, 12, 10, 8]]) # default: [16, 14, 12, 10, 8]
        for k, cboss_alphabet_size in enumerate([4]) # default: 4
        for i, cboss_feature_selection in enumerate(["none"]) # default: "none"
        for p, cboss_max_feature_count in enumerate([256]) # default: 256
    ]
    return parameters



def generate_clfs(list_of_parameters):
    cboss_results_cols = [        
        "Classifier",
        "Dataset",
        "Accuracy",
        "Fit-Time",
        "Predict-Time",
        "total_feature_count",

        "num_of_random_dilations",
        "win_lengths",
        "norm_options",
        "word_lengths",
        "alphabet_size",
        "feature_selection",
        "max_feature_count"]

    # cboss_params = {
    #     "num_of_random_dilations": params[0], 
    #     "win_lengths": params[1], 
    #     "norm_options": params[2], 
    #     "word_lengths": params[3],
    #     "alphabet_size": params[4],
    #     "feature_selection": params[5],
    #     "max_feature_count": params[6],}
    
    clfs = [[ContractableBOSS(), "CBOSS", cboss_results_cols, ["NULL", "NULL", "[True, False]", "[16, 14, 12, 10, 8]", "4", "none", "256"]]]
    for params in list_of_parameters:

        cboss_dilation_params = {
            "num_of_random_dilations": params[0], 
            "win_lengths": params[1], 
            "norm_options": params[2], 
            "word_lengths": params[3],
            "alphabet_size": params[4],
            "feature_selection": params[5],
            "max_feature_count": params[6],}

        clfs.append([ContractableBOSSDilation(**cboss_dilation_params), "CBOSS_Dilation", cboss_results_cols, list(cboss_dilation_params.values())])

    return clfs

def show_boxplots(dfs: List[pd.DataFrame], benchmark_name: str, save_boxplots: bool):
    acc_dict = {}
    # fit_dict = {}
    # predict_dict = {}

    for i, result_df in enumerate(dfs):
        boxplot_name = result_df['num_of_random_dilations'][0]
        acc_dict[boxplot_name] = result_df['Accuracy']
        # fit_dict[boxplot_name] = result_df['Fit-Time']
        # predict_dict[boxplot_name] = result_df['Predict-Time']

    acc_dfs = pd.DataFrame(acc_dict)
    # fit_dfs = pd.DataFrame(fit_dict)
    # predict_dfs = pd.DataFrame(predict_dict)

    sns.set_style('white')
    sns.despine()
    sns.boxplot(data=acc_dfs).set_title('Accuracy')
    plt.xlabel("num_of_random_dilations_per_window")
    plt.ylabel("accuracy in percent")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_acc.png")
    plt.show()

    # sns.boxplot(data=fit_dfs).set_title('Fit-Time')
    # plt.xlabel("num_of_random_dilations_per_window")
    # plt.ylabel("fit-time in seconds")
    # if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_fit.png")
    # plt.show()

    # sns.boxplot(data=predict_dfs).set_title('Predict-Time')
    # plt.xlabel("num_of_random_dilations_per_window")
    # plt.ylabel("predict-time in seconds")
    # if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_predict.png")
    # plt.show()

def show_barcharts(dfs: List[pd.DataFrame], benchmark_name: str, save_barcharts: bool):
    acc_dict = {}
    fit_dict = {}
    predict_dict = {}

    for i, result_df in enumerate(dfs):
        boxplot_name = result_df['num_of_random_dilations'][0]
        acc_dict[boxplot_name] = result_df['Accuracy']
        fit_dict[boxplot_name] = result_df['Fit-Time']
        predict_dict[boxplot_name] = result_df['Predict-Time']

    acc_dfs = pd.DataFrame(acc_dict)
    fit_dfs = pd.DataFrame(fit_dict)
    predict_dfs = pd.DataFrame(predict_dict)

    sns.set_style('white')
    sns.despine()
    sns.barplot(data=acc_dfs).set_title('Mean Accuracy')
    plt.xlabel("num_of_random_dilations_per_window")
    plt.ylabel("mean accuracy in percent")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_av_acc.png")
    plt.show()

    sns.barplot(data=fit_dfs).set_title('Fit-Time')
    plt.xlabel("num_of_random_dilations_per_window")
    plt.ylabel("fit-time in seconds")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_av_fit.png")
    plt.show()

    sns.barplot(data=predict_dfs).set_title('Predict-Time')
    plt.xlabel("num_of_random_dilations_per_window")
    plt.ylabel("predict-time in seconds")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_av_predict.png")
    plt.show()


# ### Benchmark ###

# In[13]:


import benchmark
import os

benchmark_name= "testeein"
save_data = True
save_plots = True

os.mkdir("./results/" + benchmark_name)
parameters = generate_parameters()
clfs = generate_clfs(parameters)

all_results, all_results_av = benchmark.run(clfs=clfs,datasets=datasets, benchmark_name=benchmark_name, save_data=save_data)


# In[ ]:


show_boxplots(all_results, benchmark_name, save_boxplots=save_plots)
show_barcharts(all_results_av, benchmark_name=benchmark_name, save_barcharts=save_plots)

