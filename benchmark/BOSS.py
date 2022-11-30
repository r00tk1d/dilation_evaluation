#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import ContractableBOSSDilation

from sktime.benchmarking.data import UEADataset, make_datasets

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ### Settings ###

# In[2]:


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


# In[3]:


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
    'DodgerLoopDay', #
    'DodgerLoopGame', #
    'DodgerLoopWeekend', #
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
    'Fungi', #
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint', #
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
    "SonyAIBORobotSurface1", #
    "SonyAIBORobotSurface2", #
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


# In[4]:


from typing import Dict, List


DATA_PATH = "./Univariate_ts"
datasets = make_datasets(
    path=DATA_PATH, dataset_cls=UEADataset, names=all_datasets
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
        [cboss_dilations_per_param_comb, cboss_win_lengths, cboss_norm_options, cboss_word_lengths, cboss_alphabet_size, cboss_feature_selection, cboss_max_feature_count, cboss_max_win_len_prop]
        for w, cboss_win_lengths in enumerate([[0]]) # default: Parameter existiert nicht (Nutzung in sktime togglen)
        for u, cboss_max_win_len_prop in enumerate([0.6]) # default: 1 (Nutzung in sktime togglen)

        for a, cboss_dilations_per_param_comb in enumerate([1]) # default: Parameter existiert nicht
        for n, cboss_norm_options in enumerate([[True,False]]) # default: [True, False]
        for g, cboss_word_lengths in enumerate([[8]]) # default: [16, 14, 12, 10, 8]
        for k, cboss_alphabet_size in enumerate([4]) # default: 4
        for i, cboss_feature_selection in enumerate(["chi2"]) # default: "none"
        for p, cboss_max_feature_count in enumerate([256]) # default: 256 | wird nur von feature_selection = random genutzt
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

        "dilations_per_param_comb",
        "win_lengths",
        "norm_options",
        "word_lengths",
        "alphabet_size",
        "feature_selection",
        "max_feature_count",
        "max_win_len_prop"]

    
    clfs = [[ContractableBOSS(), "CBOSS", cboss_results_cols, [0, -1, [True, False], [16, 14, 12, 10, 8], 4, "none", 256, -1.0]]] #
    for params in list_of_parameters:

        cboss_dilation_params = {
            "dilations_per_param_comb": params[0], 
            "win_lengths": params[1], 
            "norm_options": params[2], 
            "word_lengths": params[3],
            "alphabet_size": params[4],
            "feature_selection": params[5],
            "max_feature_count": params[6],
            "max_win_len_prop": params[7]}

        clfs.append([ContractableBOSSDilation(**cboss_dilation_params), "CBOSS_Dilation", cboss_results_cols, list(cboss_dilation_params.values())])

    return clfs


# ### Benchmark ###

# In[5]:


import benchmark
import os

benchmark_name= "CBOSS_DILATION_UCR_best_params"
save_data = True

if save_data: os.mkdir("./results/" + benchmark_name)
parameters = generate_parameters()
clfs = generate_clfs(parameters)


all_results, all_results_mean = benchmark.run(clfs=clfs,datasets=datasets, benchmark_name=benchmark_name, save_data=save_data)


