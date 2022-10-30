#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifierDilation

from sktime.benchmarking.data import UEADataset, make_datasets


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
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
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
    'Fungi',
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


# ### Benchmark ###

# In[4]:


DATA_PATH = "./Univariate_ts"
datasets = make_datasets(
    path=DATA_PATH, dataset_cls=UEADataset, names=fast_datasets
)
# ["ArrowHead", "Car", "CBF", "Coffee"]
# ["ArrowHead"]
# all_datasets
# fast_datasets


# TimeSeriesForest_Dilation Hyperparameter:
# tsf_n_intervals_prop = 1.0 # Anzahl an Intervallen (hier vergrößern damit mehr features entstehen)
# tsf_interval_length_prop = 1.0 # dient dazu die window size zu reduzieren (also verkleinern)
# tsf_num_of_random_dilations = 1
# tsf_n_estimators=200


#REMINDER: feature_count = 3 * n_intervals (=sqrt(series_length)) * num_of_random_dilations
def generate_parameters():
    parameters = [
        [tsf_n_intervals_prop, tsf_interval_length_prop, tsf_interval_lengths, tsf_dilations_per_interval, tsf_n_estimators, tsf_n_intervals]
        for a, tsf_n_intervals_prop in enumerate([1.0]) # default: Parameter existiert nicht (1.0)
        for f, tsf_n_intervals in enumerate([30]) # default: Parameter existiert nicht (0)
        for d, tsf_dilations_per_interval in enumerate([1]) # default: Parameter existiert nicht

        for b, tsf_interval_length_prop in enumerate([1.0, 0.8, 0.6, 0.4, 0.2])  # default: Parameter existiert nicht (1.0) (toggle in sktime repo)
        for c, tsf_interval_lengths in enumerate([0]) # default: Parameter existiert nicht (toggle in sktime repo) (min_interval (3) muss hier dabei sein)

        for e, tsf_n_estimators in enumerate([200])  # default: 200, The number of trees in the forest.
        
    ]
    return parameters

def generate_clfs(possible_parameters):
    tsf_results_cols = [        
        "Classifier",
        "Dataset",
        "Accuracy",
        "Fit-Time",
        "Predict-Time",
        "total_feature_count",
        "get_intervals_time", 
        "transform_time", 
        "fit_randomforest_time",
        
        "n_intervals_prop",
        "interval_length_prop",
        "interval_lengths",
        "dilations_per_interval",
        "n_estimators",
        "n_intervals"]
    clfs = [[TimeSeriesForestClassifier(), "TSF", tsf_results_cols, [-1.0, -1.0, "base clf", -1.0, 200, -1]]] #
    for params in possible_parameters:

        tsf_params = {
            "n_intervals_prop": params[0],
            "interval_length_prop": params[1],
            "interval_lengths": params[2],
            "dilations_per_interval": params[3],
            "n_estimators": params[4],
            "n_intervals": params[5]}

        clfs.append([TimeSeriesForestClassifierDilation(**tsf_params), "TSF_Dilation", tsf_results_cols, list(tsf_params.values())])

    return clfs


# In[5]:


import benchmark_tsf
import os

benchmark_name = "TSF_DILATION5_interval_length_prop"
save_data = True
save_plots = True

os.mkdir("./results/" + benchmark_name)
parameters = generate_parameters()
clfs = generate_clfs(parameters)

all_results, all_results_mean = benchmark_tsf.run(clfs=clfs,datasets=datasets, benchmark_name=benchmark_name, save_data=save_data)


# In[6]:


# DEPRECATED: Wird jetzt separat in einem extra notebook visualisiert
# import visualize
# visualize.boxplots(all_results, benchmark_name=benchmark_name, save_boxplots=save_plots, base_column=base_column)
# visualize.barplots(all_results_mean, benchmark_name=benchmark_name, save_barcharts=save_plots, base_column=base_column)

