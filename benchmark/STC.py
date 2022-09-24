#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifierDilation

from sktime.benchmarking.data import UEADataset, make_datasets


# ### Settings ###

# In[12]:


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


# In[13]:


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


# In[14]:


DATA_PATH = "./Univariate_ts"
datasets = make_datasets(
    path=DATA_PATH, dataset_cls=UEADataset, names=fast_datasets
)
# ["ArrowHead", "Car", "CBF", "Coffee"]
# ["ArrowHead"]
# all_datasets
# fast_datasets


# ShapeletTransformClassifierDilation Hyperparameter:
# n_shapelet_samples : int, default=10000
#     The number of candidate shapelets to be considered for the final transform.
#     Filtered down to <= max_shapelets, keeping the shapelets with the most
#     information gain.
# max_shapelet_length : int or None, default=None
#     Lower bound on candidate shapelet lengths for the transform.
# max_shapelets : int or None, default=None
#     Max number of shapelets to keep for the final transform. Each class value will
#     have its own max, set to n_classes / max_shapelets. If None uses the min between
#     10 * n_instances and 1000
# TODO n_shapelet_samples vs max_shapelets (dafür in den code schauen)


# sktime doc: "Then, given a set of k shapelets, a time series can be transformed into k features by calculating the distance from the series to each shapelet."
#REMINDER: feature_count = n_shapelet_samples OR max_shapelets ?? (each shapelet produces one feature for each time series. the distance)
def generate_parameters():
    parameters = [
        [stc_n_shapelet_samples, stc_max_shapelet_length, stc_max_shapelets]
        for a, stc_n_shapelet_samples in enumerate([10000, 15000, 20000, 25000, 30000])
        for b, stc_max_shapelet_length in enumerate([None, 7, 9, 11])
        for c, stc_max_shapelets in enumerate([None, 1000, 3000, 5000, 10000, 20000])
    ]
    return parameters



def generate_clfs(possible_parameters):
    stc_results_cols = [        
        "Classifier",
        "Dataset",
        "Accuracy",
        "Fit-Time",
        "Predict-Time",
        "total_feature_count",

        "n_shapelet_samples",
        "max_shapelet_length",
        "max_shapelets",
        ]

    clfs = [[ShapeletTransformClassifier(), "STC", stc_results_cols, ["10000", "None", "None"]]]
    for params in possible_parameters:

        stc_params = {
            "n_shapelet_samples": params[0], 
            "max_shapelet_length": params[1], 
            "max_shapelets": params[2], 
        }

        clfs.append([ShapeletTransformClassifierDilation(**stc_params), "STC_Dilation", stc_results_cols, list(stc_params.values())])

    return clfs


# ### Benchmark ###

# In[15]:


import benchmark

parameters = generate_parameters()
clfs = generate_clfs(parameters)

benchmark.run(clfs=clfs,datasets=datasets,benchmark_name="bulk_STC_fastdataset")

