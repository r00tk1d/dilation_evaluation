#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import time
import numpy as np
import pandas as pd

from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import BOSSEnsembleDilation
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifierDilation
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifierDilation

from sklearn.metrics import accuracy_score

from sktime.benchmarking.data import UEADataset, make_datasets

from convst.classifiers import R_DST_Ridge
from convst.utils.dataset_utils import load_sktime_dataset_split

from sktime.datasets import load_from_tsfile
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV

from scipy.stats import zscore


# ### Settings ###

# In[14]:


used_datasets = [
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

    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ECG200",
    "ECGFiveDays",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
]


# In[15]:


DATA_PATH = "./Univariate_ts"
datasets = make_datasets(
    path=DATA_PATH, dataset_cls=UEADataset, names=used_datasets
)
# ["ArrowHead", "Car", "CBF", "Coffee"]
# ["ArrowHead"]

# clfs hyperparameter:
# BOSSEnsembleDilation:
boss_num_of_random_dilations = 200
boss_window_sizes = [9, 11] # laut shapeletDilation Paper ist [11] am besten. Mini Rocket nutzt nur [9]. picked randomly
boss_norm_options = [True, False] # picked randomly
boss_word_lengths = [8, 4] # TODO herausfinden was genau dieser parameter setzt
# TODO dilation eventuell anders umsetzen als ROCKET?  random(1, (series.length - 1)/2) hatte gute Ergebnisse

boss_results_cols = [        
        "Classifier",
        "Dataset",
        "Accuracy",
        "Fit-Time",
        "Predict-Time",
        "num_of_random_dilations",
        "window_sizes",
        "norm_options",
        "word_length",
        "dilation_algo",]



# TSFDilation:
tsf_min_interval = 3
tsf_n_intervals_prop = 0.8
# TODO relevant? tsf_num_of_random_dilations = 200

tsf_results_cols = [        
        "Classifier",
        "Dataset",
        "Accuracy",
        "Fit-Time",
        "Predict-Time",
        "min_interval",
        "n_intervals_prop",]

#[ClassifierFunction, ClassifierName, result_col_names, hyperparameterForResultsCSV]
clfs = [
    #[BOSSEnsemble(), "BOSS", boss_results_cols, ["NULL", "10 to max", "True and False", "[16, 14, 12, 10, 8]", "NULL"]],
    [BOSSEnsembleDilation(num_of_random_dilations=boss_num_of_random_dilations, window_sizes = boss_window_sizes, norm_options=boss_norm_options, word_lengths=boss_word_lengths), "BOSS_Dilation", boss_results_cols, \
     [str(boss_num_of_random_dilations), str(boss_window_sizes), str(boss_norm_options), str(boss_word_lengths), "ROCKET"]],
    #[TimeSeriesForestClassifier(), "TimeSeriesForest", tsf_results_cols, [3, 1]],
    #[TimeSeriesForestClassifierDilation(min_interval=tsf_min_interval, n_intervals_prop=tsf_n_intervals_prop), "TimeSeriesForest_Dilation",  tsf_results_cols, \
    #    [str(tsf_min_interval), str(tsf_n_intervals_prop)]],
    
    #[ShapeletTransformClassifier(estimator=RidgeClassifierCV()), "ShapeletTransform+RidgeClassifier", shapelet_results_cols, ["DEFAULT"]], 
    #[ShapeletTransformClassifier(), "ShapeletTransform", shapelet_results_cols, ["DEFAULT"]], # uses RandomForest from sktime
    #[ShapeletTransformClassifierDilation(), "ShapeletTransformDilation", shapelet_results_cols, ["EDIT ME"]],
    #[R_DST_Ridge(), "RDST", RDST_results_cols, "DEFAULT"], # uses RidgeClassifierCV from scikitlearn
]


# ### Benchmark ###

# In[16]:



for clf in clfs:
    results = pd.DataFrame(columns=clf[2])
    for dataset in datasets:
        X_train, y_train = load_from_tsfile(dataset._train_path)
        X_test, y_test = load_from_tsfile(dataset._test_path)

        X_train = from_nested_to_3d_numpy(X_train)
        X_test = from_nested_to_3d_numpy(X_test)

        # Convert class labels to make sure they are between 0,n_classes
        le = LabelEncoder().fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        
        # z normalize data
        #X_train = zscore(X_train, axis=1)
        #X_test = zscore(X_test, axis=1)

        fit_time = time.process_time()
        clf[0].fit(X_train, y_train)
        fit_time = np.round(time.process_time() - fit_time, 5)

        predict_time = time.process_time()
        y_pred = clf[0].predict(X_test)
        predict_time = np.round(time.process_time() - predict_time, 5)

        acc = np.round(accuracy_score(y_test, y_pred), 5)
        
        results.loc[len(results)] = [clf[1], dataset.name, acc, fit_time, predict_time] + clf[3]
        
        print(f"clf {clf[1]} dataset {dataset.name} done")

    results_from_clf = results.loc[results["Classifier"] == clf[1]]
    av_acc = results_from_clf["Accuracy"].mean()
    av_fit_time = results_from_clf["Fit-Time"].mean()
    av_pred_time = results_from_clf["Predict-Time"].mean()

    av_results = pd.DataFrame(columns=clf[2])
    av_results.loc[len(av_results)] = [clf[1], "ALL_AVERAGE", av_acc, av_fit_time, av_pred_time] + clf[3]

    results = pd.concat([results, av_results], ignore_index=True)

    if not os.path.isfile("./results/" + clf[1] + "_results.csv"):
        results.to_csv("./results/" + clf[1] + "_results.csv", header=True)
    else:
        results.to_csv("./results/" + clf[1] + "_results.csv", mode='a', header=False)
    print(f"clf {clf[1]} done")
    print(av_results[['Classifier', 'Accuracy', 'Fit-Time', 'Predict-Time']])
    results = results[0:0]
    av_results = av_results[0:0]

