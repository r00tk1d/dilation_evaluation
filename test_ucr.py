# -*- coding: utf-8 -*-
"""UCR test."""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import WEASEL2

sys.path.append("../../..")


def load_from_ucr_tsv_to_dataframe_plain(full_file_path_and_name):
    """Load UCR datasets."""
    df = pd.read_csv(
        full_file_path_and_name,
        sep=r"\s+|\t+|\s+\t+|\t+\s+",
        engine="python",
        header=None,
    )
    y = df.pop(0).values
    df.columns -= 1
    return df, y


dataset_names_excerpt = [
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
    "Gun_Point",
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
    "SonyAIBORobot Surface",
    "SonyAIBORobot SurfaceII",
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    "synthetic_control",
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

others = [

]

DATA_PATH = "/Users/bzcschae/workspace/similarity/datasets/classification/"

if __name__ == "__main__":
    scores = []

    for dataset_name in dataset_names_excerpt:

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST")
        )

        # z-norm training/test data
        # X_train = X_train.to_numpy()
        # X_test = X_test.to_numpy()
        X_train = zscore(X_train, axis=1).to_numpy()
        X_test = zscore(X_test, axis=1).to_numpy()

        df_train = pd.DataFrame()
        df_train["dim_0"] = [pd.Series(x) for x in X_train]

        df_test = pd.DataFrame()
        df_test["dim_0"] = [pd.Series(x) for x in X_test]

        clfs = [
            # WEASEL(
            #    random_state=1379,
            #    window_inc=1,
            #    bigrams=True,
            #    anova=True,
            #    n_jobs=4),
            WEASEL2(
                random_state=1379,
                binning_strategy="equi-depth",
                anova=True,
                bigrams=False,
                window_inc=1,
                p_threshold=1,
                n_jobs=4,
            ),
            WEASEL2(
                random_state=1379,
                binning_strategy="equi-width",
                anova=True,
                bigrams=False,
                window_inc=1,
                p_threshold=1,
                n_jobs=4,
            ),
            WEASEL2(
                random_state=1379,
                # binning_strategy="equi-width",
                anova=True,
                bigrams=False,
                window_inc=1,
                p_threshold=1,
                n_jobs=4,
            ),
        ]

        # print(f"\nDataset: {dataset_name, np.shape(X_train)}")
        for clf in clfs:
            fit_time = time.process_time()
            clf.fit(df_train, y_train)
            fit_time = np.round(time.process_time() - fit_time, 5)

            pred_time = time.process_time()
            y_pred = clf.predict(df_test)
            pred_time = np.round(time.process_time() - pred_time, 5)

            acc = np.round(accuracy_score(y_test, y_pred), 5)


            # print(f"time={fit_time, pred_time}")
            # print(f"\taccuracy_score={acc}")
            #feature_names = len(clf.clf.named_steps["dictvectorizer"].feature_names_)
            # print(f"\tfeature count={feature_names}")
            #acc_score = clf.clf.named_steps["logisticregressioncv"].scores_
            #acc_score_agg = np.array(list(acc_score.values())).mean(axis=0).mean(axis=1)
            #best_index = acc_score_agg.argmax()
            #best_score = acc_score_agg[best_index]
            # print(f"\tval-acc={best_score}")


            scores.append((clf, dataset_name, acc, fit_time, pred_time))
            pd.DataFrame.from_records(
                scores,
                columns=[
                    "Classifier",
                    "Dataset",
                    "Accuracy",
                    "Fit-Time",
                    "Predict-Time",
                ],
            ).to_csv("scores.csv", index=None)
