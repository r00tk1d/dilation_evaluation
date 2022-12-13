import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score

from sktime.datasets import load_from_tsfile
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

from sklearn.preprocessing import LabelEncoder

from scipy.stats import zscore

import logging
logging.basicConfig(filename="./results/errors.log", level=logging.DEBUG, 
                format='%(asctime)s %(levelname)s %(name)s %(message)s')

def run(clfs, datasets, benchmark_name, save_data):
    all_results = []
    all_results_mean = []
    for i, clf in enumerate(clfs, start=1):
        results = pd.concat(Parallel(n_jobs=-1)(delayed(benchmark_clf)(clf, dataset)for dataset in datasets), ignore_index=True)
        #results = benchmark_clf(dataset=datasets[0], clf=clf) # for debugging with one dataset
        results_from_clf = results.loc[results["Classifier"] == clf[1]] # TODO wahrscheinlich nicht mehr nötig
        mean_acc = results_from_clf["Accuracy"].mean()
        mean_fit_time = results_from_clf["Fit-Time"].mean()
        mean_pred_time = results_from_clf["Predict-Time"].mean()
        try:
            mean_feature_count = results_from_clf["total_feature_count"].mean()
        except:
            mean_feature_count = -1.0
        results_mean = pd.DataFrame(columns=clf[2])
        results_mean.loc[len(results_mean)] = [clf[1], "ALL_AVERAGE", mean_acc, mean_fit_time, mean_pred_time, mean_feature_count] + clf[3]

        all_results.append(results)
        all_results_mean.append(results_mean)

        # print(results_mean[['Classifier', 'Accuracy', 'Fit-Time', 'Predict-Time', 'total_feature_count']])
        # print(f"clf {i}/{len(clfs)} done")
        results_mean = results_mean[0:0]
        results = results[0:0]
    if(save_data):
        save_results(all_results, all_results_mean, benchmark_name)
    return all_results, all_results_mean


def benchmark_clf(clf, dataset):
    result = pd.DataFrame(columns=clf[2])
    logger=logging.getLogger(__name__)
    try:
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
        try:
            feature_count = clf[0].feature_count
        except:
            feature_count = -1.0
        result.loc[len(result)] = [clf[1], dataset.name, acc, fit_time, predict_time, feature_count] + clf[3]
        
        #print(f"clf {clf[1]} dataset {dataset.name} done")
    except Exception as e:
        print(f"ERROR {e} for dataset {dataset.name} clf {clf[1]}")
        logger.error(f"ERROR {e} for dataset {dataset.name} clf {clf[1]}")
    return result

def save_results(all_results, all_results_av, benchmark_name):
    # if not os.path.isfile("./results/" + benchmark_name + ".csv"):
    #     results.to_csv("./results/" + benchmark_name + ".csv", header=True)
    # else:
    #     results.to_csv("./results/" + benchmark_name + ".csv", mode='a', header=False)
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv("./results/" + benchmark_name + "/" + benchmark_name + ".csv", header=True)
    all_results_av_df = pd.concat(all_results_av)
    all_results_av_df.to_csv("./results/" + benchmark_name + "/" + benchmark_name + "_av.csv", header=True)