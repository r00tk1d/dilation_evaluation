import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score

from sktime.datasets import load_from_tsfile
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

from sklearn.preprocessing import LabelEncoder

from scipy.stats import zscore

def run(clfs, datasets, benchmark_name, save_data):
    all_results = []
    all_results_av = []
    for i, clf in enumerate(clfs, start=1):
        results = pd.concat(Parallel(n_jobs=-1)(delayed(benchmark_clf)(clf, dataset)for dataset in datasets), ignore_index=True)
        #benchmark_clf(dataset=datasets[0], clf=clf) # for debugging with one dataset
        results_from_clf = results.loc[results["Classifier"] == clf[1]]
        av_acc = results_from_clf["Accuracy"].mean()
        av_fit_time = results_from_clf["Fit-Time"].mean()
        av_pred_time = results_from_clf["Predict-Time"].mean()
        # TODO num_of_datasets = len(results_from_clf.index)
        try:
            av_feature_count = results_from_clf["total_feature_count"].mean()
        except:
            av_feature_count = "NULL"
        av_results = pd.DataFrame(columns=clf[2])
        av_results.loc[len(av_results)] = [clf[1], "ALL_AVERAGE", av_acc, av_fit_time, av_pred_time, av_feature_count] + clf[3]

        all_results.append(results)
        all_results_av.append(av_results)
        # results = pd.concat([results, av_results], ignore_index=True)

        # if not os.path.isfile("./results/" + benchmark_name + ".csv"):
        #     results.to_csv("./results/" + benchmark_name + ".csv", header=True)
        # else:
        #     results.to_csv("./results/" + benchmark_name + ".csv", mode='a', header=False)

        print(av_results[['Classifier', 'Accuracy', 'Fit-Time', 'Predict-Time', 'total_feature_count']])
        print(f"clf {i}/{len(clfs)} done")
        av_results = av_results[0:0]
        results = results[0:0]
    if(save_data):
        save_results(all_results, all_results_av, benchmark_name)
    show_boxplots(all_results_av)


def benchmark_clf(clf, dataset):
    result = pd.DataFrame(columns=clf[2])
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
            feature_count = "NULL"
        result.loc[len(result)] = [clf[1], dataset.name, acc, fit_time, predict_time, feature_count] + clf[3]
        
        #print(f"clf {clf[1]} dataset {dataset.name} done")
    except Exception as e:
        print(f"ERROR {e} for dataset {dataset.name} clf {clf[1]}")
    return result

def save_results(all_results, all_results_av, benchmark_name):
    # if not os.path.isfile("./results/" + benchmark_name + ".csv"):
    #     results.to_csv("./results/" + benchmark_name + ".csv", header=True)
    # else:
    #     results.to_csv("./results/" + benchmark_name + ".csv", mode='a', header=False)
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv("./results/" + benchmark_name + ".csv", header=True)
    all_results_av_df = pd.concat(all_results_av)
    all_results_av_df.to_csv("./results/" + benchmark_name + "_av.csv", header=True)

def show_boxplots(list_of_dfs):
    acc_dict = {}
    fit_dict = {}
    predict_dict = {}

    for i, result_df in enumerate(list_of_dfs):
        acc_dict[i] = result_df['Accuracy']
        fit_dict[i] = result_df['Fit-Time']
        predict_dict[i] = result_df['Predict-Time']

    acc_dfs = pd.DataFrame(acc_dict)
    fit_dfs = pd.DataFrame(fit_dict)
    predict_dfs = pd.DataFrame(predict_dict)

    sns.set_style('white')
    sns.despine()
    sns.boxplot(data=acc_dfs).set_title('Accuracy')
    plt.show()

    sns.boxplot(data=fit_dfs).set_title('Fit-Time')
    plt.show()

    sns.boxplot(data=predict_dfs).set_title('Predict-Time')
    plt.show()