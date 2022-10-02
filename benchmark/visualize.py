from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# for visualizing the whole data
def boxplots(dfs: List[pd.DataFrame], benchmark_name: str, save_boxplots: bool):
    acc_dict = {}
    feature_count_dict = {}

    for i, result_df in enumerate(dfs):
        boxplot_name = result_df['n_shapelet_samples'][0] # hier immer den Parameter wählen der variiert wird
        acc_dict[boxplot_name] = result_df['Accuracy']
        feature_count_dict[boxplot_name] = result_df['total_feature_count']

    acc_dfs = pd.DataFrame(acc_dict)
    feature_count_dfs = pd.DataFrame(feature_count_dict)

    sns.set_style('white')
    sns.despine()
    sns.boxplot(data=acc_dfs).set_title('Accuracy')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("accuracy in percent")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_acc.png")
    plt.show()
    plt.clf()

    sns.boxplot(data=feature_count_dfs).set_title('Feature Count')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("total_feature_count")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_feature_count.png")
    plt.show()
    plt.clf()

# for visualizing overall mean numbers 
def barplots(dfs: List[pd.DataFrame], benchmark_name: str, save_barcharts: bool):
    acc_dict = {}
    fit_dict = {}
    predict_dict = {}
    feature_count_dict = {}

    for i, result_df in enumerate(dfs):
        barplot_name = result_df['n_shapelet_samples'][0] # hier immer den Parameter wählen der variiert wird
        acc_dict[barplot_name] = result_df['Accuracy']
        fit_dict[barplot_name] = result_df['Fit-Time']
        predict_dict[barplot_name] = result_df['Predict-Time']
        feature_count_dict[barplot_name] = result_df['total_feature_count']

    acc_dfs = pd.DataFrame(acc_dict)
    fit_dfs = pd.DataFrame(fit_dict)
    predict_dfs = pd.DataFrame(predict_dict)
    feature_count_dfs = pd.DataFrame(feature_count_dict)

    sns.set_style('white')
    sns.despine()
    sns.barplot(data=acc_dfs).set_title('Mean Accuracy')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("mean accuracy in percent")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_acc.png")
    plt.show()
    plt.clf()

    sns.barplot(data=fit_dfs).set_title('Mean Fit-Time')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("fit-time in seconds")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_fit.png")
    plt.show()
    plt.clf()

    sns.barplot(data=predict_dfs).set_title('Mean Predict-Time')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("predict-time in seconds")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_predict.png")
    plt.show()
    plt.clf()

    sns.barplot(data=feature_count_dfs).set_title('Mean Feature Count')
    plt.xlabel("n_shapelet_samples") # hier immer den Parameter wählen der variiert wird
    plt.ylabel("features")
    if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_feature_count.png")
    plt.show()
    plt.clf()