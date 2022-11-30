from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# for visualizing the whole data
def boxplots(dfs: List[pd.DataFrame], benchmark_name: str, save_boxplots: bool, base_column: str):
    acc_dict = {}
    feature_count_dict = {}
    fit_dict = {}
    predict_dict = {}

    for i, result_df in enumerate(dfs):
        boxplot_name = result_df[base_column][0] # hier immer den Parameter wählen der variiert wird
        acc_dict[boxplot_name] = result_df['Accuracy']
        feature_count_dict[boxplot_name] = result_df['total_feature_count']
        fit_dict[boxplot_name] = result_df['Fit-Time']
        predict_dict[boxplot_name] = result_df['Predict-Time']

    acc_dfs = pd.DataFrame(acc_dict)
    feature_count_dfs = pd.DataFrame(feature_count_dict)
    fit_dfs = pd.DataFrame(fit_dict)
    predict_dfs = pd.DataFrame(predict_dict)

    sns.set_style('white')
    sns.set(font_scale=1.8)
    sns.boxplot(data=acc_dfs).set_title('Accuracy')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("accuracy in percent")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_acc.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.boxplot(data=feature_count_dfs).set_title('Feature Count')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("features")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_feature_count_box.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.barplot(data=feature_count_dfs, estimator=np.mean, capsize=.2).set_title('Feature Count')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("features")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_feature_count_bar.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.boxplot(data=fit_dfs).set_title('Fit-Time')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("fit-time in seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_fit.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.barplot(data=fit_dfs, estimator=np.mean, capsize=.2).set_title('Fit-Time')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("fit-time in seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_fit_bar.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.boxplot(data=predict_dfs).set_title('Predict-Time')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("predict-time in seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_predict.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.barplot(data=predict_dfs, estimator=np.mean, capsize=.2).set_title('Predict-Time')
    plt.xticks(rotation=90)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("predict-time in seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_predict_bar.png", bbox_inches="tight")
    plt.show()
    plt.clf()

def times(dfs: List[pd.DataFrame], benchmark_name: str, save_boxplots: bool, base_column: str):
    get_intervals_time_dict = {}
    transform_time_dict = {}
    fit_randomforest_time_dict = {}

    for i, result_df in enumerate(dfs):
        boxplot_name = result_df[base_column][0] # hier immer den Parameter wählen der variiert wird
        get_intervals_time_dict[boxplot_name] = result_df['get_intervals_time']
        transform_time_dict[boxplot_name] = result_df['transform_time']
        fit_randomforest_time_dict[boxplot_name] = result_df['fit_randomforest_time']

    get_intervals_time_dfs = pd.DataFrame(get_intervals_time_dict)
    transform_time_dfs = pd.DataFrame(transform_time_dict)
    fit_randomforest_time_dfs = pd.DataFrame(fit_randomforest_time_dict)


    sns.set_style('white')
    sns.set(font_scale=1.8)

    sns.barplot(data=get_intervals_time_dfs, estimator=np.mean, capsize=.2).set_title('get_intervals')
    plt.xticks(rotation=90)
    plt.ylim(0, 2)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_time_get_intervals.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.barplot(data=transform_time_dfs, estimator=np.mean, capsize=.2).set_title('transform_time')
    plt.xticks(rotation=90)
    plt.ylim(0, 2)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_time_transform.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    sns.barplot(data=fit_randomforest_time_dfs, estimator=np.mean, capsize=.2).set_title('fit_randomforest_time')
    plt.xticks(rotation=90)
    plt.ylim(0, 2)
    plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
    plt.ylabel("seconds")
    if(save_boxplots): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_time_fit_randomforest.png", bbox_inches="tight")
    plt.show()
    plt.clf()

# for visualizing overall mean numbers from the _av csv files (currently not used)
# def barplots(dfs: List[pd.DataFrame], benchmark_name: str, save_barcharts: bool, base_column: str):
#     acc_dict = {}
#     fit_dict = {}
#     predict_dict = {}
#     feature_count_dict = {}

#     for i, result_df in enumerate(dfs):
#         barplot_name = result_df[base_column][0] # hier immer den Parameter wählen der variiert wird
#         acc_dict[barplot_name] = result_df['Accuracy']
#         fit_dict[barplot_name] = result_df['Fit-Time']
#         predict_dict[barplot_name] = result_df['Predict-Time']
#         feature_count_dict[barplot_name] = result_df['total_feature_count']

#     acc_dfs = pd.DataFrame(acc_dict)
#     fit_dfs = pd.DataFrame(fit_dict)
#     predict_dfs = pd.DataFrame(predict_dict)
#     feature_count_dfs = pd.DataFrame(feature_count_dict)

#     sns.set_style('white')
#     sns.despine()
#     sns.barplot(data=acc_dfs).set_title('Mean Accuracy')
#     plt.xticks(rotation=90)
#     plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
#     plt.ylabel("mean accuracy in percent")
#     if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_acc.png", bbox_inches="tight")
#     plt.show()
#     plt.clf()

#     sns.barplot(data=fit_dfs).set_title('Mean Fit-Time')
#     plt.xticks(rotation=90)
#     plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
#     plt.ylabel("fit-time in seconds")
#     if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_fit.png", bbox_inches="tight")
#     plt.show()
#     plt.clf()

#     sns.barplot(data=predict_dfs).set_title('Mean Predict-Time')
#     plt.xticks(rotation=90)
#     plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
#     plt.ylabel("predict-time in seconds")
#     if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_predict.png", bbox_inches="tight")
#     plt.show()
#     plt.clf()

#     sns.barplot(data=feature_count_dfs).set_title('Mean Feature Count')
#     plt.xticks(rotation=90)
#     plt.xlabel(base_column) # hier immer den Parameter wählen der variiert wird
#     plt.ylabel("features")
#     if(save_barcharts): plt.savefig("./results/" + benchmark_name + "/" + benchmark_name + "_mean_feature_count.png", bbox_inches="tight")
#     plt.show()
#     plt.clf()