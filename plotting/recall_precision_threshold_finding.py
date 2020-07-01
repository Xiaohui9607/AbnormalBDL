import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc

def get_metric_and_best_threshold_from_pr_curve(
    precision, recall, thresholds, num_pos_class, num_neg_class
):
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold

csv_path = "/home/golf/code/AbnormalBDL/exp3G3Dexp_0epoch_score_train.csv"
mean_pd = pd.read_csv(csv_path)
precisions, recalls, thresholds = precision_recall_curve(mean_pd.labels.values,mean_pd.scores.values)

_, threshold = get_metric_and_best_threshold_from_pr_curve(precisions, recalls, thresholds,
                                            mean_pd.loc[mean_pd.labels == 0].values.shape[0],
                                            mean_pd.loc[mean_pd.labels == 1].values.shape[0])

idx = np.where(thresholds==threshold)[0]
precision = precisions[idx]
recall = recalls[idx]
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", 2*precision*recall/(precision+recall))
