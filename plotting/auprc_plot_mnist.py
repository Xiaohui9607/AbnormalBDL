import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
dir = ""
AnoGAN =    [0.612, 0.292, 0.541, 0.442, 0.431, 0.426, 0.486, 0.399, 0.412, 0.359] # check
EGBAD =     [0.782, 0.298, 0.663, 0.511, 0.458, 0.429, 0.571, 0.385, 0.521, 0.342] # check
Ganormaly = [0.881, 0.661, 0.951, 0.796, 0.809, 0.868, 0.859, 0.671, 0.653, 0.533] # check
# SkipGan =   [0.448, 0.953, 0.607, 0.602, 0.615, 0.931, 0.788, 0.797, 0.659, 0.907]
Proposed =  [0.996, 0.996, 0.985, 0.928, 0.995, 0.946, 0.977, 0.984, 0.961, 0.999]
mnist_cls = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def read_csv_to_auc_mean(path, n=5, dataset_cls=mnist_cls):
    ret = [0]*10
    exps = glob.glob(os.path.join(path, "*"))
    for exp in exps:
        file_list = glob.glob(os.path.join(exp,"*.csv"))
        best_auc = 0
        for file in file_list:
            mean=pd.read_csv(file)
            mean_n=np.array(mean[mean.labels==0]['scores'])
            mean_ab=np.array(mean[mean.labels==1]['scores'])
            means = np.concatenate([mean_n, mean_ab])
            labels = np.concatenate([np.zeros_like(mean_n),np.ones_like(mean_ab)])
            best_auc = max(best_auc, roc_auc_score(labels, means))
        for i, cls in enumerate(dataset_cls):
            if cls == exp.split('/')[-1].split('_')[-2]:
                ret[i] = max(best_auc, ret[i])
    return ret

# m_n = "/mnt/AbnormalResult/M_N_mnist"
# m_n_result = read_csv_to_auc_mean(m_n)
# print(m_n_result)

x = mnist_cls
fig = plt.figure(figsize=(2200.0/300.0,1400.0/300.0))
plt.plot(x, AnoGAN, 'ro-', label='AnoGAN')
plt.plot(x, EGBAD, 'y.-', label='EGBAD')
plt.plot(x, Ganormaly, 'g+-', label='Ganormaly')
plt.plot(x, Proposed, 'bx-', label='Proposed')
plt.ylim(0,1)
plt.xlabel('class designated as anomalous class')
plt.ylabel('AUROC')
plt.legend(ncol=4, columnspacing=3, frameon=False, bbox_to_anchor=(0, 0.08, 1, 0))
plt.savefig("comparison_auroc_mnist.png", dpi=300)
