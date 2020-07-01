import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
dir = ""
AnoGAN =    [0.411, 0.492, 0.399, 0.335, 0.393, 0.321, 0.399, 0.516, 0.567, 0.511]
EGBAD =     [0.383, 0.514, 0.448, 0.374, 0.481, 0.353, 0.526, 0.577, 0.413, 0.555]
Ganormaly = [0.510, 0.631, 0.587, 0.593, 0.628, 0.683, 0.605, 0.633, 0.616, 0.617]
SkipGan =   [0.448, 0.953, 0.607, 0.602, 0.615, 0.931, 0.788, 0.797, 0.659, 0.907]
Proposed =  [0.564, 0.865, 0.748, 0.777, 0.820, 0.928, 0.800, 0.972, 0.913, 0.785]
cifar_cls = ['bird', 'automobile', 'cat', 'deer', 'dog', 'frog', 'horse', 'airplane',  'ship', 'truck']


def read_csv_to_auc_mean(path, n=5, dataset_cls=cifar_cls):
    ret = [0]*10
    exps = glob.glob(os.path.join(path, "*"))
    for exp in exps:
        file_list = glob.glob(os.path.join(exp, "test", "plots","mean*.csv"))
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

# m_n = "/mnt/AbnormalResult/M_N_cifar"
# m_n_result = read_csv_to_auc_mean(m_n)
# print(m_n_result)
dis_cifar_cls = cifar_cls
dis_cifar_cls[1] = 'car'
dis_cifar_cls[7] = 'plane'
x = dis_cifar_cls
# plt.plot(x, AnoGAN, 'ro-', label='skip-AnoGAN')
fig = plt.figure(figsize=(2200.0/300.0,1400.0/300.0))
plt.plot(x, EGBAD, 'y.-', label='EGBAD')
plt.plot(x, Ganormaly, 'g+-', label='Ganormaly')
plt.plot(x, SkipGan, 'bx-', label='SkipGan')
plt.plot(x, Proposed, 'ro-', label='Proposed')
plt.ylim(0.3,1)
plt.xlabel('class designated as anomalous class')
plt.ylabel('AUROC')
plt.legend(ncol=4, columnspacing=3, frameon=False, bbox_to_anchor=(0, 0.08, 1, 0))
plt.savefig("comparison_auroc_cifar.png", dpi=300)
