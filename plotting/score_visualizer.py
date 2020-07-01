import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import glob
titles = ["baseline", "m_pairs", "mxn"]

# files = glob.glob("../app/M_pairs_airplane_*_score_train.csv")
files = ["../app/1_pairs_airplane_2exp_10epoch_0abnidx_score_train.csv",
         "../app/M_pairs_airplane_4exp_13epoch_0abnidx_score_train.csv",
         "../app/mxn_airplane_3exp_4epoch_0abnidx_score_train.csv"]
# for filepath in files:
# filepath = '/home/golf/code/AbnormalBDL/aaa.csv'
filepaths = ["../app/1_pairs_airplane_2exp_10epoch_0abnidx_score_train.csv",
         "../app/M_pairs_airplane_4exp_13epoch_0abnidx_score_train.csv",
         "../app/mxn_airplane_3exp_4epoch_0abnidx_score_train.csv"]
fig, ax_grid = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, squeeze=False)
fig.set_figwidth(15.9)
fig.set_figheight(9)
    # abnormal_class
    # filepaths = [filepath.replace('M_pairs', '1_pairs'),
    #              filepath,
    #              filepath.replace('M_pairs', 'mxn')]
    # abnormal_class = int(filepath.split('/')[-1][-23])
abnormal_class = 0
for i, fp in enumerate(filepaths):
    N = 990
    ys = np.zeros((N, 2, 10))
    xs = np.zeros((N, 2, 10))

    for j, sp in enumerate(['train', 'test']):
        fp = fp.replace('train', sp)
        mean_pd = pd.read_csv(fp)
        mean_n = np.array(mean_pd[mean_pd.labels != abnormal_class]['scores'])
        mean_ab = np.array(mean_pd[mean_pd.labels == abnormal_class]['scores'])
        means = np.concatenate([mean_n, mean_ab])
        labels = np.concatenate([np.zeros_like(mean_n),np.ones_like(mean_ab)])
        roc_auc = roc_auc_score(labels, means)

        cls_means = [np.array(mean_pd[mean_pd.labels==cls]['scores']) for cls in range(10)]

        for cls, cls_mean in enumerate(cls_means):
            np.random.shuffle(cls_mean)
            if sp == 'train' and cls == abnormal_class:
                continue
            chosen_data = cls_mean[:1000]
            ys[:, j, cls] = np.ones_like(chosen_data[:N])*cls

            xs[:, j, cls] = np.sort(chosen_data)[:N]

        # xs[xs>0.4] = 0.4
    xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))


    ax_grid[0, i].scatter([], [], s=1000, alpha=0.2, marker=1, label="abnormal testpoint")
    ax_grid[1, i].scatter(ys[:, 1, abnormal_class], xs[:, 1, abnormal_class], s=1000, alpha=0.2, marker=1, label="abnormal testpoint")

    ys = np.delete(ys, (abnormal_class), axis=2)
    xs = np.delete(xs, (abnormal_class), axis=2)

    ax_grid[0, i].scatter(ys[:, 0], xs[:, 0], s=1000, alpha=0.2, marker=1, label="normal testpoint")
    ax_grid[1, i].scatter(ys[:, 1], xs[:, 1], s=1000, alpha=0.2, marker=1, label="normal testpoint")
    #
    ax_grid[0, i].set_xticks(np.arange(0, 11, 1))
    ax_grid[0, i].set_ylim(0, 1)

    ax_grid[0, i].set_title(titles[i]+'  |  roc: %.3f'% roc_auc)
ax_grid[1, 1].set_xlabel("selected test image", labelpad=20)
ax_grid[0, 1].set_xlabel("selected train image", labelpad=20)
ax_grid[0, 0].set_ylabel("abnormal score")
ax_grid[1, 0].set_ylabel("abnormal score")
plt.legend()
plt.tight_layout()
plt.savefig("{}.png".format('best'), dpi=300)
plt.show()
