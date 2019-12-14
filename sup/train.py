import torch
import os
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from models.evaluate import roc
from options import Options, setup_dir
from sup.model import load_data, Net


class Baseline:
    def __init__(self, opt):
        self.opt = opt
        self.portion = 0.1
        self.model = Net(opt)
        self.data = load_data(opt, self.portion)
        self.device = 'cpu' if not self.opt.gpu_ids else 'cuda'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.l_bce = nn.BCELoss(reduction='mean')
        self.model.to(self.device)

    def train(self):
        for i in range(opt.niter):
            self.train_epoch()
            self.evaluate_epoch(i)

    def train_epoch(self):
        self.model.train()
        for x, y_true in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            y_true = y_true.to(self.device).float()
            x = x.to(self.device)
            _, y_pred = self.model(x)
            loss = self.l_bce(y_pred, y_true)
            loss.backward()
            self.optimizer.step()


    def evaluate_epoch(self, epoch):
        with torch.no_grad():
            self.opt.phase = 'test'

            scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)

            for _idxData, (x_real, label) in enumerate(self.data.valid, 0):
                x_real = x_real.to(self.device)

                gt_labels[_idxData * self.opt.batchsize: _idxData * self.opt.batchsize + label.size(0)] = label

                feat, y_pred = self.model(x_real)

                scores[_idxData * self.opt.batchsize:(_idxData+1)*self.opt.batchsize].copy_(y_pred)

            scores = scores.cpu().squeeze()

            auc_means = roc(gt_labels, scores, epoch=epoch, save=os.path.join(self.opt.outf, self.opt.name,
                                                                            "test/plots/mean_at_epoch{0}.png".format(epoch)))

            self.rocs['mean_metric'].append(auc_means)


            # PLOT HISTOGRAM
            plt.ion()
            # Create data frame for scores and labels.
            scores = {}
            scores['scores'] = scores
            scores['labels'] = gt_labels.cpu()
            hist = pd.DataFrame.from_dict(scores)
            hist.to_csv("{0}/{1}/test/plots/mean_at_epoch{2}.csv".format(self.opt.outf, self.opt.name, epoch))

        pass


if __name__ == '__main__':
    opt = Options().parse()
    setup_dir(opt)
    BS = Baseline(opt)
    BS.train()