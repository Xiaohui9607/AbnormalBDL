import os
import math
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
import seaborn as sns
import random
import matplotlib.pyplot as plt
from models.evaluate import roc
from collections import OrderedDict
from utils import weights_init, Visualizer
from dataloader.dataloader import load_data
from torch.optim.lr_scheduler import StepLR
from models.networks import Generator, Discriminator
from utils.loss import lat_loss, con_loss, noise_loss, prior_loss

torch.autograd.set_detect_anomaly(True)


class ANB:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.batchsize % self.opt.split == 0, "#batchsize must be divisible w.r.t #split "
        self.opt.batchsize //= self.opt.split

        self.dataloader = {
            "gen": [load_data(self.opt) for _ in range(self.opt.n_MC_Gen)],
            "disc": [load_data(self.opt) for _ in range(self.opt.n_MC_Disc)]
        }

        self.epoch = self.opt.niter
        self.data_size = len(self.dataloader["gen"][0].train) * self.opt.batchsize
        self.visualizer = Visualizer(self.opt)
        self.device = 'cpu' if not self.opt.gpu_ids else 'cuda'
        self.global_iter = 0
        self.sgd_lr = self.opt.lr
        self.adam_lr = self.opt.lr

        self.rocs = []

        if self.opt.phase == 'train':
            # TODO: initialize network and optimizer
            self.generator_setup()
            self.discriminator_setup()
            self.optims = {
                "gen": self.optimizer_Gs_Adam,
                "disc": self.optimizer_Ds_Adam
            }
            # TODO: define discriminator loss function
            self.l_adv = nn.BCELoss(reduction='mean')
            self.l_con = con_loss(b=self.opt.scale_con, reduction='mean')
            self.l_lat = lat_loss(sigma=self.opt.sigma_lat, reduction='mean')

            # TODO: define hmc loss
            if self.opt.bayes:
                self.l_g_prior = prior_loss(prior_std=1., data_size=self.data_size)
                self.l_g_noise = noise_loss(params=self.net_Gs[0].parameters(), scale=math.sqrt(2 * self.opt.noise_alpha * self.opt.lr),
                                            data_size=self.data_size)

                self.l_d_prior = prior_loss(prior_std=1, data_size=self.data_size)
                self.l_d_noise = noise_loss(params=self.net_Ds[0].parameters(), scale=math.sqrt(2 * self.opt.noise_alpha * self.opt.lr),
                                            data_size=self.data_size)

    def generator_setup(self):
        self.net_Gs = []
        self.optimizer_Gs_Adam = []
        self.optimizer_Gs = []

        net_G = Generator(self.opt).to(self.device)
        optimizer_G_Adam = torch.optim.Adam(net_G.parameters(), lr=self.adam_lr, betas=(self.opt.beta1, 0.999))
        optimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.sgd_lr)

        self.net_Gs.append(net_G)
        self.optimizer_Gs_Adam.append(optimizer_G_Adam)
        self.optimizer_Gs.append(optimizer_G)

        if self.opt.bayes:
            self.net_Gs[0].apply(weights_init)
            for _idxmc in range(1, self.opt.n_MC_Gen):
                net_G = Generator(self.opt).to(self.device)
                # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
                net_G.apply(weights_init)
                optimizer_G_Adam = torch.optim.Adam(net_G.parameters(), lr=self.adam_lr, betas=(self.opt.beta1, 0.999))
                optimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.sgd_lr)
                self.net_Gs.append(net_G)
                self.optimizer_Gs_Adam.append(optimizer_G_Adam)
                self.optimizer_Gs.append(optimizer_G)

    def discriminator_setup(self):
        self.net_Ds = []
        self.optimizer_Ds_Adam = []
        self.optimizer_Ds = []

        net_D = Discriminator(self.opt).to(self.device)
        optimizer_D_Adam = torch.optim.Adam(net_D.parameters(), lr=self.adam_lr, betas=(self.opt.beta1, 0.999))
        optimizer_D = torch.optim.SGD(net_D.parameters(), lr=self.sgd_lr)

        self.net_Ds.append(net_D)
        self.optimizer_Ds_Adam.append(optimizer_D_Adam)
        self.optimizer_Ds.append(optimizer_D)

        if self.opt.bayes:
            self.net_Ds[0].apply(weights_init)
            for _idxmc in range(1, self.opt.n_MC_Disc):
                net_D = Discriminator(self.opt).to(self.device)
                # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
                net_D.apply(weights_init)
                optimizer_D_Adam = torch.optim.Adam(net_D.parameters(), lr=self.adam_lr, betas=(self.opt.beta1, 0.999))
                optimizer_D = torch.optim.SGD(net_D.parameters(), lr=self.sgd_lr)
                self.net_Ds.append(net_D)
                self.optimizer_Ds_Adam.append(optimizer_D_Adam)
                self.optimizer_Ds.append(optimizer_D)

    def train_epoch(self, epoch):
        # for iter, (x_real, _) in enumerate(tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train))):
        loss_D = [{'err_d': 0, 'err_d_lat': 0} for _ in range(self.opt.n_MC_Disc)]
        loss_G = [{'err_g': 0, 'err_g_con': 0} for _ in range(self.opt.n_MC_Gen)]

        for _iter in tqdm(range(len(self.dataloader["gen"][0].train)), leave=False, total=len(self.dataloader["gen"][0].train)):
            self.global_iter += 1
            if self.global_iter == self.opt.warm_up:
                print("Switching to user-specified optimizer")
                self.optims = {
                    "gen": self.optimizer_Gs,
                    "disc": self.optimizer_Ds
                }
            errors = OrderedDict([
                ('err_d', []),
                ('err_g', []),
                ('err_g_con', []),
                ('err_d_lat', [])])

            for _idxD in range(self.opt.n_MC_Disc):
                x_real, _ = next(iter(self.dataloader["disc"][_idxD].train))
                x_real = x_real.to(self.device)
                # TODO update each disc with all gens
                self.net_Ds[_idxD].zero_grad()
                label_real = torch.ones(x_real.shape[0]).to(self.device)
                pred_real, feat_real = self.net_Ds[_idxD](x_real)
                err_d_real = self.l_adv(pred_real, label_real)

                err_d_fakes = []
                err_d_lats = []

                for _idxG in range(self.opt.n_MC_Gen):
                    # self.net_Gs[_idxG].zero_grad()
                    x_fake = self.net_Gs[_idxG](x_real)
                    pred_fake, feat_fake = self.net_Ds[_idxD](x_fake.detach())
                    label_fake = torch.zeros(x_real.shape[0]).to(self.device)
                    err_d_fake = self.l_adv(pred_fake, label_fake)
                    err_d_fakes.append(err_d_fake)

                    err_d_lat = self.l_lat(feat_real, feat_fake)
                    err_d_lats.append(err_d_lat)

                err_d_total_loss = torch.zeros([1, ], dtype=torch.float32).to(self.device)
                err_g_total_lat = torch.zeros([1, ], dtype=torch.float32).to(self.device)
                for err_d_fake, err_d_lat in zip(err_d_fakes, err_d_lats):
                    err_d_loss = err_d_fake + err_d_real + err_d_lat
                    if self.opt.bayes:
                        err_d_loss += self.l_d_noise(self.net_Ds[_idxD].parameters()) + \
                                      self.l_d_prior(self.net_Ds[_idxD].parameters())
                    err_d_total_loss += err_d_loss
                    err_g_total_lat += err_d_lat

                err_d_total_loss /= self.opt.n_MC_Gen

                err_d_total_loss.backward()
                self.optims['disc'][_idxD].step()

                errors['err_d'].append(err_d_total_loss.detach())
                errors['err_d_lat'].append(err_g_total_lat.detach().reshape([1]))

            # TODO update each gen with all discs
            for _idxG in range(self.opt.n_MC_Gen):
                x_real, _ = next(iter(self.dataloader["gen"][_idxG].train))
                x_real = x_real.to(self.device)

                self.net_Gs[_idxG].zero_grad()

                x_fake = self.net_Gs[_idxG](x_real)
                err_g_con = self.l_con(x_real, x_fake)

                err_g_fakes = []

                for _idxD in range(self.opt.n_MC_Disc):
                    # self.net_Ds[_idxD].zero_grad()        # is it necessary?
                    _, feat_real = self.net_Ds[_idxD](x_real)
                    x_fake = self.net_Gs[_idxG](x_real)
                    pred_fake, feat_fake = self.net_Ds[_idxD](x_fake)
                    label_real = torch.ones(x_real.shape[0]).to(self.device)

                    err_g_fake = self.l_adv(pred_fake, label_real)
                    err_g_fakes.append(err_g_fake)

                err_g_total_loss = torch.zeros([1, ], dtype=torch.float32).to(self.device)

                for err_g_fake in err_g_fakes:

                    if self.opt.bayes:
                        err_g_fake += self.l_g_noise(self.net_Gs[_idxG].parameters()) + \
                                      self.l_g_prior(self.net_Gs[_idxG].parameters())
                    err_g_total_loss += err_g_fake

                err_g_total_loss /= self.opt.n_MC_Disc
                err_g_total_loss += err_g_con

                err_g_total_loss.backward()
                self.optims['gen'][_idxG].step()

                errors['err_g'].append(err_g_total_loss.detach())
                errors['err_g_con'].append(err_g_con.detach().reshape([1]))

            epoch_iter = _iter * self.opt.batchsize

            # ploting
            if epoch_iter % self.opt.print_freq == 0:
                errors['err_g'] = torch.mean(torch.cat(errors['err_g'])).item()
                errors['err_d_lat'] = torch.mean(torch.cat(errors['err_d_lat'])).item()
                errors['err_g_con'] = torch.mean(torch.cat(errors['err_g_con'])).item()
                errors['err_d'] = torch.mean(torch.cat(errors['err_d'])).cpu().item()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader["gen"][0].train.dataset)
                    self.visualizer.plot_current_errors(epoch, counter_ratio, errors)

            if epoch_iter % self.opt.save_image_freq == 0:
                reals, fakes = x_real, torch.cat([net_G(x_real).detach() for net_G in self.net_Gs], dim=0)
                self.visualizer.save_current_images(epoch, reals, fakes)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes)

            # sampling weight
            if self.opt.save_weight and self.global_iter > self.opt.warm_up:
                if random.uniform(0, 1) < 0.2:
                    for _idx, net_G in enumerate(self.net_Gs):
                        torch.save(net_G.state_dict(),
                                   '{0}/{1}/train/weights/Net_G_{2}_epoch_{3}_iter_{4}.pth'.format(self.opt.outf, self.opt.name,
                                                                                                   _idx, epoch, iter))

                    for _idx, net_D in enumerate(self.net_Ds):
                        torch.save(net_D.state_dict(),
                                   '{0}/{1}/train/weights/Net_D_{2}_epoch_{3}_iter_{4}.pth'.format(self.opt.outf, self.opt.name,
                                                                                               _idx, epoch, iter))

    def train(self):
        for net_D in self.net_Ds:
            net_D.train()
        for net_G in self.net_Gs:
            net_G.train()
        for epoch in range(self.opt.niter):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            # self.save_weight(epoch)

    def save_weight(self, epoch):
        for _idx, net_G in enumerate(self.net_Gs):
            torch.save(net_G.state_dict(), '{0}/{1}/train/weights/Net_G_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))

        for _idx, net_D in enumerate(self.net_Ds):
            torch.save(net_D.state_dict(), '{0}/{1}/train/weights/Net_D_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))

    def test_epoch(self, epoch, plot_hist=True):
        with torch.no_grad():
            self.opt.phase = 'test'
            means = [torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset), self.opt.n_MC_Gen), dtype=torch.float32,
                                device=self.device) for _ in range(self.opt.n_MC_Disc)]
            gt_labels = torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset),),
                                    dtype=torch.long, device=self.device)

            for _idxData, (x_real, label) in enumerate(self.dataloader["gen"][0].valid, 0):
                x_real = x_real.to(self.device)

                gt_labels[_idxData * self.opt.batchsize: _idxData * self.opt.batchsize + label.size(0)] = label
                for _idxD in range(self.opt.n_MC_Disc):
                    pred_real, feat_real = self.net_Ds[_idxD](x_real)
                    for _idxG in range(self.opt.n_MC_Gen):
                        x_fake = self.net_Gs[_idxG](x_real)
                        pred_fake, feat_fake = self.net_Ds[_idxD](x_fake)

                        sz = feat_real.size()

                        lat = (feat_real - feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])

                        lat = torch.mean(torch.pow(lat, 2), dim=1)

                        means[_idxD][_idxData * self.opt.batchsize:(_idxData+1)*self.opt.batchsize, _idxG].copy_(lat)

            vars = [torch.var(mean, dim=1, keepdim=True) for mean in means]
            means = [torch.mean(mean, dim=1, keepdim=True) for mean in means]
            vars = torch.mean(torch.cat([var + torch.pow(mean, 2) for var, mean in zip(vars, means)], dim=1), dim=1)
            means = torch.mean(torch.cat(means, dim=1), dim=1)
            vars = vars - torch.pow(means, 2)

            means = means.cpu()
            vars = vars.cpu()

            per_scores = means
            per_scores = (per_scores - torch.min(per_scores)) / (torch.max(per_scores) - torch.min(per_scores))
            auc = roc(gt_labels, per_scores, epoch=epoch, save=os.path.join(self.opt.outf, self.opt.name,
                                                                            "test/plots/mean_at_epoch{0}.png".format(epoch)))

            self.rocs.append(auc)

            # PLOT HISTOGRAM
            if plot_hist:
                plt.ion()
                # Create data frame for scores and labels.
                scores = {}
                scores['scores'] = means
                scores['labels'] = gt_labels.cpu()
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv("{0}/{1}/test/plots/mean_at_epoch{2}.csv".format(self.opt.outf, self.opt.name, epoch))
                re_var = {}
                re_var['var'] = vars
                re_var['labels'] = gt_labels.cpu()
                hist_v = pd.DataFrame.from_dict(re_var)
                hist_v.to_csv("{0}/{1}/test/plots/var_at_epoch{2}.csv".format(self.opt.outf, self.opt.name, epoch))
                all_scores = {}

                all_scores['labels'] = gt_labels.cpu()
                hist_r = pd.DataFrame.from_dict(all_scores)
                hist_r.to_csv("{0}/{1}/test/plots/scores_for_all_combination_at_epoch{2}.csv".format(self.opt.outf, self.opt.name, epoch))

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                # fig, ax = plt.subplots(figsize=(4,4));
                sns.distplot(nrm_scr, label=r'Normal Scores')
                sns.distplot(abn_scr, label=r'Abnormal Scores')

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')
                plt.savefig("{0}/{1}/test/plots/sns_at_epoch{2}.png".format(self.opt.outf, self.opt.name, epoch))
                plt.close()

    def load_weight(self, pathlist:dict):
        self.net_Gs = []
        self.net_Ds = []
        for weight in pathlist['net_G']:
            net_G = Generator(self.opt).to(self.device)
            net_G.load_state_dict(torch.load(weight))
            self.net_Gs.append(net_G)
        for weight in pathlist['net_D']:
            net_D = Discriminator(self.opt).to(self.device)
            net_D.load_state_dict(torch.load(weight))
            self.net_Ds.append(net_D)







