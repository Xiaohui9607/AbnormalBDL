import os
import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from collections import OrderedDict
from utils import weights_init, Visualizer
from dataloader.dataloader import load_data
from models.networks import Generator, Discriminator
from utils.loss import lat_loss, con_loss, noise_loss, prior_loss


class ANB:
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = load_data(opt)
        self.epoch = self.opt.niter
        self.data_size = len(self.dataloader.train) * self.opt.batchsize
        self.visualizer = Visualizer(opt)
        self.device = 'cpu' if not self.opt.gpu_ids else 'cuda'

        # TODO: initialize network and optimizer
        self.net_D = Discriminator(self.opt).to(self.device)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.opt.lr)

        self.net_Gs = []
        self.optimizer_Gs = []

        g_lr = self.opt.lr * self.opt.n_MC_samples
        # batch_norm_layers = {}
        net_G = Generator(self.opt, batch_norm_layers={}).to(self.device)
        optimizer_G = torch.optim.SGD(net_G.parameters(), lr=g_lr)

        self.net_Gs.append(net_G)
        self.optimizer_Gs.append(optimizer_G)

        if self.opt.bayes:
            # TODO: define the loss function (piror and noise) proposal by SGHMC
            self.net_Gs[0].apply(weights_init)
            for _idxmc in range(1, self.opt.n_MC_samples):
                net_G = Generator(self.opt, batch_norm_layers={}).to(self.device)
                # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
                net_G.apply(weights_init)
                optimizer_G = torch.optim.SGD(net_G.parameters(), lr=g_lr)
                self.net_Gs.append(net_G)
                self.optimizer_Gs.append(optimizer_G)

        # TODO: define discriminator loss function
        self.l_adv = nn.BCELoss(reduction='mean')
        self.l_con = con_loss(b=self.opt.scale_con, reduction='mean')
        self.l_lat = lat_loss(sigma=self.opt.sigma_lat, reduction='mean')

        # TODO: define hmc loss
        if self.opt.bayes:
            self.l_g_prior = prior_loss(prior_std=1., data_size=self.data_size*self.opt.n_MC_samples)
            self.l_g_noise = noise_loss(params=self.net_Gs[0].parameters(), scale=math.sqrt(2 * self.opt.gnoise_alpha / self.opt.lr),
                                        data_size=self.data_size*self.opt.n_MC_samples)

    def train_epoch(self, i_epoch):
        '''
        problem 1: n_mc_sample > 1 will mess up (solved)
        problem 2: original code add the err_g_lat to optimize the generator, but it's meaningless!
        problem 3: when to stop the reconstruction loss backward, can not allow it dominate all the time (no uncertainty)
        '''
        for iter, (x_real, _) in enumerate(tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train))):
            # TODO Discriminator optimize step

            self.net_D.zero_grad()
            x_real = x_real.to(self.device)
            # Generate fake input(proposal1 —— uncertainty applies in the generator parameters)
            x_fakes = []
            x_reals = []
            for net_G in self.net_Gs:
                x_real_i = x_real.clone()
                x_reals.append(x_real_i)
                x_fakes.append(net_G(x_real_i))
            x_fakes = torch.cat(x_fakes, dim=0)
            x_reals = torch.cat(x_reals, dim=0)
            label_reals = torch.ones(x_reals.shape[0]).to(self.device)
            label_fakes = torch.zeros(x_reals.shape[0]).to(self.device)

            # step1(a): Real input feed foward
            pred_reals, feat_reals = self.net_D(x_reals)
            # step1(b): Real loss
            err_d_real = self.opt.w_adv * self.l_adv(pred_reals, label_reals)

            # step2(a): Fake input feed foward
            pred_fakes, feat_fakes = self.net_D(x_fakes.detach())  # don't backprop net_G!
            # step2(cb: Fake loss
            err_d_fake = self.opt.w_adv * self.l_adv(pred_fakes, label_fakes)

            # step3: Latent feature loss
            err_g_lat = self.l_lat(feat_reals, feat_fakes)

            # TODO: add SGHMC for Discriminative (or just pure discriminative loss)
            # step4: err summerize
            err_d = err_d_fake + err_d_real + err_g_lat
            err_d.backward(retain_graph=True)

            # step5: optimize net_D
            self.optimizer_D.step()

            # TODO Generator optimize step
            for net_G in self.net_Gs:
                net_G.zero_grad()
            # step1(a): Fake input feed foward
            pred_fakes, _ = self.net_D(x_fakes)  # backprop net_G!
            # step1(b): Fake loss (gradient inverse, use label_real)
            err_g_fake = self.opt.w_adv * self.l_adv(pred_fakes, label_reals)
            # pretrain use reconstruction loss (strategy not confirm)

            # step2: Fake reconstruction loss
            if True:
                err_g_con = self.l_con(x_reals, x_fakes)

            err_g = err_g_fake + err_g_lat + err_g_con

            # do SGHMC for err_g_fake and err_g_lat
            if self.opt.bayes:
                for net_G in self.net_Gs:
                    err_g += self.l_g_noise(net_G.parameters())

                    err_g += self.l_g_prior(net_G.parameters())
            err_g.backward(retain_graph=True)
            # optimize net_G
            for optimizer_G in self.optimizer_Gs:
                optimizer_G.step()

            # printing option
            epoch_iter = iter * self.opt.batchsize
            if epoch_iter % self.opt.print_freq == 0:
                errors = OrderedDict([
                    ('err_d', err_d),
                    ('err_g', err_g),
                    ('err_g_adv', err_d_fake),
                    ('err_g_con', err_g_con),
                    ('err_g_lat', err_g_lat)])
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader.train.dataset)
                    self.visualizer.plot_current_errors(i_epoch, counter_ratio, errors)

            if epoch_iter % self.opt.save_image_freq == 0:
                reals, fakes = x_real[0:1], x_fakes[0::self.opt.batchsize]
                self.visualizer.save_current_images(i_epoch, reals, fakes)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes)

    def train(self):
        self.net_D.train()
        for net_G in self.net_Gs:
            net_G.train()
        for epoch in range(self.opt.niter):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            self.save_weight(epoch)

    def save_weight(self, epoch):
        for _idx, net_G in enumerate(self.net_Gs):
            torch.save(net_G.state_dict(), 'Net_D_{0}_epoch_{1}.pth'.format(_idx, epoch))
        torch.save(self.net_D.state_dict(), 'Net_G_epoch_{0}.pth'.format(epoch))

    def test_epoch(self, epoch, plot_hist=False):
        with torch.no_grad():
            # Load the weights of netg and netd.
            # if self.opt.load_weights:
            #     self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.features = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9 * rec + 0.1 * lat

                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                plt.ion()
                # Create data frame for scores and labels.
                scores['scores'] = self.an_scores
                scores['labels'] = self.gt_labels
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv("histogram.csv")

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

            ##
            # PLOT PERFORMANCE
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            ##
            # RETURN
            return performance

    def load_weight(self, pathlist:dict):
        pass
