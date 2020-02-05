import os
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.evaluate import roc
from collections import OrderedDict
from utils import Visualizer
from dataloader.dataloader import load_data
from models.networks import Generator, Discriminator, define_D, define_G
from utils.loss import lat_loss, con_loss
from utils import weights_init

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
        self.rocs = {
            'mean_metric':[],
            'std_metric':[]
        }
        if self.opt.DCGAN:
            self.create_D = define_D
            self.create_G = define_G
        else:
            self.create_D = Discriminator
            self.create_G = Generator
        if self.opt.phase == 'train':
            # TODO: initialize network and optimizer
            self.generator_setup()
            self.discriminator_setup()
            self.optims = {
                "gen": self.optimizer_Gs,
                "disc": self.optimizer_Ds
            }
            # TODO: define discriminator loss function
            self.l_adv = nn.BCELoss(reduction='mean')
            self.l_con = con_loss(b=self.opt.scale_con, reduction='mean')
            self.l_lat = lat_loss(sigma=self.opt.sigma_lat, reduction='mean')

    def generator_setup(self):
        self.net_Gs = []
        self.optimizer_Gs = []

        net_G = self.create_G(self.opt).to(self.device)
        optimizer_G = torch.optim.Adam(net_G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        self.net_Gs.append(net_G)
        self.optimizer_Gs.append(optimizer_G)

        # if self.opt.bayes:
        self.net_Gs[0].apply(weights_init)
        for _idxmc in range(1, self.opt.n_MC_Gen):
            net_G = self.create_G(self.opt).to(self.device)
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            net_G.apply(weights_init)
            optimizer_G = torch.optim.Adam(net_G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Gs.append(net_G)
            self.optimizer_Gs.append(optimizer_G)

    def discriminator_setup(self):
        self.net_Ds = []
        self.optimizer_Ds = []

        net_D = self.create_D(self.opt).to(self.device)
        optimizer_D = torch.optim.Adam(net_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        self.net_Ds.append(net_D)
        self.optimizer_Ds.append(optimizer_D)

        self.net_Ds[0].apply(weights_init)
        for _idxmc in range(1, self.opt.n_MC_Disc):
            net_D = self.create_D(self.opt).to(self.device)
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            net_D.apply(weights_init)
            optimizer_D = torch.optim.Adam(net_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Ds.append(net_D)
            self.optimizer_Ds.append(optimizer_D)

    def train_epoch(self, epoch):
        for _iter in tqdm(range(len(self.dataloader["gen"][0].train)), leave=False, total=len(self.dataloader["gen"][0].train)):
            self.global_iter += 1
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
                err_d_lat_loss = torch.zeros([1, ], dtype=torch.float32).to(self.device)
                for err_d_fake, err_d_lat in zip(err_d_fakes, err_d_lats):

                    err_d_loss = err_d_fake + err_d_real + err_d_lat
                    err_d_total_loss += err_d_loss
                    err_d_lat_loss += err_d_lat

                err_d_total_loss /= self.opt.n_MC_Gen

                err_d_total_loss.backward()
                self.optims['disc'][_idxD].step()

                errors['err_d'].append(err_d_total_loss.detach() - err_d_lat_loss.detach()/self.opt.n_MC_Gen)
                errors['err_d_lat'].append(err_d_lat_loss.detach())

            # TODO update each gen with all discs
            for _idxG in range(self.opt.n_MC_Gen):
                x_real, _ = next(iter(self.dataloader["gen"][_idxG].train))
                x_real = x_real.to(self.device)

                self.net_Gs[_idxG].zero_grad()

                x_fake = self.net_Gs[_idxG](x_real)
                err_g_con = self.l_con(x_real, x_fake)

                err_g_fakes = []

                for _idxD in range(self.opt.n_MC_Disc):
                    _, feat_real = self.net_Ds[_idxD](x_real)
                    x_fake = self.net_Gs[_idxG](x_real)
                    pred_fake, feat_fake = self.net_Ds[_idxD](x_fake)
                    label_real = torch.ones(x_real.shape[0]).to(self.device)

                    err_g_fake = self.l_adv(pred_fake, label_real)
                    err_g_fakes.append(err_g_fake)

                err_g_total_loss = torch.zeros([1, ], dtype=torch.float32).to(self.device)

                for err_g_fake in err_g_fakes:
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
                errors['err_g_con'] = torch.mean(torch.cat(errors['err_g_con'])).item()
                errors['err_d'] = torch.mean(torch.cat(errors['err_d'])).cpu().item()
                errors['err_d_lat'] = torch.mean(torch.cat(errors['err_d_lat'])).cpu().item()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader["gen"][0].train.dataset)
                    self.visualizer.plot_current_errors(epoch, counter_ratio, errors)

            if epoch_iter % self.opt.save_image_freq == 0:
                reals, fakes = x_real, torch.cat([net_G(x_real).detach() for net_G in self.net_Gs], dim=0)
                self.visualizer.save_current_images(epoch, reals, fakes)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes)

    def train(self):
        for net_D in self.net_Ds:
            net_D.train()
        for net_G in self.net_Gs:
            net_G.train()
        for epoch in range(self.opt.niter):
            self.train_epoch(epoch)
            self.save_weight(epoch)
            # self.test_oct(epoch)
            self.test_epoch(epoch)

    def save_weight(self, epoch):
        for _idx, net_G in enumerate(self.net_Gs):
            torch.save(net_G.state_dict(), '{0}/{1}/train/weights/Net_G_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))

        for _idx, net_D in enumerate(self.net_Ds):
            torch.save(net_D.state_dict(), '{0}/{1}/train/weights/Net_D_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))

    def test_epoch(self, epoch, plot_hist=True):
        with torch.no_grad():
            self.opt.phase = 'test'
            means = torch.empty(size=(len(self.dataloader["gen"][0].valid.dataset), self.opt.n_MC_Gen, self.opt.n_MC_Disc), dtype=torch.float32,
                                device=self.device)

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

                        means[_idxData * self.opt.batchsize:(_idxData+1)*self.opt.batchsize, _idxG, _idxD].copy_(lat)


            vars_D_based = torch.var(means, dim=1, keepdim=True)
            vars_G_based = torch.var(means, dim=2, keepdim=True)

            means_D_based = torch.mean(means, dim=1, keepdim=True)
            means_G_based = torch.mean(means, dim=2, keepdim=True)

            vars_D_based = torch.mean(vars_D_based+torch.pow(means_D_based, 2), dim=2)
            vars_G_based = torch.mean(vars_G_based+torch.pow(means_G_based, 2), dim=1)

            means = torch.mean(means_D_based, dim=2)

            if self.opt.std_policy == 'D_based':
               vars = vars_D_based - torch.pow(means, 2)
            elif self.opt.std_policy == "G_based":
               vars = vars_G_based - torch.pow(means, 2)
            else:
               vars = torch.mean(torch.cat([vars_G_based, vars_D_based], dim=1)) - torch.pow(means, 2)

            means = means.cpu().squeeze()
            vars = vars.cpu().squeeze()

            per_scores = means
            per_scores = (per_scores - torch.min(per_scores)) / (torch.max(per_scores) - torch.min(per_scores))
            auc_means = roc(gt_labels, per_scores, epoch=epoch, save=os.path.join(self.opt.outf, self.opt.name,
                                                                            "test/plots/mean_at_epoch{0}.png".format(epoch)))
            per_scores = torch.sqrt(vars)
            per_scores = (per_scores - torch.min(per_scores)) / (torch.max(per_scores) - torch.min(per_scores))
            auc_std = roc(gt_labels, per_scores, epoch=epoch, save=os.path.join(self.opt.outf, self.opt.name,
                                                                                 "test/plots/var_at_epoch{0}.png".format(
                                                                                     epoch)))
            self.rocs['mean_metric'].append(auc_means)
            self.rocs['std_metric'].append(auc_std)

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

    def update_learning_rate(self):
        """ Update learning rate based on the rule provided in options.
        """

        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('   LR = %.7f' % lr)

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

    def get_best_result(self, metric):
        return max(self.rocs[metric])







