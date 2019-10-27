import torch
import math
from torch import nn
from tqdm import tqdm
from models.loss import lat_loss, noise_loss, prior_loss
from models.utils import weights_init
from dataloader.dataloader import load_data
from models.networks import Generator, Discriminator


class ANB:
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = load_data(opt)
        self.epoch = self.opt.niter
        self.data_size = len(self.dataloader.train) * self.opt.batchsize

        # TODO: initialize network and optimizer
        self.net_D = Discriminator(self.opt)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.opt.lr)

        self.net_Gs = []
        self.optimizer_Gs = []

        net_G = Generator(self.opt, batch_norm_layers={})
        optimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.opt.lr)

        self.net_Gs.append(net_G)
        self.optimizer_Gs.append(optimizer_G)

        if self.opt.bayes:
            # TODO: define the loss function (piror and noise) proposal by SGHMC
            self.net_Gs[0].apply(weights_init)
            batch_norm_layers = {}
            for _idxmc in range(1, self.opt.n_MC_samples):
                net_G = Generator(self.opt, batch_norm_layers=batch_norm_layers)
                # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
                net_G.apply(weights_init)
                optimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.opt.lr)
                self.net_Gs.append(net_G)
                self.optimizer_Gs.append(optimizer_G)

        # TODO: define loss function
        self.l_adv = nn.BCELoss(reduction='mean')
        self.l_con = nn.L1Loss(reduction='mean')    # reduction = ?
        self.l_lat = lat_loss(0.1)  # sigma is a hyperparamter, add it to parser later

        self.l_g_prior = prior_loss(prior_std=1., data_size=self.data_size)
        self.l_g_noise = noise_loss(params=net_G.parameters(), scale=math.sqrt(2 * self.opt.gnoise_alpha / self.opt.lr), data_size=self.data_size)

    def train_epoch(self):
        for x_real, _ in tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train)):
            # TODO Discriminator optimize step

            self.net_D.zero_grad()

            # Generate fake input(proposal1 —— uncertainty applies in the generator parameters)
            x_fakes = []
            x_reals = []
            for net_G in self.net_Gs:
                x_reals.append(x_real)
                x_fakes.append(net_G(x_real))
            x_fakes = torch.cat(x_fakes, dim=0)
            x_reals = torch.cat(x_reals, dim=0)
            label_reals = torch.ones(x_reals.shape[0])
            label_fakes = torch.zeros(x_reals.shape[0])

            # step1(a): Real input feed foward
            pred_reals, feat_reals = self.net_D(x_reals)
            # step1(b): Real loss
            err_d_real = self.l_adv(pred_reals, label_reals)

            # step2(a): Fake input feed foward
            pred_fakes, feat_fakes = self.net_D(x_fakes.detach())  # don't backprop net_G!
            # step2(cb: Fake loss
            err_d_fake = self.l_adv(pred_fakes, label_fakes)

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
            err_g_fake = self.l_adv(pred_fakes, label_reals)
            # pretrain use reconstruction loss (strategy not confirm)
            if True:
                err_g_con = self.l_con(x_reals, x_fakes)
            err_g = err_g_fake + err_g_lat + err_g_con
            # if self.opt.bayes:
            #     for net_G in self.net_Gs:
            #         err_g += self.l_g_noise(net_G.parameters()) + self.l_g_prior(net_G.parameters())
            err_g.backward(retain_graph=True)
            for optimizer_G in self.optimizer_Gs:
                optimizer_G.step()
            # do SGHMC for err_g_fake and err_g_lat


            pass

    def train(self):
        # for self.epoch in range(self.opt.iter, self.opt.niter):
        #     self.train_one_epoch()
        #     res = self.test()
        #     if res['AUC'] > best_auc:
        #         best_auc = res['AUC']
        #         self.save_weights(self.epoch)
        #     self.visualizer.print_current_performance(res, best_auc)
        # print(">> Training model %s.[Done]" % self.name)
        self.net_D.train()
        # print(f">> Training {self.name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        for net_G in self.net_Gs:
            net_G.train()
        for i in range(10):
            self.train_epoch()
