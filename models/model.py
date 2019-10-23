import torch
from tqdm import tqdm
from models.statsutil import weight_init
from dataloader.dataloader import load_data
from models.networks import Generator, Discriminator


class ANB:
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = load_data(opt)
        self.epoch = self.opt.niter

        # TODO: initialize network and optimizer
        self.net_D = Discriminator(self.opt)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.opt.lr)

        # net_G = Generator(self.opt)
        # optioptimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.opt.lr)

        self.net_Gs = []
        self.optimizer_Gs = []

        # TODO: define likelihood loss function(discriminator), reconstruction loss, feat_loss

        if self.opt.bayes:
            # TODO: define the loss function (piror and noise) proposal by SGHMC

            batch_norm_layers = {}
            for _idxmc in range(0, self.opt.n_MC_samples):
                net_G = Generator(self.opt, batch_norm_layers=batch_norm_layers)
                net_G.apply(weight_init)
                optioptimizer_G = torch.optim.SGD(net_G.parameters(), lr=self.opt.lr)
                self.net_Gs.append(net_G)
                self.optimizer_Gs.append(optioptimizer_G)
        pass

    def train_epoch(self):
        for data in tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train)):
            # TODO Discriminator optimize step

            # step1: Real input feed foward
            pass

            # step2(a): Generate fake input(proposal1 —— uncertainty applies in the generator parameters)

            # step2(b): Fake input feed foward
            pass

            # step3: optimize net_D

            # TODO Generator optimize step

            # do the SGHMC
            pass

    def train(self):
        for i in range(10):
            self.train_epoch()
