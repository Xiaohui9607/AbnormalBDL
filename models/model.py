import torch
from models.networks import Generator, Discriminator
from dataloader.dataloader import load_data
import tqdm
class ANB:
    def __init__(self, opt):
        self.opt = opt
        self.dataloader = load_data(opt)

        self.net_G = Generator()
        self.net_D = Discriminator()
        self.epoch = self.opt.niter



    def train_epoch(self):
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            # TODO Discriminator optimize step

            # step1: Real input feed foward
            pass

            # step2(a): Generate fake input(proposal1 —— uncertainty applies in the generator parameters)

            # step2(b): Fake input feed foward
            pass

            # step3: optimize net_D

            # TODO Generator optimize step
            pass

    def train(self):
        for i in range(10):
            self.train_epoch()
