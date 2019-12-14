import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from models.networks import Encoder
from dataloader.dataloader import Data
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, opt, affine=True):
        super(Net, self).__init__()
        self.backbone = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers,
                               affine=affine)
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv2d(opt.nz, 1, 3, 1, 1, bias=False))
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        feat = self.backbone(x)
        classifier = self.classifier(feat)
        classifier = classifier.view(-1, 1).squeeze(1)
        return feat, classifier


def get_mnist_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0, portion=0.1):
    # Get images and labels.
    sample_gap = int(0.1/portion)
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    tst_img, tst_lbl = valid_ds.data, valid_ds.targets

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx][::sample_gap]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx][::sample_gap]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:

    train_ds.data = nrm_trn_img.clone()
    valid_ds.data = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    valid_ds.targets = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return train_ds, valid_ds


def load_data(opt, portion):
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
    valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
    train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class), portion)

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)
