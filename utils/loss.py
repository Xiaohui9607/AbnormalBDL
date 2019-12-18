import torch
from torch import nn
from torch.autograd import Variable

class con_loss(nn.Module):
    def __init__(self, b, reduction='mean'):
        super(con_loss, self).__init__()
        self.b = torch.tensor(b)
        if reduction in ['sum', 'mean']:
            self.reduction = reduction
        else:
            raise KeyError

    def forward(self, x, target):
        if self.reduction == 'mean':
            return torch.mean(torch.abs(x-target)) / self.b
        else:
            return torch.abs(x-target) / self.b


class lat_loss(nn.Module):
    def __init__(self, sigma, reduction='mean'):
        super(lat_loss, self).__init__()
        self.sigma = torch.tensor(sigma)
        if reduction in ['sum', 'mean']:
            self.reduction = reduction
        else:
            raise KeyError

    def forward(self, x, target):
        if self.reduction == 'mean':
            return torch.mean(torch.pow(x-target, 2)) * 1/torch.pow(self.sigma, 2)
        else:
            return torch.pow(x-target, 2) * 1/torch.pow(self.sigma, 2)