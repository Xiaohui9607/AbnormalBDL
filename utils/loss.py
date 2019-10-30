import torch
from torch import nn
from torch.autograd import Variable


class noise_loss(torch.nn.Module):
    # need the scale for noise standard deviation
    # scale = noise  std
    def __init__(self, params, scale=None, data_size=None):
        super(noise_loss, self).__init__()
        # initialize the distribution for each parameter
        #self.distributions = []
        self.noises = []
        for param in params:
            noise = torch.zeros_like(param.data) # will fill with normal at each forward
            self.noises.append(noise)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1.
        self.data_size = data_size

    def forward(self, params, scale=None, data_size=None):
        # scale should be sqrt(2*alpha/eta)
        # where eta is the learning rate and alpha is the strength of drag term
        if scale is None:
            scale = self.scale
        if data_size is None:
            data_size = self.data_size

        assert scale is not None, "Please provide scale"
        noise_loss = 0.0
        for noise, var in zip(self.noises, params):
            # This is scale * z^T*v
            # The derivative wrt v will become scale*z
            _noise = noise.normal_(0, 1).clone().detach()
            noise_loss += scale * torch.sum(_noise * var)
        noise_loss /= data_size
        return noise_loss


class prior_loss(torch.nn.Module):
    # negative log Gaussian prior
    def __init__(self, prior_std=1., data_size=None):
        super(prior_loss, self).__init__()
        self.data_size = data_size
        self.prior_std = prior_std

    def forward(self, params, data_size=None):
        if data_size is None:
            data_size = self.data_size
        prior_loss = 0.0
        for var in params:
            prior_loss += torch.sum(var*var/(self.prior_std * self.prior_std))
        prior_loss /= data_size
        return prior_loss


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
            return torch.mean(x-target) / self.b
        else:
            return (x-target) / self.b


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