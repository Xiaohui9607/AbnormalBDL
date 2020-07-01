import torch
from torch import nn
import numpy as np
from torch.nn import init
import functools


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(opt, norm='batch', use_dropout=False, init_type='normal'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    num_layer = int(np.log2(opt.isize))
    netG = UnetGenerator(opt.nc, opt.nc, num_layer, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(netG, init_type, opt.gpu_ids)

##
def define_D(opt, norm='batch', use_sigmoid=False, init_type='normal'):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = BasicDiscriminator(opt)
    return init_net(netD, init_type, opt.gpu_ids)

class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True, affine=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=False))
        csize, cndf = isize / 2, ndf
        _idxbn = 0
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf, affine=affine))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=False))
            _idxbn += 1

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat, affine=affine))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=False))
            cndf = cndf * 2
            csize = csize / 2
            _idxbn += 1

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0, affine=True):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        _idxbn = 0
        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf, affine=affine))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(False))
        _idxbn += 1

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2, affine=affine))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(False))
            cngf = cngf // 2
            csize = csize * 2
            _idxbn += 1

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf, affine=affine))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(False))
            _idxbn += 1

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self, opt, affine=True):
        super(Generator, self).__init__()
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers,
                               affine=affine)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers,
                               affine=affine)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.feat = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv2d(opt.nz, 1, 3, 1, 1, bias=False))
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.feat(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

class Discriminator_Deepem(nn.Module):
    def __init__(self, opt):
        super(Discriminator_Deepem, self).__init__()
        self.feat = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.nz = opt.nz
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential()
        # self.classifier.add_module('dropout', nn.Dropout(0.5))
        self.classifier.add_module('classifier', nn.Conv2d(opt.nz, 1, 3, 1, 1, bias=False))
        self.classifier.add_module('Sigmoid', nn.Sigmoid())


    def forward(self, x, dropout=None):
        features = self.feat(x).squeeze()
        if dropout is None:
            features = self.dropout(features)

        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features

def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

class MLP_Generator(nn.Module):
    def __init__(self, opt):
        super(MLP_Generator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(122,64),
                                     nn.Tanh(),
                                     nn.Linear(64, 32),
                                     nn.Tanh(),
                                     nn.Linear(32, 16))
                                     # nn.Tanh(),
                                     # nn.Linear(16, 8))
        self.decoder = nn.Sequential(#nn.Linear(8,16),
                                     #nn.Tanh(),
                                     nn.Linear(16, 32),
                                     nn.Tanh(),
                                     nn.Linear(32, 64),
                                     nn.Tanh(),
                                     nn.Linear(64, 122))
    def forward(self, x):
        feature = self.encoder(x)
        ret = self.decoder(feature)
        return ret


class MLP_Discriminator(nn.Module):
    def __init__(self, opt):
        super(MLP_Discriminator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(122,64),
                                     nn.Tanh(),
                                     nn.Linear(64, 32),
                                     nn.Tanh(),
                                     nn.Linear(32, 16),
                                     # nn.Tanh(),
                                     # nn.Linear(16),
                                     nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(16,1),
                                        nn.Sigmoid())
    def forward(self, x):
        feature = self.encoder(x)
        ret = self.classifier(feature)
        return  ret, feature

# if __name__ == '__main__':
#     x = torch.ones(64, 118)
#     G = MLP_Generator()
#     D = MLP_Discriminator()
#     cls = D(x)
#     pass
