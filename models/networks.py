import torch
from torch import nn


class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True, batch_norm_layers={}, affine=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        if 'Encoder' not in batch_norm_layers:
            batch_norm_layers['Encoder'] = []
            csize, cndf = isize / 2, ndf
            for t in range(n_extra_layers):
                batch_norm_layers['Encoder'].append(nn.BatchNorm2d(cndf, affine=affine))

            while csize > 4:
                out_feat = cndf * 2
                batch_norm_layers['Encoder'].append(nn.BatchNorm2d(out_feat, affine=affine))
                cndf = cndf * 2
                csize = csize / 2
        else:
            print("reusing batchnorm layers for %s" % self.__class__.__name__)

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
            # main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
            #                 nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            batch_norm_layers['Encoder'][_idxbn])
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=False))
            _idxbn += 1

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            # main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
            #                 nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            batch_norm_layers['Encoder'][_idxbn])
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
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0, batch_norm_layers={}, affine=True):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        if 'Decoder' not in batch_norm_layers:
            batch_norm_layers['Decoder'] = []
            cngf, tisize = ngf // 2, 4
            while tisize != isize:
                cngf = cngf * 2
                tisize = tisize * 2
            batch_norm_layers['Decoder'].append(nn.BatchNorm2d(cngf, affine=affine))

            csize, _ = 4, cngf
            while csize < isize // 2:
                batch_norm_layers['Decoder'].append(nn.BatchNorm2d(cngf // 2, affine=affine))
                cngf = cngf // 2
                csize = csize * 2

            for t in range(n_extra_layers):
                batch_norm_layers['Decoder'].append(nn.BatchNorm2d(cngf, affine=affine))

        else:
            print("reusing batchnorm layers for %s" % self.__class__.__name__)

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
                        batch_norm_layers['Decoder'][_idxbn])
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(False))
        _idxbn += 1

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            batch_norm_layers['Decoder'][_idxbn])
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
                            batch_norm_layers['Decoder'][_idxbn])
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
    def __init__(self, opt, batch_norm_layers={}, affine=True):
        super(Generator, self).__init__()
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers,
                               batch_norm_layers=batch_norm_layers, affine=affine)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers,
                               batch_norm_layers=batch_norm_layers, affine=affine)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('classifier', nn.Conv2d(1, 1, 3, 1, 1, bias=False))
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
