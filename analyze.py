

##
# LIBRARIES

from options import Options, setup_dir
from models.model_mxn import model_mxn
from models.model_mpairs import model_mpairs
import argparse
import glob, os
def main():
    """ Training
    """
    path = '/mnt/AbnormalResult/'

    exps = [os.path.join(path, '1_cifar/1_pairs_airplane_2/train')]

    for exp in exps:
        optfile = os.path.join(exp, 'opt.txt')
        opt = Options().parse_from_file(optfile)
        opt.batchsize = 64
        if opt.setting == 'mxn':
            model = model_mxn(opt)
        else:
            model = model_mpairs(opt)
        for iter in range(opt.niter):
            weight_path = {
                'net_G': sorted(glob.glob(os.path.join(exp, 'weights', 'Net_G*_epoch_%d.pth*'%iter))),
                'net_D': sorted(glob.glob(os.path.join(exp, 'weights', 'Net_D*_epoch_%d.pth*'%iter)))
            }
            if len(weight_path['net_D']) != opt.n_MC_Disc and len(weight_path['net_G']) != opt.n_MC_Gen:
                continue
            try:
                model.load_weight(weight_path)
            except:
                continue
            print("{}_{}".format(opt.name, iter))
            model.compute_epoch(iter)


if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.parse_known_args()
