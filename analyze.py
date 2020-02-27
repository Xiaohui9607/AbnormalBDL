

##
# LIBRARIES

from options import Options, setup_dir
from models.model_mxn import model_mxn


def main():
    """ Training
    """
    opt = Options().parse()
    setup_dir(opt)
    model = model_mxn(opt)
    path = '/Users/golf/code/Abnormal_result/Abnormal_result/cifar/M_N/mxn_automobile_1/train/weights/'
    weight_path = {
        'net_G':[path+'Net_G_0_epoch_2.pth', path+'Net_G_1_epoch_2.pth', path+'Net_G_2_epoch_2.pth'],
        'net_D':[path+'Net_D_0_epoch_2.pth', path+'Net_D_1_epoch_2.pth', path+'Net_D_2_epoch_2.pth']
    }
    model.load_weight(weight_path)

    gt_label, pred = model.test_epoch(0)
    gt_label[gt_label not in [1, 9]] =0


if __name__ == '__main__':
    main()