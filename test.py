import glob
from options import Options
from models.model import ANB

def main():
    """ Testing
    """
    opt = Options().parse()
    model = ANB(opt)
    weights = {
        'net_G': glob.glob("/home/golf/code/AbnormalBDL/output/exp3/train/weights/Net_G*"),
        'net_D': glob.glob("/home/golf/code/AbnormalBDL/output/exp3/train/weights/Net_D*")
    }
    model.load_weight(weights)
    model.test_epoch(5)


if __name__ == '__main__':
    main()
