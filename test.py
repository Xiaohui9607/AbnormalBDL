import glob
from options import Options, setup_dir
from models.model import ANB
import matplotlib.pyplot as plt

def main():
    """ Testing
    """
    opt = Options().parse()
    setup_dir(opt)
    # opt.phase = "test"

    model = ANB(opt)
    dtld = model.dataloader['gen'][0]
    for x,y in dtld.train:
        image = x[0].squeeze().numpy()*255
        plt.imshow(image, cmap='gray')
        plt.show()
        pass
    # weights = {
    #     'net_G': glob.glob("/home/golf/code/AbnormalBDL/output/exp6/train/weights/Net_G*"),
    #     'net_D': glob.glob("/home/golf/code/AbnormalBDL/output/exp6/train/weights/Net_D*")
    # }
    # model.load_weight(weights)
    # model.test_oct(100)


if __name__ == '__main__':
    main()
