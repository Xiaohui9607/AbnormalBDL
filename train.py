"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""


##
# LIBRARIES

from options import Options, setup_dir
from models.model import ANB

##
def main():
    """ Training
    """
    opt = Options().parse()
    setup_dir(opt)
    model = ANB(opt)
    model.train()


if __name__ == '__main__':
    main()