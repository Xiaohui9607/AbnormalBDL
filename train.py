

##
# LIBRARIES

from options import Options, setup_dir
from models.basemodel import ANBase
from models.model_mxn import model_mxn
from models.model_mpairs import model_mpairs
##
def main():
    """ Training
    """
    opt = Options().parse()
    setup_dir(opt)

    model = model_mxn(opt)
    model.test_epoch(0)


if __name__ == '__main__':
    main()
