

##
# LIBRARIES

from options import Options, setup_dir
from models.basemodel import ANBase
from models.model_mxn import model_mxn
from models.model_mpairs import model_mpairs
# from models.model_mxn_deepem import model_mxn_deepem
##
def main():
    """ Training
    """
    opt = Options().parse()
    setup_dir(opt)
    if opt.setting == "mxn":
        model = model_mxn(opt)
    elif opt.setting == "mpairs":
        model = model_mpairs(opt)
    model.train()




if __name__ == '__main__':
    main()
