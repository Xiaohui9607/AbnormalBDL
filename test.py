from options import Options
from models.model import ANB

def main():
    """ Testing
    """
    opt = Options().parse()
    model = ANB(opt)
    weights = {
        'net_G':[],
        'net_D':[]
    }
    model.load_weight()
    result = model.inference()


if __name__ == '__main__':
    main()
