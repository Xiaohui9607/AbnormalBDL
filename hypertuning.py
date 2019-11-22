from hyperopt import pyll, hp
from options import Options, setup_dir
from models.model import ANB
from hyperopt import fmin, tpe
import hyperopt
'''
hyperparameters that neet to be tune.
1. batchsize
2. nz
3. ngf
4. ndf
5. n_MC_Gen
6. n_MC_Disc
7. niter
8. lr
9. sigma_lat
10. scale_con
11. w_adv
12. std_policy
'''
count = 0

def objective(args):
    global count
    print(args)
    opt = Options().parse()
    opt.nz = args['nz']
    opt.lr = args['lr']
    opt.niter = args['niter']
    opt.n_MC_Gen = args['n_MC_Gen']
    opt.n_MC_Disc = args['n_MC_Disc']
    opt.batchsize = args['batchsize']
    opt.std_policy = args['std_policy']
    opt.name = "exp_%d" % count
    setup_dir(opt)
    count += 1
    model = ANB(opt)
    model.train()
    return model.get_best_result(args['metric'])

# define a search space
space = {
    'batchsize': hp.choice("batchsize", [64, 128, 256, 512]),
    'nz': hp.choice('nz', [16, 32, 64, 128, 256]),
    'n_MC_Gen': hp.choice('n_MC_Gen', [3, 4, 5, 6, 7]),
    'n_MC_Disc': hp.choice('n_MC_Disc', [3, 4, 5, 6, 7]),
    'niter': hp.choice('niter', [30]),
    'lr': hp.uniform('lr', 0.001, 0.0001),
    'std_policy': hp.choice('std_policy', ['D_based', 'G_based', 'DG_based']),
    'metric': hp.choice('metric', ['mean_metric', 'std_metric']),
         }


if __name__ == '__main__':
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
    with open("hyper.txt", 'w') as f:
        f.write("best\n")
        f.write(best)
        f.write("\nhyperopt.space_eval(space, best)\n")
        f.write(hyperopt.space_eval(space, best))