from hyperopt import pyll, hp
from options import Options
from models.model import ANB
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

def objective(args):
    opt = Options().parse()
    opt.nz = args['nz']
    opt.lr = args['lr']
    opt.niter = args['niter']
    opt.n_MC_Gen = args['n_MC_Gen']
    opt.n_MC_Disc = args['n_MC_Disc']
    opt.batchsize = args['batchsize']
    opt.std_policy = args['std_policy']

    model = ANB(opt)
    model.train()
    return model.get_best_result()

# define a search space
space = {
    'batchsize': hp.choice("batchsize", [64, 128, 256, 512]),
    'nz': hp.choice('nz', [16, 32, 64, 128, 256]),
    'n_MC_Gen': hp.choice('n_MC_Gen', [3, 4, 5, 6, 7]),
    'n_MC_Disc': hp.choice('n_MC_Disc', [3, 4, 5, 6, 7]),
    'niter': hp.choice('niter', [20, 25, 30]),
    'lr': hp.uniform('lr', 0.001, 0.0001),
    'std_policy': hp.choice('std_policy', ['D_based', 'G_based', 'DG_based'])
         }


from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)