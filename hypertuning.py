from hyperopt import pyll, hp
from options import Options, setup_dir
from models.model import ANB
from hyperopt import fmin, tpe
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
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

def train_model(args):
    for i in range(1):
        auc = objective(args)
        tune.track.log(mean_accuracy=auc)
        
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


if __name__ == '__main__':
    ray.init()
    space = {
        'batchsize': hp.choice("batchsize", [64, 128, 256, 512]),
        'nz': hp.choice('nz', [16, 32, 64, 128, 256]),
        'n_MC_Gen': hp.choice('n_MC_Gen', [3, 4, 5, 6, 7]),
        'n_MC_Disc': hp.choice('n_MC_Disc', [3, 4, 5, 6, 7]),
        'niter': hp.choice('niter', [1]),
        'lr': hp.uniform('lr', 0.001, 0.0001),
        'std_policy': hp.choice('std_policy', ['D_based', 'G_based', 'DG_based']),
        'metric': hp.choice('metric', ['mean_metric', 'std_metric']),
             }
    algo = HyperOptSearch(space, max_concurrent=8, reward_attr="neg_mean_loss")
    analysis = tune.run(train_model, name="my_exp", num_samples=5, search_alg=algo,resources_per_trial={"gpu":0.15})
    with open("hyper.txt", 'w') as f:
        f.write(analysis.get_best_config(metric="mean_accuracy"))