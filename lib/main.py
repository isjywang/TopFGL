import os
import sys
import argparse
import random
import copy
import time
import torch
from pathlib import Path

from training import *
from utils import *
from setup import *

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--RWE', type=int, default = 8)
    parser.add_argument('--hidden_top', type=int, default = 8)
    parser.add_argument('--knn_method', type=str, default = 'pyg')
    
    parser.add_argument('--check',type=int, default= 0 )
    parser.add_argument('--dataset',type=str, default='Cora')
    parser.add_argument('--clients',type=int, default= 20 )
    parser.add_argument('--mode',type=str, default='disjoint')
    
    parser.add_argument('--repair_fre',type=float, default=3)
    parser.add_argument('--warm_epoch',type=float, default=0.3)
    parser.add_argument('--k',type=int, default=8)
    parser.add_argument('--ccs', type=int, default=10)
    parser.add_argument('--alg', type=str, default='fedtop')

    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=2)


    begin = time.time()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("now the method is: ", args.alg)
    print("now the dataset is: ", args.dataset)
    print("the clients num is :", args.clients)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_config(args)
    
    # preparing data
    print("Preparing data ...") 
    splitedData = load_FGL_data(args)  
    print("Done")
    print("weight decay:",args.weight_decay)
    print("local epoch:",args.local_epoch)
    
    init_clients, init_server = init_FGL_parti(splitedData, args)

    if args.alg == 'fedtop':
        run_fedtop(args, init_clients, init_server, args.num_rounds, args.local_epoch)
    elif args.alg == 'local':
        run_local(args, init_clients, init_server, args.num_rounds, args.local_epoch)
    else: ## fedavg
        run_fedavg(args, init_clients, init_server, args.num_rounds, args.local_epoch)
        
    end = time.time()
    print("all time:", end - begin)

    sys.stdout.flush()
    os._exit(0)


