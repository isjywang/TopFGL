import random
from random import choices
import numpy as np
import pandas as pd
import os
from scipy.special import rel_entr
import scipy
import torch
from torch_geometric.utils import erdos_renyi_graph, degree
from torch_geometric.transforms import OneHotDegree


from models import *
from server import *
from client import *
from utils import *

def init_FGL_parti(splitedData, args):
    clients = []
    for c in range(args.clients):
        client_graph, num_features, num_labels, train_size = splitedData[c]
        ## init clients
        if args.alg == 'topfgl':
            main_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, main_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            set_seed(args)
            
            topology_learner = Top_learner(num_features, args.hidden, args.hidden_top, num_labels, args.dropout, args.RWE)            
            optimizer_struct = torch.optim.Adam(filter(lambda p: p.requires_grad, topology_learner.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            set_seed(args)
            clients.append(   Client_topfgl(main_model, c, train_size, client_graph, optimizer, args, topology_learner, optimizer_struct, num_labels)) 
        else: 
            cmodel = GCN(num_features, args.hidden, num_labels, args.dropout)       
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            clients.append(Client(cmodel, c, train_size, client_graph, optimizer, args))

    ## init server
    if args.alg == 'topfgl':
        main_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
        set_seed(args)
        topology_learner = Top_learner(num_features, args.hidden, args.hidden_top, num_labels, args.dropout, args.RWE)  
        set_seed(args)

        all_model, all_model_t = [], []
        for i in range(args.clients):
            m_model = GCN(num_features, args.hidden, num_labels, args.dropout) 
            all_model.append(m_model)
            set_seed(args)
            topology_learner = Top_learner(num_features, args.hidden, args.hidden_top, num_labels, args.dropout, args.RWE)  
            all_model_t.append(topology_learner)
            set_seed(args)
        server = Server_topfgl(main_model, topology_learner, args.device, num_features, num_labels, args, all_model, all_model_t)
        
    else:
        smodel = GCN(num_features, args.hidden, num_labels, args.dropout) 
        server = Server(smodel, args.device)
    
    return clients, server