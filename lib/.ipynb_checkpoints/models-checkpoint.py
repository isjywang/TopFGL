import torch
import torch.nn.functional as F
import random
from torch_geometric.nn import *
from torch_geometric.nn import GCNConv


        

class GCN(torch.nn.Module):
    def __init__(self, n_feat, nhid, nclass, dropout): ## n_se 1
        super(GCN,self).__init__()
        from torch_geometric.nn import GCNConv

        self.dropout = dropout
        self.n_feat = n_feat
        self.nhid = nhid
        self.nclass = nclass

        self.conv1 = GCNConv(self.n_feat, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
     
        self.clsif = torch.nn.Linear(self.nhid, self.nclass)
        self.nhid = nhid
        
    def forward(self, data, gr=False):
        x, edge_index = data.x, data.edge_index 

        x = self.conv1(x, edge_index) ## x,hid
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x1 = self.conv2(x, edge_index)
        x2 = F.relu(x1)
        x3 = F.dropout(x2, training=self.training)
        x = self.clsif(x3)
        if gr:
            return x1, x
        else:
            return x
    
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

## topology learner
class Top_learner(torch.nn.Module):  ## RandomWalk Embedding + Homogeneity Embedding
    def __init__(self, n_feat, nhid, nhid_top, nclass, dropout, RWE):
        super(Top_learner,self).__init__()
        self.dropout, self.n_feat, self.nhid, self.nhid_top, self.nclass, self.nrw = dropout, n_feat, nhid, nhid_top, nclass, RWE
        
        self.input_hidden = self.nrw + 2
        self.output_hidden = max(1,int(self.input_hidden/self.nhid*self.nclass))

        self.structural_layer_1 = GCNConv(self.input_hidden, self.nhid_top)  ## RandomWalk + Homogeneity
        self.structural_layer_2 = GCNConv(self.nhid_top, self.output_hidden) 
        self.clsif = torch.nn.Linear(self.nclass+ self.output_hidden, self.nclass)


    def forward(self, data, emb_a, check=False):
        x, edge_index, structual_emb = data.x, data.edge_index, data.stc_enc

        x = self.structural_layer_1(data.stc_enc, edge_index) 
        x = F.dropout(x, training=self.training)
        x = self.structural_layer_2(x, edge_index)

        x = F.relu(x)
        emb_b = F.dropout(x, training=self.training)
        emb = torch.cat((emb_a, emb_b), dim=-1)
        x = self.clsif(emb)
        
        return x
        
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)




