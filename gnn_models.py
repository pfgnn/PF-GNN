import torch
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import GINConv, PNAConv, NNConv
from torch_geometric.nn import BatchNorm, GraphSizeNorm
from torch_geometric.utils import *
from torch_geometric.nn.inits import *
from torch_scatter import scatter_add
from utils import model_reset_parameters
from torch_geometric.nn import NNConv, Set2Set

class GNN(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=75, readout_bn=False, divide_input=False, deg=None):
        super(GNN, self).__init__()
        edim = hidden//3
        self.iter = num_layers
        aggregators = ['min', 'max', 'std', 'mean'] #'sum's
        scalers = ['identity', 'amplification', 'attenuation']
        self.normalize = ModuleList() 
        self.conv = ModuleList()
        divide_input=False
        for i in range(self.iter):
            if i!=self.iter-1: divinp=True
            else: divinp=False
            self.conv += [PNAConv(in_channels=hidden, out_channels=hidden,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=edim, towers=5, pre_layers=1, post_layers=1,
                            divide_input=True)]
            self.normalize += [BatchNorm(hidden)]
        self.gnorm = GraphSizeNorm()
        self.mlp = Linear(self.iter*hidden, hidden)

    def reset_parameters(self):
        for i in range(self.iter):
            self.conv[i].reset_parameters()
            self.normalize[i].reset_parameters()
        self.mlp.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        nodelist  = []
        for i in range(self.iter):
            m = self.conv[i](x, edge_index, edge_attr)
            m = self.gnorm(m, batch)
            x = self.normalize[i](m) #, batch)
            x = F.relu(x, inplace=True)
            nodelist.append(x)
        readout = torch.cat(nodelist, dim=-1)
        readout = self.mlp(readout)
        return x, readout
        

class GNN_TRIANGLES(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=75, readout_bn=False, divide_input=False, deg=None):
        super(GNN_TRIANGLES, self).__init__()
        edim = hidden//3
        self.iter = num_layers
        self.conv = GINConv(Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden)),
                              train_eps=True)
        self.gru = GRU(hidden, hidden)
        self.fc_cat = Sequential(Linear((self.iter)*hidden, 2*hidden), ReLU(), Linear(2*hidden, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        model_reset_parameters(self)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = x.unsqueeze(0)
        nodelst = []
        for i in range(self.iter):
            m = F.relu(self.conv(x, edge_index))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)    
            nodelst.append(x)
        readout = self.fc_cat(torch.cat(nodelst, dim=1))
        return x, readout

