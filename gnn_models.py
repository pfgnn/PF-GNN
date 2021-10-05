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


class PNA_GNN_QM9(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=75, readout_bn=False, divide_input=False, deg=None):
        super(PNA_GNN_QM9, self).__init__()
        aggregators = [ 'max', 'std', 'mean'] #'sum's 'mean'] #
        scalers = ['identity']#, 'amplification', 'attenuation']
        self.dim  = dim = hidden
        self.edim = edim = 64
        self.iter = num_layers
        self.conv = ModuleList()
        self.normalize = ModuleList()
        for i in range(self.iter):
            self.conv += [PNAConv(in_channels=hidden, out_channels=hidden,
                                aggregators=aggregators, scalers=scalers, deg=deg,
                                edge_dim=edim, towers=4, pre_layers=1, post_layers=1,
                                divide_input=False)]
            self.normalize += [BatchNorm(hidden)]
        self.gru = GRU(hidden, hidden)
        
        # self.fc_u = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        # self.fc_v = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        # self.fc = Sequential(Linear(2*dim, edim), ReLU(), Linear(2*dim, edim))
        self.edge_mesg = Sequential(Linear(2*hidden, 2*hidden), ReLU(), Linear(2*hidden, 2*edim), ReLU())
        self.edge_update = Sequential(Linear(1*2*edim, edim), ReLU(), Linear(2*edim, edim))
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = x
        h = out.unsqueeze(0)
        for i in range(self.iter):
            m = F.relu(self.conv[i](out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            
            u = out[edge_index[0,:]]
            v = out[edge_index[1,:]]
            m_ed = self.edge_mesg(torch.cat((u,v), dim=1))
            edge_attr = self.edge_update(torch.cat((m_ed,  edge_attr), dim=1))
            # edge_attr = F.relu(self.fc(self.fc_u(u) * self.fc_v(v)))
            # edge_attr = torch.matmul(m_ed.unsqueeze(1), self.nn(edge_attr).view(-1,self.edim,self.edim)).squeeze(1)
            

        # readout = torch.cat(nodelist, dim=-1)
        # out = self.mlp_node(readout)
        # readout_edge = torch.cat(nodelist_edge, dim=-1)
        # edge_attr = self.mlp_edge(readout_edge)
        return out, edge_attr, out #batch.max()+1



class GNN_QM9_pna(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=75, readout_bn=False, divide_input=False, deg=None):
        super(GNN_QM9_pna, self).__init__()
        edim = hidden
        self.iter = num_layers
        self.dim = dim = hidden
        # self.edge_mesg = Sequential(Linear(2*hidden, 2*hidden), ReLU(), Linear(2*hidden, hidden))
        # self.edge_update = Sequential(Linear(2*hidden, 2*hidden), ReLU(), Linear(2*hidden, hidden))#,
        #                     # ReLU(), Linear(hidden, hidden))
        self.edim = edim = 64
        aggregators = ['mean', 'max', 'std'] #'sum's 'mean'] #
        scalers = ['identity'] #,'amplification', 'attenuation']
        self.dim = dim = hidden
        self.conv = ModuleList()
        for i in range(self.iter):
            self.conv += [PNAConv(in_channels=hidden, out_channels=hidden,
                                    aggregators=aggregators, scalers=scalers, deg=deg,
                                    edge_dim=edim, towers=4, pre_layers=2, post_layers=2,
                                    divide_input=False)]

        self.gru = GRU(hidden, hidden)
        self.fc_u = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        self.fc_v = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        self.fc = Sequential(Linear(2*dim, 2*dim), ReLU(), Linear(2*dim, edim))
        self.edge_update = Sequential(Linear(1*2*edim, 2*edim), ReLU(), Linear(2*edim, edim))#,
        # self.set2set = Set2Set(dim, processing_steps=3)
        # self.set2set_e = Set2Set((dim), processing_steps=3)
        # self.mlp = Sequential(Linear(2*(dim + dim), 4*dim), ReLU(), 
        #             Linear(4*dim, 2*dim), ReLU(),
        #             Linear(2*dim, 12))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     model_reset_parameters(self)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = x
        h = x.unsqueeze(0)
        nodelist_edge=[]
        nodelist_edge.append(edge_attr)
        for i in range(self.iter):
            m = self.conv[i](out, edge_index, edge_attr)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
                        
            u = out[edge_index[0,:]]
            v = out[edge_index[1,:]]
            # m_ed = self.mlp_edge1(torch.cat((u,v), dim=1))
            # edge_attr = self.mlp_edge2(torch.cat((m_ed,  edge_attr), dim=1))            
            m_ed = F.relu(self.fc(self.fc_u(u) * self.fc_v(v)))
            edge_attr = self.edge_update(torch.cat((m_ed,  edge_attr), dim=1))            
            
        # out = self.set2set(out, batch) 
        # out_ed = self.set2set_e(edge_attr, batch[edge_index[0,:]]) + \
        #          self.set2set_e(edge_attr, batch[edge_index[1,:]])        
        # out = self.mlp(torch.cat([out, out_ed], dim=1))

        return out, edge_attr, out #batch.max()+1 #edge_attr



class GNN_QM9_nn(torch.nn.Module):
    def __init__(self, num_layers=3, hidden=75, readout_bn=False, divide_input=False, deg=None):
        super(GNN_QM9_nn, self).__init__()
        edim = hidden
        self.iter = num_layers
        self.dim = dim = hidden
        self.edim = edim = 64
        nn = Sequential(Linear(edim, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(hidden, hidden)
        self.fc_u = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        self.fc_v = Sequential(Linear(dim, 2*dim), ReLU(), Linear(2*dim, 2*dim), ReLU())
        self.fc = Sequential(Linear(2*dim, 2*dim), ReLU(), Linear(2*dim, edim))
        self.edge_update = Sequential(Linear(1*2*edim, 2*edim), ReLU(), Linear(2*edim, edim))
        # self.set2set = Set2Set(dim, processing_steps=3)
        # self.set2set_e = Set2Set((dim), processing_steps=3)
        # self.mlp = Sequential(Linear(2*(dim + dim), 4*dim), ReLU(), 
        #             Linear(4*dim, 2*dim), ReLU(),
        #             Linear(2*dim, 12))


    # def reset_parameters(self):
    #     for i in range(self.iter):
    #         self.conv[i].reset_parameters()
    #         self.normalize[i].reset_parameters()
    #     self.mlp.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = x
        h = x.unsqueeze(0)
        nodelist_edge=[]
        nodelist_edge.append(edge_attr)
        for i in range(self.iter):
            m = self.conv(out, edge_index, edge_attr)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
                        
            u = out[edge_index[0,:]]
            v = out[edge_index[1,:]]
            m_ed = F.relu(self.fc(self.fc_u(u) * self.fc_v(v)))
            edge_attr = self.edge_update(torch.cat((m_ed,  edge_attr), dim=1))            
            
        # out = self.set2set(out, batch) 
        # out_ed = self.set2set_e(edge_attr, batch[edge_index[0,:]]) + \
        #          self.set2set_e(edge_attr, batch[edge_index[1,:]])             
        # out = self.mlp(torch.cat([out, out_ed], dim=1))

        return out, edge_attr, out #batch.max()+1 #edge_attr
