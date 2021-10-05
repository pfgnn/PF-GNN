import os.path as osp
import time
import argparse
import torch 
from torch.nn import Embedding
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.nn.inits import *
from torch_geometric.utils import degree
from worker import train, test
# from model import PFGNN_Net
from gnn_models import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.nn import Sequential, Linear, ReLU, GRU, ModuleList
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from utils import model_reset_parameters
################################################### 

import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.utils import remove_self_loops
import time
import argparse
from torch.nn import Embedding
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.utils import degree
#from worker import train, test
from torch_geometric.utils import *
from torch_geometric.nn.inits import *
from torch_scatter import scatter_add, scatter_mean
from utils import model_reset_parameters
from gnn_models import GNN
from gnn_models import GNN, GNN_TRIANGLES, PNA_GNN_QM9, GNN_QM9_nn, GNN_QM9_pna #GNN_QM9


class PFGNN_Net(torch.nn.Module):
    def __init__(self, outdim, dim=150, depth=2, num_particles=4, gnn=GNN, deg=None):
        super(PFGNN_Net, self).__init__()
        self.dim = dim      
        self.policy_func = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), Linear(dim, 1)) 
                           # POLICY(num_layers=1) 
        self.depth = depth
        self.num_particles = num_particles
        self.fc_cat = ModuleList()
        self.convs = ModuleList()
        # self.conv_iter = 3
        self.convs.append(gnn(num_layers=5, hidden=dim, readout_bn=False, divide_input=True, deg=deg))
        for i in range(self.depth):
            divide_input=True 
            # else: divide_input=True
            self.convs.append(gnn(num_layers=1, hidden=dim, readout_bn=False, divide_input=divide_input, deg=deg))
        self.relabel_fc = Sequential(Linear(dim, dim), ReLU(inplace=True), Linear(dim, dim))
        self.fc_particle = Sequential(Linear((self.depth*1+1)*dim, (self.depth*1+1)*dim))
        self.node_fc1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.node_fc2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 1))
        # self.mlp = Sequential(Linear((self.depth*1+1)*dim, self.depth*dim), ReLU(), 
        #             Linear(self.depth*dim, dim), ReLU(),
        #             Linear(dim, outdim))
        self.mlp = Sequential(Linear((self.depth+1)*dim, 2*dim), ReLU(), 
            Linear(2*dim, dim), ReLU(),
            Linear(dim, outdim))
        #self.normalize = BatchNorm(dim)
        # self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=2*dim, num_heads=2)
        model_reset_parameters(self)
        
                       
    def transition(self, node_ids, h_node, edge_index, edge_attr, batch, ind):
        '''
        node_ids : node indices to individualize

        '''

        # Individualize
        new_ones = torch.ones(h_node.size(),device=h_node.device)
        new_ones[node_ids] = F.relu(self.relabel_fc(h_node[node_ids]))
        h_nodes =  h_node * new_ones

        # Refinement
        index = ind+1
        h_nodes, readout = self.convs[index](h_nodes, edge_index, edge_attr, batch)
      
        return h_nodes, readout

    def composite_graph(self, nodes, edge_index, batch, num_particles, edge_attr=None):
        '''
        Make a composite batch of graphs comining all components

        nodes : (n, d)
        edge_index : (2, e)
        batch : (batch_size,)

        nodes_composite : (num_particles*n, d)
        edges_composite : (2, num_particles*e)
        batch_composite : (num_particles*batch_size, )
        
        '''
        batch_size = batch.max()+1
        batchsizes = torch.arange(0,num_particles*batch_size, batch_size, device=nodes.device).view(-1,1)
        batch_composite = batch.repeat(1,num_particles).view(num_particles,-1) + batchsizes
        batch_composite = batch_composite.view(-1)
        #nodes  n x d
        nodes_composite = nodes.unsqueeze(0).repeat(num_particles,1,1).view(-1, nodes.size(-1))
        #edges
        batchsizes_edge = torch.arange(0,num_particles*nodes.size(0),nodes.size(0), device=nodes.device).view(-1,1)
        edges_composite = edge_index.view(-1).repeat(1, num_particles).view(num_particles,-1) + batchsizes_edge
        edges_composite = edges_composite.view(num_particles,2,-1).permute(1,0,2).contiguous().view(2,-1) 
        #edge attr
        edge_attr_composite = None
        if edge_attr is not None:
            edge_attr_composite = edge_attr.unsqueeze(0).repeat(num_particles,1,1).view(-1, edge_attr.size(-1))

        return nodes_composite, edges_composite, batch_composite, edge_attr_composite

    def update_weights_with_obs(self, weight_log, nodes_comp, batch_comp, batch_size):
        ## weights : particles X batch_size
        nodes_comp = F.relu(self.node_fc1(nodes_comp))
        graph_comp = global_add_pool(nodes_comp, batch_comp)
        observation = (self.node_fc2(graph_comp)).sum(-1) #F.logsigmoid
        observation = observation.view(self.num_particles, batch_size)
        #
        weight_log = weight_log + observation 
        weight_log = weight_log - torch.logsumexp(weight_log, dim=0, keepdim=True)
        #
        return weight_log

    def resample(self, nodes_comp, readout_comp, weight_prob, batch):
        ## nodes_comp = (numparticles*nodes) x dim
        ## weight_probs = numparticles x batchsize
        #### entropy = torch.sum(-weight_prob * weight_prob.log())
        #### if entropy < 0.1 :
        
        alpha = 0.5
        weights_sampling = alpha * weight_prob + (1 - alpha) * 1 / self.num_particles
        weights_sampling = weights_sampling.transpose(1,0)#.contiguous()
        indices = torch.multinomial(weights_sampling, self.num_particles, replacement=True)
        # indices: batchsize x numparticles
        
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous() #numparticles x batchsize
        indexes = indices.clone()
        offset = torch.arange(batch_size, device=nodes_comp.device, dtype=torch.long).unsqueeze(0)
        indexes = offset + indexes * batch_size
        flatten_indices = indexes.view(-1, 1).squeeze()

        weight_probs = (weight_prob.view(-1,1)[flatten_indices])
        weight_new = weight_probs / (alpha * weight_probs + (1-alpha) / self.num_particles)
        weight_log_new = torch.log(weight_new+1e-29).view(self.num_particles,-1)
        weight_log = weight_log_new - torch.logsumexp(weight_log_new, dim=0, keepdim=True)
        
        num_nodes = batch.size(0)
        nodes_arange = torch.arange(num_nodes, dtype=torch.long, device=nodes_comp.device)
        nodes_comp_old = nodes_comp.view(self.num_particles, num_nodes, -1)
        nodes_list = []
        for i in range(self.num_particles):
            idx = indices[i][batch]
            nodes_particle_i = nodes_comp_old[idx, nodes_arange]
            nodes_list.append(nodes_particle_i.unsqueeze(0))
        nodes_comp_new = torch.cat(nodes_list, dim=0).contiguous().view(-1, nodes_comp.size(-1))
        
        readout_comp_old = readout_comp.view(self.num_particles, num_nodes, -1)
        readout_list = []
        for i in range(self.num_particles):
            idx = indices[i][batch]
            readout_particle_i = readout_comp_old[idx, nodes_arange]
            readout_list.append(readout_particle_i.unsqueeze(0))
        readout_comp_new = torch.cat(readout_list, dim=0).contiguous().view(-1, nodes_comp.size(-1))
        
        return nodes_comp_new, readout_comp_new, weight_log

    def sample_action(self, nodes, edge_index, batch, k=1, iteration=0, pol_hist=[]):
        '''
        Returns the indices of the nodes to individualize
        by sampling from Policy function on nodes
        '''
        score = self.policy_func(nodes).sum(-1)
        # score = score.masked_fill(pol_hist == 0, -1e9) # For not sampling previously individualized nodes
        score = softmax(score, batch)
        
        num_nodes = scatter_add(batch.new_ones(nodes.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)
        
        index = torch.arange(batch.size(0), dtype=torch.long, device=nodes.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)
        dense_x = nodes.new_full((batch_size * max_num_nodes, ), 0.)
        dense_x[index] = score
        probs = dense_x.view(batch_size, max_num_nodes).contiguous()
        
        if True : #self.training :
            perm = torch.multinomial(probs, k, replacement=True)
            prob_vals = probs.gather(1, perm).log()
        else:
            prob_vals, perm = probs.topk(k=k, dim=1)
            prob_vals = prob_vals.log()
            #prob_vals_topk = prob_vals
            perm_topk = perm

        perm = perm + cum_num_nodes.view(-1, 1)
        prob_vals_topk, perm_topk = None, None
        return perm, prob_vals, prob_vals_topk, perm_topk

    def forward(self, node_emb, edge_emb, data):
        batch_size, num_nodes = data.batch.max()+1, data.x.size(0) 
        final_out_list = []
        weight_probs_list = []
        log_probs = []
        
        out = node_emb
        edge_attr = edge_emb  

        out, h_out = self.convs[0](out, data.edge_index, edge_attr, data.batch)

        weight_prob = torch.tensor([1/self.num_particles]*self.num_particles, device=out.device).unsqueeze(1).repeat(1, batch_size)
        weight_log = weight_prob.log()
        #####TRIANGLES-h_out, ZINC-out
        readout = h_out.unsqueeze(0).repeat(self.num_particles, 1, 1).view(-1, h_out.size(1)) 
        readout = readout #* weight_prob[:, data.batch].view(-1).unsqueeze(-1)
        final_out_list.append(readout)
        weight_probs_list.append(weight_prob.unsqueeze(-1).unsqueeze(-1))

        policy_hist = torch.ones((self.num_particles*out.size(0)), dtype=torch.long, device=out.device)
        nodes_comp, edge_index_comp, batch_comp, edge_attr_composite = self.composite_graph(out, data.edge_index, data.batch, \
                                            num_particles=self.num_particles, edge_attr=edge_attr)
        for i in range(self.depth):    
            perm, score, score_topk, perm_topk = self.sample_action(nodes_comp, edge_index_comp, \
                            batch_comp, iteration=i, pol_hist=policy_hist)
            
            score = score.view(self.num_particles, -1).t()
            log_probs.append(score)
            perm = perm.view(-1)  
            
            policy_hist[perm] = 0 #to mask selected vertices for next iteration
            nodes_comp, readout = self.transition(perm, nodes_comp, edge_index_comp, edge_attr=edge_attr_composite, batch=batch_comp, ind=i)
              
            weight_prob = torch.exp(weight_log)    
            nodes_comp, readout_comp, weight_log = self.resample(nodes_comp, readout, weight_prob, data.batch)
            
            weight_prob = torch.exp(weight_log)    
            read_out = readout_comp #* weight_prob[:, data.batch].view(-1).unsqueeze(-1)  
            #nodes = nodes_comp.view(self.num_particles, out.size(0), -1) #torch.cat(out_list, dim=0)          
            #readout = readout.view(self.num_particles, out.size(0), -1).sum(0) #torch.cat(readout_list, dim=0)
            
            final_out_list.append(readout)
            weight_probs_list.append(weight_prob.unsqueeze(-1).unsqueeze(-1))    
        # out = final_out_list[-1]
        out = torch.cat(final_out_list,dim=-1)
        out = out.view(self.num_particles, num_nodes, -1)
        
        out = scatter_add(out, data.batch, dim=1) # out is particls x batchsize x (L*dim)
        out = out.view(self.num_particles, batch_size, self.depth+1, -1)
        weight_prob = torch.cat(weight_probs_list, dim=-2)
        # print (weight_prob)
        out = (out * weight_prob).max(0)[0].view(batch_size, -1)
        # out = (out[:,:,-1,:] * weight_prob[:,:,-1,:]).mean(0).view(batch_size, -1)

        out = self.mlp(out)
        log_probs = torch.cat(log_probs, dim=1).squeeze(-1)
        return out, log_probs, batch_size

################################################### 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PF-GNN-IR')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--factor', type=float, default=0.7)
    parser.add_argument('--patience', type=float, default=20)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--dim', type=int, default=150,
                        help='hidden dim. (default=150)')
    parser.add_argument('--task_name', type=str, default="ZINC")
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='run on multiple gpus (default: True)')
    parser.add_argument('--depth', type=int, default=2,
                        help='number of IR iterations (default: 2)')
    parser.add_argument('--num_particles', type=int, default=4,
                        help='number of particles of particle filter (default: 4)')
    args = parser.parse_args()
    print (args)

    # path = osp.join(osp.dirname(osp.realpath(__file__)), \
    #                 'data', args.task_name)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC')
    # transforms = T.Compose([Complete]) AddVnode())
    
    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    # Compute in-degree histogram over training data.
    deg = torch.zeros(5, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    class Net(torch.nn.Module):
        def __init__(self, node_infeat, edge_infeat, outdim, dim, depth, num_particles, gnn=GNN, deg=None):
            super(Net, self).__init__()
            self.dim = dim
            self.node_emb = Embedding(node_infeat, dim) 
            self.edge_emb = Embedding(edge_infeat, dim//3) 
            self.pfgnn = PFGNN_Net(outdim=outdim, dim=dim, depth=depth, num_particles=num_particles, gnn=gnn, deg=deg)
            # self.gnn = gnn(num_layers=5, hidden=dim, readout_bn=True, divide_input=False, deg=deg)
            # self.mlp = Sequential(Linear(dim, dim), ReLU(), 
            #         Linear(dim, dim), ReLU(),
            #         Linear(dim, outdim))

        def reset_parameters(self):
            self.node_emb.reset_parameters()
            self.edge_emb.reset_parameters()
            self.pfgnn.reset_parameters()
            
        def forward(self, data):
            out = self.node_emb(data.x.squeeze())
            edge_attr = self.edge_emb(data.edge_attr)  
            out, log_probs, batch_size = self.pfgnn(node_emb=out, edge_emb=edge_attr, data=data)
            # o_out,h_out  = self.gnn(out, data.edge_index, edge_attr=edge_attr, batch=data.batch)
            # out = global_add_pool(h_out, data.batch)
            # out = self.mlp(out)
            return out, log_probs, data.batch.max()+1 #batch_size log_probs, 

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = Net(node_infeat=21, edge_infeat=4, outdim=1, dim=args.dim, depth=args.depth, num_particles=args.num_particles, gnn=GNN, deg=deg)
    
    if args.parallel == True:
        loader = DataListLoader
        model =  DataParallel(model).to(device)
    else:
        loader = DataLoader
        model = model.to(device)
    
    train_loader = loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = loader(val_dataset, batch_size=args.batch_size) 
    test_loader = loader(test_dataset, batch_size=args.batch_size) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience,
                                min_lr=args.min_lr)
    print("Total parameters: ", sum(p.numel() for p in model.parameters()))

    best_val = None
    start = end = 0.
    for epoch in range(1, args.epochs):
        start = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, l1, l2 = train(model=model, loader=train_loader, optimizer=optimizer, device=device, parallel=args.parallel, regression=True)
        val_mae = test(model=model, loader=val_loader, device=device, parallel=args.parallel, regression=True)
        if best_val is None or val_mae <= best_val:
            best_val = val_mae
            test_mae = test(model=model, loader=test_loader, device=device, parallel=args.parallel, regression=True)
        scheduler.step(val_mae)
        end = time.time()
        diff = end - start
        print(f'Epoch: {epoch:02d}, Time: {diff:.3f} LR: {lr:.5f} Loss: {loss:.4f}, regr-loss: {l1:.4f}, policy-loss: {l2:.4f}, Val: {val_mae:.4f}, '
            f'Test: {test_mae:.4f}')


if __name__ == "__main__":
    main()
