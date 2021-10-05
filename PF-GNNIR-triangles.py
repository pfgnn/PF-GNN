import os.path as osp
import time
import argparse
import torch 
from torch.nn import Linear 
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn.inits import *
from torch_geometric.utils import degree
from worker import train, test
from model import PFGNN_Net
from gnn_models import GNN, GNN_TRIANGLES
import torch.nn.functional as F
import numpy as np
import networkx as nx
np.set_printoptions(precision=5, suppress=True,linewidth=np.inf)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PF-GNNIR')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='input batch size for training (default: 60)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--factor', type=float, default=0.7)
    parser.add_argument('--patience', type=float, default=20)
    parser.add_argument('--min_lr', type=float, default=0.000001)
    # parser.add_argument('--num_split', type=int, default=5,
    #                     help='number of splits.')
    parser.add_argument('--dim', type=int, default=150,
                        help='hidden dim. (default=150)')
    parser.add_argument('--task_name', type=str, default="ZINC")
    parser.add_argument('--parallel', type=bool, default=True,
                        help='run on multiple gpus (default: True)')
    parser.add_argument('--depth', type=int, default=2,
                        help='number of IR iterations (default: 2)')
    parser.add_argument('--num_particles', type=int, default=4,
                        help='number of particles of particle filter (default: 4)')
    args = parser.parse_args()


    class HandleNodeAttention(object):
        def __call__(self, data):
            data.attn = torch.softmax(data.x, dim=0).flatten()
            data.x = None
            return data
    transform = T.Compose([HandleNodeAttention(), T.OneHotDegree(max_degree=14)])
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TRIANGLES')
    dataset = TUDataset(path, name='TRIANGLES', use_node_attr=True,
                        transform=transform)
    train_dataset = dataset[:30000]
    val_dataset = dataset[30000:35000]
    test_dataset = dataset[35000:]
    deg = torch.zeros(14, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    class Net(torch.nn.Module):
        def __init__(self, node_infeat, outdim, dim, depth, num_particles, gnn=GNN_TRIANGLES, deg=None):
            super(Net, self).__init__()
            self.dim = dim
            self.linear = Linear(node_infeat, dim)
            self.pfgnn = PFGNN_Net(outdim=outdim, dim=dim, depth=depth, num_particles=num_particles, gnn=gnn, deg=deg)

        def reset_parameters(self):
            self.linear.reset_parameters()
            self.pfgnn.reset_parameters()
            
        def forward(self, data):
            out = F.relu(self.linear(data.x))
            edge_attr = None
            out, log_probs, batch_size = self.pfgnn(node_emb=out, edge_emb=edge_attr, data=data)
            return out, log_probs, batch_size

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = Net(node_infeat=dataset.num_features, outdim=int(dataset.num_classes+1), 
                dim=args.dim, depth=args.depth, num_particles=args.num_particles, gnn=GNN_TRIANGLES, deg=deg)
    if args.parallel is True:
        loader = DataListLoader
        model =  DataParallel(model).to(device)
    else:
        loader = DataLoader
        model = model.to(device)
    train_loader = loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = loader(val_dataset, batch_size=args.batch_size)
    test_loader = loader(test_dataset, batch_size=args.batch_size//2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                        factor=args.factor, patience=args.patience,
                                        min_lr=args.min_lr)
    print("Total parameters: ", sum(p.numel() for p in model.parameters()))
    torch.autograd.set_detect_anomaly(True)            
    start = None
    best_val_acc = None
    for epoch in range(1, args.epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        start = time.time()
        loss, l1, l2 = train(model=model, loader=train_loader, optimizer=optimizer, 
                                device=device, parallel=args.parallel, regression=False)

        train_correct, train_loss = test(model=model, loader=train_loader, device=device, parallel=args.parallel, regression=False)
        val_correct, val_loss = test(model=model, loader=val_loader, device=device, parallel=args.parallel, regression=False)
        test_correct, test_loss = test(model=model, loader=test_loader, device=device, parallel=args.parallel, regression=False)
        scheduler.step(val_loss)
        
        train_acc = train_correct.sum().item() / train_correct.size(0)
        val_acc = val_correct.sum().item() / val_correct.size(0)

        test_acc1 = test_correct[:5000].sum().item() / 5000
        test_acc2 = test_correct[5000:].sum().item() / 5000

        if best_val_acc is None or val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc_S = test_acc1
            best_test_acc_L = test_acc2

        end = time.time()
        print(('Epoch: {:03d}, Time: {:.3f}, LR: {:.5f}, Loss: {:.4f},  ValLoss: {:.4f}, TrainAcc: {:.3f}, ValAcc: {:.3f}, '
            'TestAcc Orig : {:.3f}, TestAcc Large: {:.3f}, Best Orig : {:.3f}, Best Large: {:.3f}').format(
                epoch, end-start, lr, loss, val_loss, train_acc, 
                val_acc, test_acc1, test_acc2, best_test_acc_S, best_test_acc_L))



if __name__ == "__main__":
    main()
