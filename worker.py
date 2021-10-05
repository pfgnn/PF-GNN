import torch
import torch.nn.functional as F

classify_loss = torch.nn.CrossEntropyLoss(reduction='none')
regress_los = lf = torch.nn.L1Loss(reduction='none')

def train(model, loader, optimizer, device, parallel=True, regression=True):
    model.train()
    gamma = 0.1
    total_loss, total_loss_main, total_loss_policy = 0, 0, 0
    for data in loader:
        optimizer.zero_grad()
        if parallel == True:
            data_list = data
            y = torch.cat([data.y for data in data_list]).to(device)
        else:
            data = data.to(device)
            y = data.y
        out, ls2, batch_size = model(data)
        if regression == True:    
            loss1 = (out.squeeze() - y).abs() #.mean()
        else:
            loss1 = classify_loss(out, y.to(torch.long))
        likelihood = loss1.unsqueeze(1).detach() #- val_ls.unsqueeze(1)
        loss2 = (ls2*(likelihood)).mean(1) # ls2:b x (k*d)
        loss = loss1.mean() + (gamma * loss2.mean()) #+ ls2
        loss.backward()
        total_loss += loss.item() * batch_size.sum().item() 
        total_loss_main += loss1.mean().item() * batch_size.sum().item() 
        total_loss_policy += loss2.mean().item() * batch_size.sum().item() 
        optimizer.step()
    length = len(loader.dataset)
    return total_loss / length, total_loss_main / length, total_loss_policy / length

@torch.no_grad()
def test(model, loader, device, parallel=True, regression=True):
    model.eval()
    total_error = 0
    corrects = []
    loss_all = 0
    for data in loader:
        if parallel == True:
            data_list = data
            y = torch.cat([data.y for data in data_list]).to(device)
        else:
            data = data.to(device)
            y = data.y
        out, ls2, batch_size = model(data)
        if regression == True:
            total_error += (out.squeeze() - y).abs().sum().item()
        else:
            pred = out.max(1)[1]
            corrects.append(pred.eq(y.to(torch.long)))
            loss =  classify_loss(out, y.to(torch.long)).mean()
            loss_all += loss.item() * batch_size.sum().item() 

    if regression == True:
        return total_error / len(loader.dataset)
    else:
        return torch.cat(corrects, dim=0), loss_all/len(loader.dataset)


    

