import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import random
import copy
import torch.nn.functional as F
import pickle
class LocalUpdate(object):
    def __init__(self, dataset=None, idxs=None, ep=None, device=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=True)
        self.ep=ep
        self.device=device
    def train(self, net):
        net.train()
        # train and update
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        epoch_loss = []
        for iter in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                # print(list(log_probs.size()))
                # print(labels)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if 0:#batch_idx % 5 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def cal_local_grad(self,net):
        net.eval()
        grd=[]
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            net.zero_grad()
            log_probs = net(images)
            # print(list(log_probs.size()))
            # print(labels)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            grd.append(net.grad)
        return  sum(grd)/len(grd)
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def test(model, test_data,device):
    model.to(device)
    model.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(test_data, batch_size=64)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(device), target.cuda(device)
        else:
            data, target = data.cpu(), target.cpu()
        log_probs = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
import pickle
def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    openfile = directory + '/' + file_name
    f = open(openfile, 'wb') # Open the file, if the file does not exist, create it.
    pickle.dump(dictionnary, f) # Write data into the file.
    f.close() # Close the file.

def load_pkl(path):
    f=open(path,'rb')
    a=pickle.load(f)
    f.close()
    return a

def mkdir(path):
    import os
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('The path has been created.')
        return True
    else:
        print('The path has existed.')
        return False