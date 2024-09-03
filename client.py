import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import random
import copy
import torch.nn.functional as F
import pickle

class Client():
    def __init__(self,dataset=None,id=None,idx=None,ep=None,device=None,init_model=None) -> None:
        self.ldr_train=DataLoader(DatasetSplit(dataset, idx), batch_size=64, shuffle=True)
        self.id=id
        self.ep=ep
        self.device=device
        self.model=copy.deepcopy(init_model)
        self.loss_func = nn.CrossEntropyLoss()
    def local_update(self):
        self.model.to(self.device)
        self.model.train()
        # train and update
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        epoch_loss = []
        for iter in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
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
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.state_dict()),
                                        num_sample=len(self.ldr_train.dataset)
                                        )
    def receive_from_edgeserver(self, shared_state_dict):
        self.model.load_state_dict(shared_state_dict)
    def cal_local_grad(self):
        self.model.to(self.device)
        self.model.eval()
        grd=[]
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(images)
            # print(list(log_probs.size()))
            # print(labels)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            gradients=[]
            for para in self.model.parameters():
                gradients.append(para.grad.view(-1))
            grd.append(torch.cat(gradients))
        result = torch.mean(torch.stack(grd), dim=0)
        return result
    def cal_local_loss(self):
        self.model.to(self.device)
        self.model.eval()
        loss_list=[]
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(images)
            # print(list(log_probs.size()))
            # print(labels)
            loss = self.loss_func(log_probs, labels)
            loss_list.append(loss.item())
        return sum(loss_list)/len(loss_list)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label