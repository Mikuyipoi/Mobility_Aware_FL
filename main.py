import copy
import os.path

from option import args_parser
import torch
from client import Client
from edge import Edge
from server import Server
from mobility import Mobility
from sampling import edge_noniid, iid,noniid,dirichlet_noniid,edge_noniid_cifar
from net import CNNCifar,CNNMnist
from torchvision import datasets, transforms
import numpy as np
from fedavg import test
from fedavg import mkdir,save_pkl,load_pkl
import matplotlib as plt
from scheduler import scheduler
def main():
    args=args_parser()

    #save files
    filename = 'result/client{}_bs_{}_epoch{}_v{}_lyp{}_edgeagg{}_{}_{}_{}_seed{}/'.format(args.num_clients, args.num_edges, args.num_communication,args.v,args.lyp_v,args.num_edge_aggregation,args.policy,args.iid,args.dataset, args.seed)
    mkdir(filename)
    save_pkl(args, filename, 'args.pkl')
    acc_list=[]
    access_list=[]
    t_list=[]
    rho_list=[]
    energy_list=[]
    position_list=[]
    zone_list=[]
    delta_list=[]
    loss_list=[]
    #random seed setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cp_rho1=np.random.rand(args.num_clients)*0.8


    #dataset and model
    if args.dataset=='cifar':
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if os.path.exists('data/cifar_model.pkl'):
            net_global=load_pkl('data/cifar_model.pkl')
        else:
            net_global=CNNCifar()
            save_pkl(net_global,'data','cifar_model.pkl')
    if args.dataset=='fashion':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                         transform=trans_fashion_mnist)
        if os.path.exists('data/mnist_model.pkl'):
            net_global=load_pkl('data/mnist_model.pkl')
        else:
            net_global=CNNMnist()
            save_pkl(net_global,'data','mnist_model.pkl')
    if args.dataset=='mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if os.path.exists('data/mnist_model.pkl'):
            net_global=load_pkl('data/mnist_model.pkl')
        else:
            net_global=CNNMnist()
            save_pkl(net_global,'data','mnist_model.pkl')

    #data split
    if os.path.exists('data/dict_users_{}_{}.pkl'.format(args.dataset,args.iid)):
        dict_users=load_pkl('data/dict_users_{}_{}.pkl'.format(args.dataset,args.iid))
    else:
        if args.iid=='iid':
            dict_users=iid(dataset_train,args.num_clients)
        if args.iid=='niid':
            dict_users=noniid(dataset_train,args.num_clients)
        if args.iid=='edge_niid':
            dict_users=edge_noniid_cifar(dataset_train,args.num_edges,args.num_clients,args.num_local_epoch)
        if args.iid=='dirichlet':
            #degree_noniid=1， larger is more iid
            dict_users=dirichlet_noniid(dataset_train,0.1,args.num_clients)
        save_pkl(dict_users,'data','dict_users_{}_{}.pkl'.format(args.dataset,args.iid))
        print("saved datasplit")

    datasize=[len(dict_users[i]) for i in range(args.num_clients)]
    #mobility init
    if os.path.exists('data/mob_{}_{}_{}.pkl'.format(args.v,args.num_communication,args.time)):
        Mob_list=load_pkl('data/mob_{}_{}_{}.pkl'.format(args.v,args.num_communication,args.time))
        Mob=Mob_list[0]
        print('loaded')
    else:
        Mob=Mobility(args.size,args.v,int(args.num_clients/args.num_edges),args.num_edges)
        Mob_list=[]
        Mob_list.append(copy.deepcopy(Mob))
    clients=[Client(dataset_train,i,dict_users[i],args.num_local_epoch,args.device,net_global) for i in range(args.num_clients)]
    edges=[Edge(i) for i in range(args.num_edges)]
    server=Server(net_global)
    for edge in edges:
        server.send_to_edge(edge)
    lyp_Queue=[]
    lyp_Queue.append([0]*args.num_clients)
    for i in range(args.num_communication):
        for edge in edges:
            edge.refresh_edgeserver()#清空接收缓存

        grd_client=[torch.tensor(0)]*args.num_clients
        for j in range(args.num_edges):
            for idx in Mob.zone_client_id[j]:
                edges[j].send_to_client(clients[idx])
                grd_client[idx]=clients[idx].cal_local_grad()

        grd_edge=[torch.mean(torch.stack([grd_client[index] for index in Mob.zone_client_id[j]]), dim=0)  for j in range(args.num_edges) if Mob.zone_client_id[j]]
        delta=[]
        for idx in range(args.num_clients):
            delta.append(torch.norm(grd_edge[Mob.zone[idx]]-grd_client[idx],p=2))
        #print(delta)
        delta_list.append(delta)
        reciprocal_delta=[1/d for d in delta]
        edge_r_delta=[sum([reciprocal_delta[index] for index in Mob.zone_client_id[j]] ) for j in range(args.num_edges)]
        rho=[args.channel_num*reciprocal_delta[j]/edge_r_delta[Mob.zone[j]]  for j in range(args.num_clients)]
        rho=[j.item() if j<1 else 1 for j in rho]
        if i==0 and args.policy=='cp2':
            cp2_rho=copy.deepcopy(rho)
        tmax=[args.time]*args.num_clients
        copy_mob=copy.deepcopy(Mob)
        position_list.append(copy.deepcopy(Mob.client_locations))
        zone_list.append(copy.deepcopy(Mob.zone))
        if os.path.exists('data/mob_{}_{}_{}.pkl'.format(args.v, args.num_communication,args.time)):
            Mob=Mob_list[i+1]
        else:
            Mob.move(args.time)
            Mob_list.append(copy.deepcopy(Mob))
        cross_list, time_list=Mob.cal_cross_time()
        for idx,j in enumerate(cross_list):
            tmax[j]=time_list[idx]
                #如果是采用调度策略的话 只有这里需要改动 改为policy给出idx
        t_list.append(tmax)
        scd=scheduler(args,datasize,tmax,copy_mob,lyp_Queue[-1])
        if args.policy=='policy':
            idxs,energy=scd.policy()
        if args.policy=='random':
            idxs,energy=scd.random()
        if args.policy=='max':
            idxs,energy=scd.max_power_and_frequency()
        if args.policy=='energy':
            idxs,energy=scd.energy_driven()
        if args.policy=='loss':
            loss=[]
            for client in clients:
                loss.append(client.cal_local_loss())
            idxs,energy=scd.loss_driven(loss)
        if args.policy=='add':
            idxs,energy=scd.policy_add_rho(rho)
        if args.policy=='RS':
            idxs,energy=scd.rs()
        if args.policy=='constant':
            idxs,energy=scd.constant_participation()
            rho=[0.4]*args.num_clients
        if args.policy=='cp1':
            idxs,energy=scd.constant_participation()
            rho=copy.deepcopy(cp_rho1)
        if args.policy=='cp2':
            idxs,energy=scd.constant_participation()
            rho=copy.deepcopy(cp2_rho)
        energy_list.append(energy)
        a=idx_to_a(idxs,args)
        access_list.append(a)
        rho_list.append(rho)
        q=[lyp_Queue[-1][j]-a[j]+rho[j] for j in range(args.num_clients)]
        q=[j if j>0 else 0 for j in q]
        lyp_Queue.append(q)
        for idx in idxs:
            clients[idx].local_update()
            clients[idx].send_to_edgeserver(edges[Mob.previous_zone[idx]])
        for edge in edges:
            edge.aggregate()
            edge.send_to_cloudserver(server)
        #全局进行虚拟聚合计算acc
        server.aggregate()
        net_global.load_state_dict(server.shared_state_dict)
        acc, loss = test(net_global, dataset_test, args.device)
        acc_list.append(acc)
        loss_list.append(loss)
        #真实聚合再次广播模型
        if (i+1)%args.num_edge_aggregation==0:
            for edge in edges:
                server.send_to_edge(edge)
        if (i+1)%50==0:
            save_pkl(access_list, filename, 'access.pkl')
            save_pkl(energy_list, filename, 'energy.pkl')
            save_pkl(t_list, filename, 'tmax.pkl')
            save_pkl(rho_list, filename, 'rho.pkl')
            save_pkl(acc_list, filename, 'acc.pkl')
            save_pkl(lyp_Queue, filename, 'queue.pkl')
            save_pkl(position_list, filename, 'position.pkl')
            save_pkl(zone_list, filename, 'zone.pkl')
            save_pkl(Mob_list, filename, 'mob_{}_{}_{}.pkl'.format(args.v, args.num_communication, args.time))
            save_pkl(delta_list, filename, 'delta.pkl')
            save_pkl(loss_list, filename, 'loss.pkl')
            save_pkl(net_global, filename, 'net_global.pkl')

    print('access:',access_list)
    print('rho:',rho_list)
    print('energy:',energy_list)




def idx_to_a(idxs,args):
    a=[0]*args.num_clients
    for idx in idxs:
        a[idx]=1
    return a
def a_to_idx(a,args):
    idxs=[]
    for idx,i in enumerate(a):
        if i==1:
            idxs.append(idx)
    return idxs
if __name__ == '__main__':
    main()