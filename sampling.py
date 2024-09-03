import numpy as np
from torchvision import datasets, transforms

def edge_noniid(dataset, num_edge, num_client, l=5):
     labels = dataset.train_labels.numpy()
     dict_edges = [np.array([],dtype = int) for _ in range(num_edge)]
     dict_users = [np.array([],dtype = int) for _ in range(num_client)]

     idx = np.arange(len(labels))
     idxs_labels = np.vstack((idx, labels))
     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
     idx = idxs_labels[0, :]
     
     num_shards = num_edge * l
     num_imgs = int(len(dataset) / num_shards)
     idx_shard = [i for i in range(num_shards)]

     num_item = int(len(dataset) / num_client)

     for i in range(num_edge):
          rand_set = set(np.random.choice(idx_shard, l, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
               dict_edges[i] = np.concatenate((dict_edges[i], idx[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
     for i in range(num_edge):
          for j in range(int(num_client/num_edge)):  # Changed to iterate 10 times for each edge server
               client_id = i * 10 + j
               dict_users[client_id] = set(np.random.choice(dict_edges[i], num_item, replace=False))
               dict_edges[i] = list(set(dict_edges[i]) - dict_users[client_id])

     return dict_users
def edge_noniid_cifar(dataset, num_edge, num_client, l=5):
        labels = np.array(dataset.targets)
        dict_edges = [np.array([],dtype = int) for _ in range(num_edge)]
        dict_users = [np.array([],dtype = int) for _ in range(num_client)]

        idx = np.arange(len(labels))
        idxs_labels = np.vstack((idx, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idx = idxs_labels[0, :]

        num_shards = num_edge * l
        num_imgs = int(len(dataset) / num_shards)
        idx_shard = [i for i in range(num_shards)]

        num_item = int(len(dataset) / num_client)

        for i in range(num_edge):
            rand_set = set(np.random.choice(idx_shard, l, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_edges[i] = np.concatenate((dict_edges[i], idx[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        for i in range(num_edge):
            for j in range(int(num_client/num_edge)):  # Changed to iterate 10 times for each edge server
                client_id = i * 10 + j
                dict_users[client_id] = set(np.random.choice(dict_edges[i], num_item, replace=False))
                dict_edges[i] = list(set(dict_edges[i]) - dict_users[client_id])

        return dict_users

def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
         all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def noniid(dataset, num_users,l=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * l, int(len(dataset) / (num_users * l))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, l, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def dirichlet_noniid(dataset, degree_noniid, num_users):
    train_labels = np.array(dataset.targets)

    # print(train_labels)
    num_classes = len(dataset.classes)

    label_distribution = np.random.dirichlet([degree_noniid] * num_users, num_classes)

    print(label_distribution)
    print(sum(label_distribution), sum(np.transpose(label_distribution)), sum(sum(label_distribution)))

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(num_classes)]

    dict_users = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            dict_users[i] += [idcs]

    # print(dict_users, np.shape(dict_users))

    dict_users = [set(np.concatenate(idcs)) for idcs in dict_users]

    return dict_users
if __name__ == '__main__':
    '''
     trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
     dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
     dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

     dict_users = edge_noniid(dataset_train, 4, 40, 5)
     labels = dataset_train.train_labels.numpy()
     print(np.bincount(labels,minlength=10))
     for i in range(40):
          clabel=labels[list(dict_users[i])]
          print(np.bincount(clabel,minlength=10))
    '''
    trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
    dict_users = edge_noniid_cifar(dataset_train, 4, 40, 5)
    from fedavg import save_pkl
    save_pkl(dict_users, 'data', 'dict_users_cifar_{}.pkl'.format('edge_niid'))