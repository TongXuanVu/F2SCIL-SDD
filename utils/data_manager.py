from PIL import Image
from torch.utils.data import Dataset
import torch, copy
import os, pdb, random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights2(w, cls_clnt_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))

    fc2_weight_matrix_sum = torch.zeros_like(w[0]["fc.fc2.weight"])
    for i in range(len(w)):
        expanded_fc2_weights = torch.tensor(cls_clnt_weight[i]).reshape(5,1).cuda()
        fc2_weight_matrix_sum += w[i]["fc.fc2.weight"] * expanded_fc2_weights
    w_avg["fc.fc2.weight"] = fc2_weight_matrix_sum
    return w_avg


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(np.take(y_train,dataidx,mode='clip'), return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))

    n_nets = len(net_dataidx_map)
    n_classes = len(np.unique(y_train))
    unique_classes = np.unique(y_train)
    net_cls_counts = np.zeros((n_nets, n_classes), dtype=int)
    for net_i, dataidx in net_dataidx_map.items():
        # unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        unq, unq_cnt = np.unique(np.take(y_train,dataidx,mode='clip'), return_counts=True)
        for i, cls in enumerate(unique_classes):
            if cls in unq:
                idx = np.where(unq == cls)[0][0]
                net_cls_counts[net_i, i] = unq_cnt[idx]
    return net_cls_counts


def partition_data(y_train, beta=0.4, n_parties=5):

    data_size = len(y_train)

    if beta == 0:  # for iid
        labels = np.unique(y_train)
        samples_per_class_per_party = data_size // (n_parties * len(labels))
        net_dataidx_map = {i: [] for i in range(n_parties)}
        for label in labels:
            idx_k = np.where(y_train == label)[0]
            np.random.shuffle(idx_k)
            splits = np.array_split(idx_k, n_parties)
            for i in range(n_parties):
                net_dataidx_map[i].extend(splits[i][:samples_per_class_per_party])
        for i in range(n_parties):
            np.random.shuffle(net_dataidx_map[i])

    elif beta > 0:  # for niid
        min_size = 0
        min_require_size = 1
        # label = np.unique(y_train).shape[0]
        labels = np.unique(y_train)
        net_dataidx_map = {}
        # print(labels)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in labels:
                idx_k = np.where(y_train == k)[0]
                # num_samples_k = len(idx_k)
                # print(f"Class {k}:{num_samples_k} samples")
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(  # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map


class DummyDataset(Dataset):
    def __init__(self, images, targets, trsf, use_path=False):
        assert len(images) == len(targets), "Data size error!"
        self.images = images
        self.targets = targets
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        targets = self.targets[idx]
        return idx, image, targets


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))





def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
