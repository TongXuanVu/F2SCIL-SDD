import os
import numpy as np
import torch
from torch import nn
import time, os, math
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    new_acc = []
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    class_boundaries = range(0, nb_old + 5, increment)
    for class_id in class_boundaries:
        class_label = "{}".format(str(class_id).rjust(2, "0"))
        class_indices = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]

        if len(class_indices) > 0:
            class_accuracy = np.around(accuracy_score(y_true[class_indices], y_pred[class_indices]) * 100, decimals=2)

        else:
            class_accuracy = 0.0
        all_acc[class_label] = class_accuracy

        if class_id >= nb_old:
            new_acc.append(class_accuracy)
    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc,new_acc


def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1 )
    return F.kl_div(q, p, reduction=reduction ) * (T*T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def split_images_labels(imgs): # all img names
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def _collect_all_images(nums, it,root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                if it == None:
                    files.sort()
                    # random.shuffle(files)
                    # files = files[:nums]
                    # files = files[20*256:20*256+nums]       # discard the ealry-stage data
                    files = files[-nums:]  # 40*256
                else:
                    files.sort()
                    files = files[(it-1)*256:(it-1)*256+nums]       # discard the ealry-stage data
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join( dirpath, f) )
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root,it=None, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, it,self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
        self.root, len(self), self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        if len(input[0].shape) == 4:
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        elif len(input[0].shape) == 3:
            mean = input[0].mean([0, 2])
            var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False)
        else:
            raise ValueError(f"Unsupported input shape: {input[0].shape}")
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False)
        self._idx+=1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class UnlabeledTensorDataset(torch.utils.data.Dataset):
    def __init__(self, root, it=None, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.tensors = self._collect_all_tensors(nums, it, self.root)
        self.transform = transform
        
        self.all_data = []
        if len(self.tensors) > 0:
            for path in self.tensors:
                data = torch.load(path)
                self.all_data.append(data)
            self.all_data = torch.cat(self.all_data, dim=0)
        else:
            self.all_data = torch.empty(0)

    def _collect_all_tensors(self, nums, it, root):
        tensors = []
        for dirpath, dirnames, files in os.walk(root):
            for f in sorted(files):
                if f.endswith('.pt'):
                    tensors.append(os.path.join(dirpath, f))
        
        if nums is not None and len(tensors) > 0:
            if it is None:
                tensors = tensors[-nums:]
            else:
                tensors = tensors[(it-1)*256:(it-1)*256+nums]
        return tensors

    def __getitem__(self, idx):
        return self.all_data[idx]

    def __len__(self):
        return len(self.all_data)


class TensorPool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, tensors, targets=None):
        # tensors shape: [batch, 1, 31]
        path = os.path.join(self.root, f"{self._idx}.pt")
        torch.save(tensors.detach().cpu(), path)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledTensorDataset(self.root, transform=transform, nums=nums)
