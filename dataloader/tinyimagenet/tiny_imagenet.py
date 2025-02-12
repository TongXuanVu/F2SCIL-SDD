# coding UTF-8
import os
import os.path as osp

import numpy as np
from numpy import loadtxt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNet(Dataset):

    def __init__(self, root='data', train=True, transform=None, index_path=None, index=None, base_sess=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.img_path = os.path.join(self.root, 'tiny-imagenet-200', 'val', 'images')
        self.gt_path = os.path.join(self.root, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
        self.train_dir = os.path.join(root, 'tiny-imagenet-200/train')
        self.dir_class_ids = os.path.join(root, 'tiny-imagenet-200/wnids.txt')
        self.class_ids = loadtxt(self.dir_class_ids, dtype=str, unpack=False)
        self.class_to_idx = {}
        for label, cid in enumerate(self.class_ids):
            self.class_to_idx[cid] = label

        self.transform = transform
        self.train = train  # training set or test set
        self.data = []
        self.targets = []
        self.classes = []
        self.data2label = {}
        self._pre_operate(self.root)

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def _pre_operate(self,root):

        if self.train:
            for label, cid in enumerate(self.class_ids):
                dir_train = os.path.join(self.train_dir, cid)
                dir_train_img = os.path.join(dir_train, 'images')
                dir_scan = sorted(os.listdir(dir_train_img))
                for i, item in enumerate(dir_scan):
                    path = os.path.join(dir_train_img, item)
                    self.data.append(path)
                    self.targets.append(label)
                    self.data2label[path] = label

        else:
            with open(self.gt_path) as f:
                lines = f.readlines()
            lines = [x.strip() for x in lines]
            for x in range(len(lines)):
                tokens = lines[x].split()
                item = tokens[0]
                path = os.path.join(self.img_path, item)
                self.data.append(path)
                self.classes.append(tokens[1])
                label = self.class_to_idx[tokens[1]]
                self.targets.append(label)
                self.data2label[path] = label

    def SelectfromTxt(self, data2label, index_path):
        # select from txt file, and make cooresponding mampping.
        index = []
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for x in range(len(lines)):
            tokens = lines[x].split()
            index.append(tokens[0])
        data_tmp = []
        targets_tmp = []
        for i in index:
            name = i[:9]
            IMAGE_PATH = os.path.join(self.train_dir,name,"images")
            img_path = os.path.join(IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


# if __name__ == '__main__':
#     dataroot = '../../data'
#     txt_path = "../../data/index_list/" + "tiny_imagenet" + "/session_" + "2" + '.txt'
#     class_index = np.arange(100)
#     trainset = TinyImageNet(root=dataroot,train=False, index=class_index)
#     a = 1