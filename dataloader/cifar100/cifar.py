from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import datasets,models,transforms
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


class Cifar_helper(Dataset):
    def __init__(self, args):
        super(Cifar_helper, self).__init__()
        self.args = args
        self.train_data_root = os.path.join('data/index_list/cifar100')
        self.cifar_root = os.path.join('data')
        self.data_list = []
        for i in range(1, 10):
            self.data_list.append([os.path.join(self.train_data_root, 'session_' + str(i) + '.txt'),
                                   os.path.join(self.train_data_root, 'test_' + str(i) + '.txt')])
        self.set_dataset_variables()
        self.init_data_list()
        self.set_dataset()
        self.set_cuda_device()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_dataset_variables(self):

        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=63 / 255),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.507, 0.487, 0.441),
                                                                        (0.267, 0.256, 0.276)), ])
        # Set the pre-processing steps for test set
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.507, 0.487, 0.441),
                                                                       (0.267, 0.256, 0.276)), ])
        # Initial the dataloader
        self.trainset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=True, download=True,
                                                      transform=self.transform_train)
        self.testset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=False, download=True,
                                                     transform=self.transform_test)
        self.testset2 = torchvision.datasets.CIFAR100(root=self.cifar_root, train=False, download=True,
                                                      transform=self.transform_test)
        self.evalset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=False, download=False,
                                                     transform=self.transform_test)


    def get_data_list_from_txt(self, txt_path):

        a = open(txt_path, 'r')
        b = a.readlines()
        data_list = [int(c.strip()) for c in b]
        return data_list

    def init_data_list(self):
        self.training_list = []
        self.testing_list1 = []
        self.testing_list2 = []
        for i in range(self.args['tasks']):
            self.training_list.append(self.get_data_list_from_txt(self.data_list[i][0]))
            self.testing_list1.append(self.get_data_list_from_txt(self.data_list[i][1]))
            if i == 0:
                self.testing_list2.append(self.get_data_list_from_txt(self.data_list[i][1]))
            else:
                accum_list = []
                accum_list.extend(self.testing_list2[i - 1])
                accum_list.extend(self.get_data_list_from_txt(self.data_list[i][1]))
                self.testing_list2.append(accum_list)
            print('session', i, len(self.training_list[i]), len(self.testing_list1[i]),len(self.testing_list2[i]))

    def set_dataset(self):

        self.X_train_total = np.array(self.trainset.data)
        self.Y_train_total = np.array(self.trainset.targets)
        self.X_test_total = np.array(self.testset.data)
        self.Y_test_total = np.array(self.testset.targets)

    def get_current_phase_dataloader(self, task):
        X_train = self.X_train_total[self.training_list[task]]
        Y_train = self.Y_train_total[self.training_list[task]]
        X_test1 = self.X_test_total[self.testing_list1[task]]
        Y_test1 = self.Y_test_total[self.testing_list1[task]]
        X_test2 = self.X_test_total[self.testing_list2[task]]
        Y_test2 = self.Y_test_total[self.testing_list2[task]]

        self.trainset.data = X_train.astype('uint8')
        self.trainset.targets = Y_train
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args['local_bs'],
                                                  shuffle=True, num_workers=self.args['num_workers'])
        # Set the test dataloader
        self.testset.data = X_test1.astype('uint8')
        self.testset.targets = Y_test1
        testloader1 = torch.utils.data.DataLoader(self.testset, batch_size=self.args['local_bs'],
                                                  shuffle=False, num_workers=self.args['num_workers'])
        self.testset2.data = X_test2.astype('uint8')
        self.testset2.targets = Y_test2
        testloader2 = torch.utils.data.DataLoader(self.testset2, batch_size=self.args['local_bs'],
                                                  shuffle=False, num_workers=self.args['num_workers'])
        print(len(testloader1), len(testloader2))

        return trainloader, testloader1, testloader2
