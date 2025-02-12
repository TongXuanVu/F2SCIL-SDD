import numpy as np
import torch
from dataloader.cifar100.cifar import Cifar_helper
from dataloader.miniimagenet.miniimagenet import MiniImageNet


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader_all, testloader_new = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader_all, testloader_new = get_new_dataloader(args, session)

    return trainset, trainloader, testloader_new, testloader_all


def get_base_dataloader(args):

    class_index = np.arange(args['base_class'])

    trainset = MiniImageNet(train=True,index=class_index, base_sess=True)
    testset = MiniImageNet(train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args['local_bs'], shuffle=True,
                                              num_workers=4, pin_memory=True)
    testloader_all = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args['local_bs'], shuffle=False, num_workers=4, pin_memory=True)
    testloader_new = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args['local_bs'], shuffle=False, num_workers=4, pin_memory=True)

    return trainset, trainloader, testloader_all, testloader_new


def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args['dataset'] + "/session_" + str(session + 1) + '.txt'

    trainset = MiniImageNet(train=True,index_path=txt_path)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args['local_bs'], shuffle=True,
                                              num_workers=4, pin_memory=True)

    # test on all encountered classes
    class_all = get_all_classes(args, session)
    class_new = get_new_classes(args, session)

    testset_all = MiniImageNet(train=False,index=class_all)
    testset_new = MiniImageNet(train=False,index=class_new)

    testloader_all = torch.utils.data.DataLoader(dataset=testset_all, batch_size=args['local_bs'], shuffle=False,
                                                 num_workers=4, pin_memory=True)
    testloader_new = torch.utils.data.DataLoader(dataset=testset_new, batch_size=args['local_bs'], shuffle=False,
                                                 num_workers=4, pin_memory=True)

    return trainset, trainloader, testloader_all, testloader_new


def get_all_classes(args, session):
    class_list = np.arange(args['base_class'] + session * args['incremental_class'])
    return class_list


def get_new_classes(args, session):
    class_list = np.arange(args['base_class'] + (session - 1) * args['incremental_class'],
                           args['base_class'] + session * args['incremental_class'])
    return class_list
