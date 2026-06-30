import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from convs.generator import Generator
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, record_net_data_stats, average_weights, average_weights2, \
    setup_seed
from utils.toolkit import count_parameters
import copy
import torch.nn.functional as F
from torchvision import transforms
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.Loss import CEandSCE, CEandKD, CE
import torch.nn.init as init
from utils.replay_syn import GlobalSynthesizer
from utils.synthesizers import LocalSynthesizer
from utils.toolkit import KLDiv, UnlabeledImageDataset, UnlabeledTensorDataset
import convs.modified_resnet_cifar as modified_resnet_cifar
import convs.modified_resnet_subimagenet as modified_renet_subimagenet
import convs.modified_linear as modified_linear
import convs.cnn1d as cnn1d
from convs.generator import Generator, Generator1D
import csv

dataset = "cifar100"
if dataset == "cifar100":
    synthesis_batch_size = 256
    sample_batch_size = 256
    g_steps = 50
    s_steps = 40
    kd_steps = 200
    warmup = 20
    lr_g = 0.001
    lr_z = 0.01
    T = 10
    act = 0.0
    reset_l0 = 1
    reset_bn = 0
    bn_mmt = 0.9
    tau = 1
    data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**dict(data_normalize)),
    ])

else:
    synthesis_batch_size = 256
    sample_batch_size = 256
    g_steps = 50
    s_steps = 40
    is_maml = 0
    kd_steps = 400
    kd_steps2 = 200
    warmup = 20
    lr_g = 0.001
    lr_z = 0.01
    oh = 0.1
    T = 10
    act = 0.0
    adv = 1.0
    bn = 0.1
    reset_l0 = 0
    reset_bn = 0
    bn_mmt = 0.9
    syn_round = 200
    tau = 1
    data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(84, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**dict(data_normalize)),
    ])


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


normalizer = Normalizer(**dict(data_normalize))


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = self.models[0](x)
        for model in self.models[1:]:
            logits = model(x)
            logits_total += logits
        logits_e = logits_total / len(self.models)
        return logits_e


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class TARGET(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if self.args["dataset"] == "cifar100":
            self._network = modified_resnet_cifar.resnet20(num_classes=self.base_class)
        elif self.args["dataset"] == "ciciot23":
            self._network = cnn1d.cnn1d(num_classes=self.base_class)
        else:
            self._network = modified_renet_subimagenet.resnet18(num_classes=self.base_class)

    def after_task(self):
        self._known_classes = self._total_classes

    def kd_train(self, student, teacher, criterion, optimizer):

        student.train()
        teacher.eval()
        teacher.cuda()
        student.cuda()
        loader = self.get_all_replay_data()
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def kd_train2(self, student, model_list, cls_clnt_weight_tensor, criterion, optimizer):
        student.train()
        student.cuda()
        loader = self.get_syn_data_loader()
        data_iter = DataIter(loader)
        for it in range(kd_steps):
            images = data_iter.next().cuda()
            with torch.no_grad():
                weighted_t_logit = torch.zeros((len(images), self._total_classes - self._known_classes)).cuda()
                for i in range(len(model_list)):
                    current_model = model_list[i]
                    current_model = current_model.cuda()
                    current_model.eval()
                    logits = current_model(images)[:, self._known_classes:self._total_classes]
                    weighted_t_logit += cls_clnt_weight_tensor[i, self._known_classes:self._total_classes] * logits
            s_out = student(images.detach())[:, self._known_classes:self._total_classes]
            loss_s = criterion(s_out, weighted_t_logit.detach())
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

        return student

    def reply_data_generation(self, teacher, testloader):

        nz = 256
        img_size = 32 if self.args["dataset"] == "cifar100" else 64
        if self.args["dataset"] == "mini_imagenet": img_size = 84

        img_shape = (3, 32, 32) if self.args["dataset"] == "cifar100" else (3, 84, 84)
        if self.args["dataset"] == "mini_imagenet": img_shape = (3, 84, 84)  # (3, 224, 224)
        
        if self.args["dataset"] == "ciciot23":
            img_shape = (1, 31)
            generator = Generator1D(nz=nz, ngf=64, img_size=31, nc=1).cuda()
        else:
            generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).cuda()
        teacher = teacher.cuda()
        acc = self._compute_accuracy(teacher, testloader)
        self.logger.info("replay_teacher_acc {}".format(acc))

        student = copy.deepcopy(teacher)
        student.apply(weight_init)
        in_features = student.fc.in_features
        out_features = student.fc.out_features
        self.logger.info("Student Feature {} Class {}".format(in_features, out_features))
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        synthesizer = GlobalSynthesizer(copy.deepcopy(teacher), student, generator,
                                        nz=nz, num_classes=self._total_classes, img_size=img_shape, init_dataset=None,
                                        save_dir=tmp_dir,
                                        transform=train_transform, normalizer=normalizer,
                                        synthesis_batch_size=synthesis_batch_size, sample_batch_size=sample_batch_size,
                                        iterations=g_steps, lr_g=lr_g, lr_z=lr_z,
                                        reset_l0=reset_l0, reset_bn=reset_bn,
                                        bn_mmt=bn_mmt, args=self.args)
        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                                    momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=2e-4)
        prog_bar = tqdm(range(self.args["syn_round"]))
        for _, it in enumerate(prog_bar):
            synthesizer.synthesize()  # generate synthetic data
            if it >= warmup:
                student.train()
                self.kd_train(student, teacher, criterion, optimizer)  # kd_steps
                test_acc = self._compute_accuracy(student, testloader)
                info = ("Task {},Replay Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    self._cur_task, it + 1, self.args["syn_round"], test_acc, ))
                prog_bar.set_description(info)
                scheduler.step()

        print("For task {}, Replay data generation completed! ".format(self._cur_task))

    def get_replay_data_loader(self):
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        print(" syn_bs:{}, data_dir: {}".format(self.args["syn_bs"], data_dir))
        
        DatasetClass = UnlabeledTensorDataset if self.args.get("dataset") == "ciciot23" else UnlabeledImageDataset
        replay_dataset = DatasetClass(data_dir, transform=train_transform, nums=self.nums1)
        replay_data_loader = torch.utils.data.DataLoader(
            replay_dataset, batch_size=self.args["syn_bs"], shuffle=True,
            num_workers=4, pin_memory=False, )
        return replay_data_loader

    def get_all_replay_data(self):
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        DatasetClass = UnlabeledTensorDataset if self.args.get("dataset") == "ciciot23" else UnlabeledImageDataset
        replay_dataset = DatasetClass(data_dir, transform=train_transform, nums=self.nums1)
        loader = torch.utils.data.DataLoader(
            replay_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=False, sampler=None)
        return loader

    def get_syn_data_loader(self):
        data_dir = os.path.join(self.save_dir2, "task_{}".format(self._cur_task))
        DatasetClass = UnlabeledTensorDataset if self.args.get("dataset") == "ciciot23" else UnlabeledImageDataset
        syn_dataset = DatasetClass(data_dir, transform=train_transform, nums=self.nums1)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=False)
        return syn_data_loader

    def get_replay_dataloader(self):
        cumulative_dataset = None
        DatasetClass = UnlabeledTensorDataset if self.args.get("dataset") == "ciciot23" else UnlabeledImageDataset
        
        if self.args.get("dataset") == "ciciot23":
            from dataloader.ciciot23_helper import Ciciot23_helper
            ciciot_helper = Ciciot23_helper(self.args, data_root=self.args.get('data_dir', 'C:/FederatedLearning/FL/core/data_split'))

        for task in range(self._cur_task):
            data_dir = os.path.join(self.save_dir2, "task_{}".format(task))
            if not os.path.exists(data_dir):
                data_dir = os.path.join(self.save_dir, "task_{}".format(task))

            # Dynamically calculate 1% of total training samples for this task
            if self.args.get("dataset") == "ciciot23":
                total_samples = 0
                for client_idx in range(self.args["num_users"]):
                    client_dset = ciciot_helper.get_client_train_dataset(task, client_idx)
                    if client_dset is not None:
                        total_samples += len(client_dset)
                
                # 1% of total training samples
                target_samples = int(total_samples * 0.01)
                # Convert to number of batch files (each file contains 256 samples)
                nums = (target_samples + 255) // 256
            else:
                nums = self.args['nums1'] if task == 0 else self.args['nums2']
                if not os.path.exists(os.path.join(self.save_dir2, "task_{}".format(task))) and not os.path.exists(os.path.join(self.save_dir, "task_{}".format(task))):
                    nums = 6000 if task == 0 else 500

            current_dataset = DatasetClass(data_dir, transform=train_transform, nums=nums)
            
            if cumulative_dataset is None:
                cumulative_dataset = current_dataset
            else:
                cumulative_dataset = ConcatDataset([cumulative_dataset, current_dataset])

        combined_data_loader = torch.utils.data.DataLoader(
            cumulative_dataset, batch_size=self.args["syn_bs"], shuffle=True,
            num_workers=0 if self.args.get("dataset") == "ciciot23" else 4, pin_memory=False)

        return combined_data_loader

    def syn_data_generation(self, testloader1, teacher, model_list, cls_clnt_weight_tensor):
        best_acc = 0
        best_epoch = 0
        nz = 256
        img_size = 32 if self.args["dataset"] == "cifar100" else 84
        if self.args["dataset"] == "mini_imagenet": img_size = 84
        img_shape = (3, 32, 32) if self.args["dataset"] == "cifar100" else (3, 84, 84)
        if self.args["dataset"] == "mini_imagenet": img_shape = (3, 84, 84)  # (3, 224, 224)
        
        if self.args["dataset"] == "ciciot23":
            img_shape = (1, 31)
            generator = Generator1D(nz=nz, ngf=64, img_size=31, nc=1).cuda()
        else:
            generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).cuda()
        in_features = teacher.fc.in_features
        out_features = teacher.fc.out_features
        self.logger.info("Syn_teacher Feature {} Class{}".format(in_features, out_features))

        student = copy.deepcopy(teacher)
        student.apply(weight_init)
        in_features = student.fc.in_features
        out_features = student.fc.out_features
        self.logger.info("Syn_student Feature {} Class{}".format(in_features, out_features))

        tmp_dir = os.path.join(self.save_dir2, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        synthesizer = LocalSynthesizer(model_list, student, generator, cls_clnt_weight_tensor,
                                       nz=nz, num_classes1=self._known_classes, num_classes2=self._total_classes,
                                       img_size=img_shape, save_dir=tmp_dir,
                                       transform=train_transform, normalizer=normalizer,
                                       synthesis_batch_size=synthesis_batch_size, sample_batch_size=sample_batch_size,
                                       iterations=s_steps, warmup=warmup, lr_g=lr_g, lr_z=lr_z,
                                       reset_l0=reset_l0, reset_bn=reset_bn,
                                       bn_mmt=bn_mmt, args=self.args)

        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                                    momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=2e-4)
        prog_bar = tqdm(range(self.args["syn_round2"]))
        for _, it in enumerate(prog_bar):
            synthesizer.synthesize()  # generate synthetic data
            if it >= warmup:
                self.kd_train2(student, model_list, cls_clnt_weight_tensor, criterion, optimizer)  # kd_steps
                test_acc = self._compute_accuracy(student, testloader1)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = it
                info = ("Task {},syn Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f} , best_epoch:{}"
                        .format(self._cur_task, it + 1, self.args['syn_round'], best_acc, best_epoch))
                prog_bar.set_description(info)
                scheduler.step()
        self.logger.info("For task {}, syn data generation completed! ".format(self._cur_task))

    def incremental_train(self, task, trainloader1, train_set, testloader1, testloader2):
        self._cur_task = task
        if self.args["dataset"] == "ciciot23":
            rem_classes = self.args["num_class"] - self.args["base_class"]
            inc_tasks = self.args["tasks"] - 1
            classes_per_task = [self.args["base_class"]]
            if inc_tasks > 0:
                base_inc = rem_classes // inc_tasks
                remainder = rem_classes % inc_tasks
                for i in range(inc_tasks):
                    classes_per_task.append(base_inc + (1 if i < remainder else 0))
            self._new_classes = classes_per_task[self._cur_task]
        else:
            if self._cur_task == 0:
                self._new_classes = self.args['base_class']
            else:
                self._new_classes = self.args['incremental_class']
        self._total_classes = self._known_classes + self._new_classes

        # Load weights from the previous task if resuming
        if self.args.get("mode") == "resume":
            rounds_per_task = self.args["inc_ep"] * self.args["com_round"]
            resume_task = (self.args["resume_round"] - 1) // rounds_per_task if self.args["resume_round"] > 0 else 0
            if self._cur_task == resume_task and self._cur_task > 0:
                prev_session_path = os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task - 1) + '.pth')
                if os.path.exists(prev_session_path):
                    checkpoint = torch.load(prev_session_path, map_location="cpu", weights_only=False)
                    self._network.load_state_dict(checkpoint['global_model'])
                    self.logger.info(f"=> RESUME: Đã tải thành công {prev_session_path} làm Teacher cho Task {self._cur_task}")

        if self._cur_task == 0:
            if self.args["dataset"] == "cifar100":
                self._network = modified_resnet_cifar.resnet20(num_classes=self.base_class)
            elif self.args["dataset"] == "ciciot23":
                self._network = cnn1d.cnn1d(num_classes=self.base_class)
            else:
                self._network = modified_renet_subimagenet.resnet18(num_classes=self.base_class)

            in_features = self._network.fc.in_features
            out_features = self._network.fc.out_features
            self.logger.info("Task{},Feature:{}, Class:{}".format(self._cur_task, in_features, out_features))

        elif self._cur_task == 1:
            best_checkpoint = torch.load(
                os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task-1) + '.pth'))
            self._network.load_state_dict(best_checkpoint['global_model'])
            self._old_network = copy.deepcopy(self._network)
            in_features = self._network.fc.in_features
            out_features = self._network.fc.out_features
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self._new_classes)
            new_fc.fc1.weight.data = self._network.fc.weight.data
            self._network.fc = new_fc

            self._network = self._network.cuda()
            acc = self._compute_accuracy(self._network, testloader2)
            self.logger.info("begin task{} acc {}".format(self._cur_task, acc))
            new_in_features = self._network.fc.in_features
            new_out_features = self._network.fc.out_features
            self.logger.info("Task{},Feature:{}, Class:{}".format(self._cur_task, new_in_features, new_out_features))

        else:
            self._old_network = copy.deepcopy(self._network)
            in_features = self._network.fc.in_features
            out_features1 = self._network.fc.fc1.out_features
            out_features2 = self._network.fc.fc2.out_features
            self.logger.info(
                "Task{},in_features:{}, out_features1:{},out_features2:{}".format(self._cur_task, in_features,
                                                                                  out_features1, out_features2))
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2,
                                                       self.args["incremental_class"])
            new_fc.fc1.weight.data[:out_features1] = self._network.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self._network.fc.fc2.weight.data
            new_fc.sigma.data = self._network.fc.sigma.data
            self._network.fc = new_fc
            new_in_features = self._network.fc.in_features
            new_out_features = self._network.fc.out_features
            self.logger.info("Task{},Feature:{}, Class:{}".format(self._cur_task, new_in_features, new_out_features))

        self.logger.info("Task {}, Learning on {}-{}".format(self._cur_task, self._known_classes, self._total_classes))
        self.logger.info(
            "Task {} All params: {}, Trainable params: {}".format(self._cur_task, count_parameters(self._network),
                                                                  count_parameters(self._network, True)))

        if self.args.get("mode") == "resume":
            rounds_per_task = self.args["inc_ep"] * self.args["com_round"]
            resume_task = (self.args["resume_round"] - 1) // rounds_per_task if self.args["resume_round"] > 0 else 0
            if self._cur_task < resume_task:
                self.logger.info(f"=> Bỏ qua việc huấn luyện cho Task {self._cur_task} (Chỉ dựng kiến trúc mạng).")
                return

        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if self._cur_task == 0:
            if self.args.get("dataset") == "ciciot23":
                self._incremental_local_train(train_set, testloader1, testloader2, None, self._network)
            else:
                self._base_update(self._network, trainloader1, testloader2)
            self.reply_data_generation(self._network, testloader2)
        if self._cur_task != 0:
            self.replay_data_loader = self.get_replay_dataloader()
            self._incremental_local_train(train_set, testloader1, testloader2, self._old_network, self._network)

        self.logger.info("Task{} models training finished!".format(self._cur_task))

    def _base_update(self, model, trainloader, testloader):
        best_checkpoint = None
        best_acc = 0
        lr_milestone = [60, 70]
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.args['base_lr'], nesterov=True,
                              momentum=self.args['custom_momentum'], weight_decay=self.args['custom_weight_decay'])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestone, gamma=self.args["lr_factor"])
        prog_bar = tqdm(range(self.args["base_ep"]))
        for _, com in enumerate(prog_bar):
            total_loss = 0.0
            model.train()
            scheduler.step()
            for batch_idx, (data, targets) in enumerate(trainloader):
                images, labels = data.cuda(), targets.cuda()
                optimizer.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(trainloader)

            if com % 1 == 0:
                model.eval()
                test_acc = self._compute_accuracy(model, testloader)
                if best_acc < test_acc:
                    best_acc = test_acc
                    best_checkpoint = {'global_model': copy.deepcopy(model.state_dict()),
                                       'epoch': com, 'test_acc': best_acc}

                info = ("Task {}, Epoch {}/{} =>  Test_acc {:.2f} , Loss {}".format(
                    self._cur_task, com + 1, self.args["base_ep"], best_acc, average_loss))
                prog_bar.set_description(info)

        torch.save(best_checkpoint,
                   os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task) + '.pth'))

        best_checkpoint = torch.load(
            os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task) + '.pth'))
        model.load_state_dict(best_checkpoint['global_model'])
        test_acc = self._compute_accuracy(model, testloader)
        epoch = best_checkpoint['epoch']
        self.logger.info("epoch {} ,best_acc {}".format(epoch, test_acc))

    def _incremental_local_train(self, trainset, testloader1, testloader2, teacher, model):
        best_checkpoint = None
        best_local_weights = None
        best_acc = 0
        model_list = []
        if teacher is not None:
            teacher = teacher.cuda()
        model = model.cuda()
        lr_milestone = [20, 30]
        
        num_clients = self.args["num_users"]
        
        if self.args.get("dataset") == "ciciot23":
            clnt_cls_num = np.zeros((num_clients, self._total_classes))
            from dataloader.ciciot23_helper import Ciciot23_helper
            ciciot_helper = Ciciot23_helper(self.args, data_root=self.args.get('data_dir', 'C:/FederatedLearning/FL/core/data_split'))
            for idx in range(num_clients):
                client_dset = ciciot_helper.get_client_train_dataset(self._cur_task, idx)
                if client_dset is not None:
                    targets = client_dset.targets
                    unique, counts = np.unique(targets, return_counts=True)
                    for u, c in zip(unique, counts):
                        if u < self._total_classes:
                            clnt_cls_num[idx][int(u)] = c
        else:
            user_groups = partition_data(trainset.targets, beta=self.args['beta'], n_parties=num_clients)
            clnt_cls_num = record_net_data_stats(trainset.targets, user_groups)

        self.logger.info("The samples of clients......")
        self.logger.info(clnt_cls_num)
        cls_num = np.sum(clnt_cls_num, axis=0)
        cls_clnt_weight = np.round(clnt_cls_num / (np.tile(cls_num[np.newaxis, :], (num_clients, 1)) + 1e-6),
                                   decimals=2)
        cls_clnt_weight_tensor = torch.tensor(cls_clnt_weight).cuda()
        self.logger.info("The weight of clients......")
        self.logger.info(cls_clnt_weight)

        # --- RESUME LOGIC ---
        start_it = 0
        if self.args.get("mode") == "resume":
            rounds_per_task = self.args["inc_ep"] * self.args["com_round"]
            resume_task = (self.args["resume_round"] - 1) // rounds_per_task if self.args["resume_round"] > 0 else 0
            if self._cur_task == resume_task:
                last_completed_round = self.args["resume_round"] - 1
                if last_completed_round > 0:
                    checkpoint_path = os.path.join(self.args["model_save_dir"], f"checkpoint_round_{last_completed_round}.pth")
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                        model.load_state_dict(checkpoint['global_model'])
                        self.logger.info(f"=> RESUME: Đã tải thành công Checkpoint vòng {last_completed_round}!")
                    else:
                        self.logger.info(f"=> WARNING: Không tìm thấy {checkpoint_path} để resume.")
                start_it = last_completed_round % self.args["inc_ep"]
        # --------------------

        for it in range(start_it, self.args["inc_ep"]):
            local_weights = []
            user_model = {}
            all_clients_test_acc = []
            
            for idx in range(self.args["num_users"]):
                user_model[idx] = copy.deepcopy(model)

                if self.args['loss'] == 'CE':
                    criterion = CE(num_classes=self._known_classes)
                elif self.args['loss'] == 'CEandSCE':
                    criterion = CEandSCE(alpha=1, beta=1.2, lam=1, num_classes=self._known_classes)
                elif self.args['loss'] == 'CEandKD':
                    criterion = CEandKD(alpha=5,num_classes=self._known_classes)
                else:
                    raise Exception('NO LOSS!')

                user_optimizer = self.set_optim(self._cur_task, user_model[idx])
                
                # Decay learning rate manually based on communication round 'it'
                decay_factor = 1.0
                if it >= 25:
                    decay_factor = self.args["lr_factor"] ** 2
                elif it >= 15:
                    decay_factor = self.args["lr_factor"]
                
                for g in user_optimizer.param_groups:
                    g['lr'] = g['lr'] * decay_factor
                                                          
                if self.args.get("dataset") == "ciciot23":
                    client_dset = ciciot_helper.get_client_train_dataset(self._cur_task, idx)
                    if client_dset is None or len(client_dset) == 0:
                        continue
                    local_train_loader = DataLoader(client_dset, batch_size=self.args["local_bs"], shuffle=True, num_workers=0)
                else:
                    local_train_loader = DataLoader(DatasetSplit(trainset, user_groups[idx]),
                                                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4)

                user_model[idx].train()
                if teacher is not None:
                    teacher.eval()
                
                prog_bar = tqdm(range(self.args["com_round"]))
                for _, com in enumerate(prog_bar):
                    # user_scheduler.step() # Disabled scheduler.step() since we scale LR at round level
                    if self._cur_task == 0:
                        iter_loader = enumerate(local_train_loader)
                    else:
                        iter_loader = enumerate(zip(local_train_loader, self.replay_data_loader))
                    
                    for batch_data in iter_loader:
                        if self._cur_task == 0:
                            batch_idx, (images, targets) = batch_data
                        else:
                            batch_idx, ((images, targets), syn_input) = batch_data
                            syn_input = syn_input.cuda()

                        images, labels = images.cuda(), targets.cuda()
                        user_optimizer.zero_grad()

                        output = user_model[idx](images)
                        
                        if self._cur_task == 0:
                            loss = F.cross_entropy(output, labels)
                        else:
                            t_out = teacher(syn_input.detach())
                            s_out = user_model[idx](syn_input)
                            loss = criterion(output, labels,  t_out, s_out)
                            
                        loss.backward()
                        user_optimizer.step()

                    if com % 1 == 0:
                        test_acc = self._compute_accuracy(user_model[idx], testloader2)
                        info = ("Task {},Client {} Epoch {}/{} =>  Test_acc {:.2f},".format(
                            self._cur_task, idx, com + 1, self.args["com_round"], test_acc, ))
                        prog_bar.set_description(info)

                local_weights.append(user_model[idx].state_dict())
                client_test_acc = self._compute_accuracy(user_model[idx], testloader2)
                all_clients_test_acc.append(client_test_acc)
                del local_train_loader
                torch.cuda.empty_cache()

            if len(local_weights) > 0:
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
            
            # --- EVALUATE AND LOG CSV ---
            metrics = self.calculate_comprehensive_metrics(model, testloader2)
            round_idx = self._cur_task * self.args["inc_ep"] + it + 1
            
            csv_file = os.path.join(self.args["log_dir"], "metrics_f2scil.csv")
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Round", "Task", "Loss", "Accuracy", "Micro_P", "Micro_R", "Micro_F1", 
                                     "Macro_P", "Macro_R", "Macro_F1", "Weighted_P", "Weighted_R", "Weighted_F1"])
                writer.writerow([round_idx, self._cur_task, metrics["loss"], metrics["accuracy"], 
                                 metrics["micro_p"], metrics["micro_r"], metrics["micro_f1"],
                                 metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"],
                                 metrics["weighted_p"], metrics["weighted_r"], metrics["weighted_f1"]])
            
            if not os.path.exists(self.args["model_save_dir"]):
                os.makedirs(self.args["model_save_dir"])
            torch.save({'global_model': copy.deepcopy(model.state_dict()), 'epoch': it, 'test_acc': metrics["accuracy"]},
                       os.path.join(self.args["model_save_dir"], f"checkpoint_round_{round_idx}.pth"))
            
            self.logger.info(" round {} ,acc {} ".format(it, all_clients_test_acc))

            if best_acc < metrics["accuracy"]:
                best_acc = metrics["accuracy"]
                best_checkpoint = {'global_model': copy.deepcopy(model.state_dict()), 'epoch': it,
                                   'test_acc': best_acc}
                best_local_weights = local_weights

        # Keep original logic for ensemble and generator
        if best_local_weights is not None:
            for i in range(len(best_local_weights)):
                net = copy.deepcopy(self._network)
                net.load_state_dict(best_local_weights[i])
                model_list.append(net)
        if len(model_list) > 0:
            ensemble_model = Ensemble(model_list)
            acc = self._compute_accuracy(ensemble_model, testloader2)
            self.logger.info("Ensemble model acc :{}".format(acc))

        self.syn_data_generation(testloader1, model, model_list, cls_clnt_weight_tensor)
        syn_dataloader = self.get_syn_data_loader()

        file_path = os.path.join(self.clients_root, "Task_{}".format(self._cur_task))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(os.path.join(file_path, 'clients.pth'), 'wb') as f:
            pickle.dump(best_local_weights, f)
            
        torch.save(best_checkpoint,
                   os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task) + '.pth'))
        best_checkpoint = torch.load(
            os.path.join(self.args["model_save_dir"], self.args['dataset'] + '_session_' + str(self._cur_task) + '.pth'))
        model.load_state_dict(best_checkpoint['global_model'])
        test_acc = self._compute_accuracy(model, testloader2)

        self.logger.info("After Task {} Test_acc {}".format(self._cur_task, test_acc))
        if self._cur_task > 0:
            self._local_finetune(testloader2, syn_dataloader, model_list)

    def _local_finetune(self, testloader2, syn_dataloader, model_list):
        syn_data_inputs = []
        syn_data_labels = []
        test_acc_tea = []
        num_new_classes = self._total_classes - self._known_classes
        all_client_new_test_acc = np.empty((0, num_new_classes), dtype=float)

        best_checkpoint = torch.load(
            os.path.join(self.args["model_save_dir"],  self.args['dataset'] + '_session_' + str(self._cur_task) + '.pth'))
        self._network.load_state_dict(best_checkpoint['global_model'])
        self._network = self._network.cuda()
        test_acc = self._compute_accuracy(self._network, testloader2)
        self.logger.info("before finetune acc {}".format(test_acc))

        global_model = copy.deepcopy(self._network)
        file_path = os.path.join(self.clients_root, "Task_{}".format(self._cur_task))
        with open(os.path.join(file_path, 'clients.pth'), 'rb') as f:
            local_weights = pickle.load(f)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        acc = self._compute_accuracy(global_model, testloader2)
        test_acc_tea.append(acc)

        for syn_input in syn_dataloader:
            syn_input = syn_input.cuda()
            self._network.eval()
            with torch.no_grad():
                output = 16 * self._network(syn_input)
                probs = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                mask = (probs.max(1).values > self.args['t']).cpu()
                filtered_inputs = syn_input.cpu()[mask]
                filtered_labels = predicted.cpu()[mask]
                syn_data_inputs.append(filtered_inputs)
                syn_data_labels.append(filtered_labels)

        syn_data_inputs = torch.cat(syn_data_inputs, dim=0)
        syn_data_labels = torch.cat(syn_data_labels, dim=0)
        new_syn_dataset = TensorDataset(syn_data_inputs, syn_data_labels)
        self.logger.info("samples:{}".format(len(new_syn_dataset)))
        new_syn_dataloader = DataLoader(new_syn_dataset, batch_size=self.args["local_bs"], shuffle=False)

        for i in range(len(model_list)):
            acc = self._compute_accuracy(model_list[i], testloader2)
            cnn_acc, new_acc, _ = self.eval_task(new_syn_dataloader, model_list[i])
            self.logger.info("Client {} ,CNN: {}".format(i, cnn_acc["grouped"]))
            all_client_new_test_acc = np.vstack([all_client_new_test_acc, new_acc])
            test_acc_tea.append(acc)
        self.logger.info(all_client_new_test_acc)
        col_sums = np.sum(all_client_new_test_acc, axis=0)
        normalized_matrix = np.around((all_client_new_test_acc / (col_sums + 1e-3)), decimals=2)
        self.logger.info(normalized_matrix)

        self.logger.info("initial acc: {}".format(test_acc_tea))
        global_weights2 = average_weights2(local_weights, normalized_matrix)
        self._network.load_state_dict(global_weights2)

        test_acc = self._compute_accuracy(self._network, testloader2)
        self.logger.info("After finetune server acc {}".format(test_acc))
        cnn_acc, new_acc, _ = self.eval_task(testloader2, self._network)
        self.logger.info("After finetune server CNN: {}".format(cnn_acc["grouped"]))

