import copy
import logging
import numpy as np
import torch
import sys
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.data_manager import DummyDataset
from torchvision import transforms
import os
import csv
import torch.nn.functional as F
import convs.modified_resnet_cifar as modified_resnet_cifar
import convs.modified_resnet_subimagenet as modified_renet_subimagenet

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = 0
        self.base_class = args.get("base_class", 60)
        self.incremental_class = args.get("incremental_class", 5)
        self._network = modified_resnet_cifar.resnet20(num_classes=self.base_class)
        # self._network = modified_renet_subimagenet.resnet18(num_classes=self.base_class)
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5
        self.args = args
        self.each_task = args["incremental_class"]
        self.seed = args["seed"]
        self.tasks = args["tasks"]
        self.wandb = args["wandb"]
        self.save_dir = args["save_dir"]
        self.save_dir2 = args["save_dir2"]
        self.dataset_name = args["dataset"]
        self.clients_root = os.path.join('run/clients')
        self.nums1 = args["nums1"]
        self.nums2 = args["nums2"]
        # ----
        args["memory_size"] = 1000
        args["memory_per_class"] = 20
        args["fixed_memory"] = False
        if not os.path.exists(self.args['log_dir']):
            os.makedirs(self.args['log_dir'])

        self.logger = logging.getLogger()
        self.logger.setLevel((logging.INFO))
        timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_save_dir = os.path.join(self.args['log_dir'], self.args['method'] + timestr + '.txt')
        fh = logging.FileHandler(log_save_dir)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        h1 = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(h1)
        self.logger.info(timestr)
        self.logger.info(args)

        if self.args['wandb'] == 1:
            self.wandb.init(project=args.project)
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = "0"
        # self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def real_build_rehearsal_memory(self):
        pass

    def combine_dataset(self, pre_dataset, cur_dataset, size):
        # correct
        idx = pre_dataset.idxs
        pre_labels = pre_dataset.dataset.targets[idx]  # label 22, wrong
        pre_data = pre_dataset.dataset.data[idx]

        idx = cur_dataset.idxs
        cur_labels = cur_dataset.dataset.targets[idx]
        cur_data = cur_dataset.dataset.images[idx]

        if size != 0:
            idxs = np.random.choice(range(len(pre_dataset.idxs)), size, replace=False)
            selected_exemplar_data, selected_exemplar_label = pre_data[idxs], pre_labels[idxs]

            combined_data = np.concatenate((cur_data, selected_exemplar_data), axis=0)
            combined_label = np.concatenate((cur_labels, selected_exemplar_label), axis=0)
            # combined_label = np.concatenate(combined_label)
            # idata = _get_idata(self.dataset_name)
            # _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
            # trsf = transforms.Compose([*_train_trsf, *_common_trsf])      
            # combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)
        else:
            combined_data = np.concatenate((cur_data, pre_data), axis=0)
            combined_label = np.concatenate((cur_labels, pre_labels), axis=0)
            # combined_data, combined_label = np.vstack((cur_dataset.images, pre_dataset.images)), np.vstack((cur_dataset.labels, pre_dataset.labels))
            # combined_label = np.concatenate(combined_label)

        combined_dataset = DummyDataset(combined_data, combined_label, use_path=False)

        return combined_dataset

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:  # false
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def set_optim(self, task, model):
        lr_backbone = 0.01
        lr_fc1 = 0.1
        lr_fc2 = 0.1
        if task == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            return optimizer
        if task > 0:
            if self.args.get("dataset") == "ciciot23":
                # For CNN1D
                ignored_params = list(map(id, model.fc.fc1.parameters()))
                ignored_params.extend(list(map(id, model.fc.fc2.parameters())))
                base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)
                
                tg_params_new = [{'params': base_params, 'lr': lr_backbone, 'weight_decay': self.args.get("custom_weight_decay", 5e-4)},
                                 {'params': model.fc.fc1.parameters(), 'lr': lr_fc1,
                                  'weight_decay': self.args.get("custom_weight_decay", 5e-4)},
                                 {'params': model.fc.fc2.parameters(), 'lr': lr_fc2,
                                  'weight_decay': self.args.get("custom_weight_decay", 5e-4)}]
            else:
                # For ResNet
                ignored_params = list(map(id, model.fc.fc1.parameters()))
                ignored_params.extend(list(map(id, model.fc.fc2.parameters())))
                ignored_params.extend(list(map(id, model.layer3.parameters())))
                base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
                base_params = filter(lambda p: p.requires_grad, base_params)

                tg_params_new = [{'params': base_params, 'lr': lr_backbone, 'weight_decay': self.args.get("custom_weight_decay", 5e-4)},
                                 {'params': model.layer3.parameters(), 'lr': lr_backbone,
                                  'weight_decay': self.args.get("custom_weight_decay", 5e-4)},
                                 {'params': model.fc.fc1.parameters(), 'lr': lr_fc1,
                                  'weight_decay': self.args.get("custom_weight_decay", 5e-4)},
                                 {'params': model.fc.fc2.parameters(), 'lr': lr_fc2,
                                  'weight_decay': self.args.get("custom_weight_decay", 5e-4)}]

            optimizer = optim.SGD(tg_params_new, nesterov=True, momentum=self.args.get("custom_momentum", 0.9),
                                  weight_decay=self.args.get("custom_weight_decay", 5e-4))
            return optimizer

    def calculate_comprehensive_metrics(self, model, testloader):
        y_pred_list = []
        y_true_list = []
        test_loss = 0.0

        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                
                loss = F.cross_entropy(outputs, labels, reduction='sum')
                test_loss += loss.item()
                
                _, y_pred = torch.max(outputs, 1)

                y_pred_list.extend(y_pred.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())

        test_loss /= len(testloader.dataset)
        accuracy = accuracy_score(y_true_list, y_pred_list)
        
        # Calculate Precision, Recall, F1
        metrics_micro = precision_recall_fscore_support(y_true_list, y_pred_list, average='micro', zero_division=0)
        metrics_macro = precision_recall_fscore_support(y_true_list, y_pred_list, average='macro', zero_division=0)
        metrics_weighted = precision_recall_fscore_support(y_true_list, y_pred_list, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true_list, y_pred_list)
        
        # Calculate FPR (Macro)
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        # Add epsilon to prevent division by zero
        FPR_per_class = FP / (FP + TN + 1e-8)
        macro_fpr = np.mean(FPR_per_class)

        results = {
            "loss": test_loss,
            "accuracy": accuracy,
            "micro_p": metrics_micro[0], "micro_r": metrics_micro[1], "micro_f1": metrics_micro[2],
            "macro_p": metrics_macro[0], "macro_r": metrics_macro[1], "macro_f1": metrics_macro[2],
            "weighted_p": metrics_weighted[0], "weighted_r": metrics_weighted[1], "weighted_f1": metrics_weighted[2],
            "macro_fpr": macro_fpr,
            "cm": cm
        }
        return results

    def calculate_accuracy(self, model, testloader):

        y_pred_list = []
        y_true_list = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, y_pred = torch.max(outputs, 1)

                y_pred_list.extend(y_pred.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())

        accuracy = accuracy_score(y_true_list, y_pred_list)

        return accuracy

    def calculate_classwise_accuracy(self, model, testloader):

        classwise_accuracy = {}

        unique_classes = np.unique(torch.tensor(testloader.dataset.targets))

        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.cuda()
                outputs = model(inputs)
                _, y_pred = torch.max(outputs, 1)
                for class_label in unique_classes:
                    class_indices = (labels.numpy() == class_label)
                    class_accuracy = accuracy_score(labels.numpy()[class_indices], y_pred.numpy()[class_indices])

                    if class_label not in classwise_accuracy:
                        classwise_accuracy[class_label] = []

                    classwise_accuracy[class_label].append(class_accuracy)

        for class_label, accuracies in classwise_accuracy.items():
            classwise_accuracy[class_label] = np.mean(accuracies)
        return classwise_accuracy

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped, new_acc = accuracy(y_pred.T[0], y_true, self._known_classes, self._total_classes, increment=1)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret, new_acc

    def eval_task(self, testloader, model):
        self.testloader = testloader
        y_pred, y_true = self._eval_cnn(self.testloader, model)
        cnn_accy, new_acc = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.testloader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, new_acc, nme_accy


    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def test(self, model, loader):
        model.eval()
        test_loss = 0
        correct = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                test_loss += F.cross_entropy(outputs, targets, size_average=False).item()

                predicts = torch.max(outputs, dim=1)[1]
                correct += predicts.eq(targets.view_as(predicts)).sum().item()

        test_loss /= len(loader.dataset)
        acc = 100. * correct / len(loader.dataset)
        print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(test_loss, acc))
        return acc, test_loss

    def _eval_cnn(self, loader, model):
        model.eval()
        y_pred, y_true = [], []
        for _, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.cuda())
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.cuda())
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        print("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )  # empty list
        self._class_means = np.zeros((self._total_classes, self.feature_dim))  # shape, (20, 64)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        # for each old class, xx
        for class_idx in range(self._known_classes):  # 0 for the first task
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        print("Constructing exemplars...({} per classes)".format(m))
        # for current task
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )  # return dataset for one class, 500 samples
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)  # get feature maps
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)  # (100, 32, 32, 3)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        print(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    def visualize_with_tsne1(self,model, data_loader):
        best_checkpoint = torch.load(
            os.path.join(self.args["model_save_dir"], 'model_incremental_' + str(self._cur_task) + '.pth'))
        model.load_state_dict(best_checkpoint['global_model'])
        model = model.cuda()
        model.eval()

        all_features = []  # 获取样本特征向量
        all_labels = []  # 如果您有标签的话
        class_centers = []
        for batch_idx, (images, targets) in enumerate(data_loader):
            features = torch.tensor(images)
            features = features.cuda()
            # mask = torch.tensor([label % 5 == 0 for label in targets])
            all_labels.append(targets)
            with torch.no_grad():
                if self.args.get("dataset") != "ciciot23":
                    for layer in [model.conv1, model.bn1, model.relu, model.layer1, model.layer2, model.layer3,
                                  model.avgpool]:
                        features = layer(features)
                features = features.view(features.size(0), -1)
                all_features.append(features)
        all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
        all_features = torch.cat(all_features, dim=0).detach().cpu().numpy()

        tsne = TSNE(n_components=2)  # 或者3，根据需要选择维度
        reduced_features = tsne.fit_transform(all_features)
        for label in np.unique(all_labels):
            class_center = np.mean(reduced_features[all_labels == label], axis=0)
            class_centers.append(class_center)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='tab20', s=20)  # 如果有标签的话
        for i, center in enumerate(class_centers):
            # label_index = i * 5
            plt.text(center[0], center[1], str(i), fontsize=8, color='black', ha='center', va='center')
        plt.title('t-SNE Visualization of Task {} Training Sample Features'.format(self._cur_task))
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar()  # 如果有标签的话
        name = "train"
        plt.savefig(os.path.join(self.args['visual'], 'figure_{}{}.png'.format(name, self._cur_task)))

    def visualize_with_tsne(self, model, data_loader):
        # best_checkpoint = torch.load(
        #     os.path.join(self.args["model_save_dir"], 'model_incremental_' + str(self._cur_task-1) + '.pth'))
        # self._old_network.load_state_dict(best_checkpoint['global_model'])
        # self._old_network = self._old_network.cuda()
        # model = copy.deepcopy(self._network)
        model = model.cuda()
        model.eval()
        all_features = []
        all_labels = []
        class_centers = []
        all_probs = []
        for batch in data_loader:
            inputs = batch
            features = torch.tensor(inputs)
            features = features.cuda()
            with torch.no_grad():
                logits = model(features)
                probs = F.softmax(logits, dim=1)

                max_probs, max_indices = torch.max(probs, dim=1)
                predicted_labels = torch.argmax(logits, dim=1)
                mask = torch.tensor([label % 5 == 0 for label in predicted_labels])
                all_probs.append(max_probs[mask])
                all_labels.append(predicted_labels[mask])
                if self.args.get("dataset") != "ciciot23":
                    for layer in [model.conv1, model.bn1, model.relu, model.layer1, model.layer2, model.layer3,
                                  model.avgpool]:
                        features = layer(features)
                features = features.view(features.size(0), -1)
                all_features.append(features[mask])
        all_probs = torch.cat(all_probs, dim=0).detach().cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
        all_features = torch.cat(all_features, dim=0).detach().cpu().numpy()

        # 应用t-SNE进行降维
        tsne = TSNE(n_components=2)  # 或者3，根据需要选择维度
        reduced_features = tsne.fit_transform(all_features)
        for label in np.unique(all_labels):
            class_center = np.mean(reduced_features[all_labels == label], axis=0)
            class_centers.append(class_center)
        # 可视化降维后的数据
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='tab20', s=20)  # 如果有标签的话
        for i, center in enumerate(class_centers):
            label_index = i * 5
            plt.text(center[0], center[1], str(label_index), fontsize=8, color='black', ha='center', va='center')
        plt.title('t-SNE Visualization of Task {} Replay Sample Features'.format(self._cur_task))
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar()
        name = "replay_feature"
        plt.savefig(os.path.join(self.args['visual'], 'figure_{}{}.png'.format(name, self._cur_task)))

        tsne = TSNE(n_components=2, perplexity=30)
        combined_features = np.column_stack((all_probs, all_features))
        reduced_features = tsne.fit_transform(combined_features)
        plt.figure(figsize=(8, 6))
        plt.scatter(combined_features[:, 0], reduced_features[:, 0], cmap='tab20', s=20)
        plt.title('t-SNE Visualization of Task {} Replay Sample Probs'.format(self._cur_task))
        plt.xlabel('Probability')
        plt.ylabel('feature')
        name = "replay_probs"
        plt.savefig(os.path.join(self.args['visual'], 'figure_{}{}.png'.format(name, self._cur_task)))

        plt.figure(figsize=(8, 6))
        hist, bins, _ = plt.hist(all_probs, bins=np.arange(0, 1.1, 0.1),
                                 color='red', alpha=0.7, align='mid', width=0.095, edgecolor='black')
        plt.title('Bar chart of probability distribution of task {}'.format(self._cur_task))
        plt.xlabel('probability')
        plt.ylabel('number of sample')
        # plt.grid(True)
        for i in range(len(hist)):
            plt.text(bins[i] + 0.05, hist[i] + 2, str(int(hist[i])), ha='center')
        name = "replay_samples"
        plt.savefig(os.path.join(self.args['visual'], 'figure_{}{}.png'.format(name, self._cur_task)))
