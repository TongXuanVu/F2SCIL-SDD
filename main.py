import argparse
import wandb, os
import time
from utils.data_manager import setup_seed
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from dataloader.cifar100.cifar import Cifar_helper
from dataloader.data_utils import get_dataloader
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
# Exp settings
parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
parser.add_argument('--num_workers', type=int, default='4', help='the number of workers for loading data')
parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
parser.add_argument('--save_dir', type=str, default="", help='save the base session syn data')
parser.add_argument('--save_dir2', type=str, default="", help='save the incremental sessions syn data')
parser.add_argument('--save_png', type=str, default="run/result", help='save the result')
parser.add_argument('--project', type=str, default="TARGET", help='wandb project')
parser.add_argument('--group', type=str, default="exp1", help='wandb group')
parser.add_argument('--agg', type=str, default="None", help='aggregation method')
parser.add_argument('--seed', type=int, default=1008, help='random seed')
# federated continual learning settings
parser.add_argument('--dataset', type=str, default="tinyImagenet", help='which dataset')
parser.add_argument('--tasks', type=int, default=11, help='total number of tasks')
parser.add_argument('--num_class', type=int, default=200, help='total of class')
parser.add_argument('--base_class', type=int, default=100, help='base class')
parser.add_argument('--incremental_class', type=int, default=10, help='incremental class')
parser.add_argument('--method', type=str, default="ours", help='choose a learner')
parser.add_argument('--aggregate', type=str, default="WA", help='choose a aggregate method')
parser.add_argument('--net', type=str, default="resnet20", help='choose a model')
parser.add_argument('--syn_round', type=int, default=100, help='generator epochs')
parser.add_argument('--syn_round2', type=int, default=100, help='generator epochs')
parser.add_argument('--num_users', type=int, default=5, help='num of clients')
parser.add_argument('--synthesis_batch_size', type=int, default=256, help='synthesis batch size')
parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
parser.add_argument('--syn_bs', type=int, default=25, help='syn batch size')
parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
parser.add_argument('--base_ep', type=int, default=100, help='base training epochs')

parser.add_argument('--com_round', type=int, default=15, help='local train epochs')
parser.add_argument('--inc_ep', type=int, default=1, help='communicate rounds')
parser.add_argument('--beta', type=float, default=1, help='control the degree of label skew')
parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
parser.add_argument('--nums1', type=int, default=10000, help='the num of replay data')
parser.add_argument('--nums2', type=int, default=1500, help='the num of synthetic data')
parser.add_argument('--kd', type=int, default=1, help='for kd loss')
parser.add_argument('--bn', type=int, default=1, help='for bn loss')
parser.add_argument('--oh', type=int, default=1, help='for oh loss')
parser.add_argument('--adv', type=int, default=1, help='for adv loss')
parser.add_argument('--en', type=int, default=1, help='for en loss')
parser.add_argument('--mu', type=int, default=1, help='for en loss')
parser.add_argument('--memory_size', type=int, default=300, help='the num of real data per task')
parser.add_argument('--model_save_dir', default="run/model", type=str, help='model_save_dir')
parser.add_argument('--student_save_dir', default="run/student", type=str, help='student_save_dir')
parser.add_argument('--log_dir', type=str, default=os.path.join('./log', 'Cifar100'), help='log dir')
parser.add_argument('--ckp_prefix', type=str, default='', help='Checkpoint prefix')
parser.add_argument('--visual', type=str, default='run/visual', help='Checkpoint prefix')
parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--weight_decay', default=6e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate for the 0-th phase')
parser.add_argument('--fc_lr', default=0.01, type=float, help='learning rate for the following phases')
parser.add_argument('--lr_G', default=0.001, type=float, help='learning rate for the training of generator')
parser.add_argument('--loss', default='CEandSCE', type=str, help='LOSS')
parser.add_argument('--t', default=0.8, type=float, help='Threshold value')
args = parser.parse_args()


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "ours":
        return TARGET(args)
    else:
        assert 0


def train(args):
    setup_seed(args["seed"])

    learner = get_learner(args["method"], args)

    for task in range(args["tasks"]):
        if args["dataset"] == "cifar100":
            trainloader, testloader1, testloader2 = Helper.get_current_phase_dataloader(task)
            train_set = trainloader.dataset
            learner.incremental_train(task, trainloader, train_set, testloader1, testloader2)
        else:
            trainset, trainloader, testloader1, testloader2 = get_dataloader(args, task)
            learner.incremental_train(task, trainloader, trainset, testloader1, testloader2)  # train for one task
        learner.after_task()


if __name__ == '__main__':

    start_time = time.time()
    print("Start time:", time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(start_time)))

    args.init_cls = args.base_class
    args.exp_name = f"{args.dataset}_{'base_session'}"
    if args.method == "ours":
        dir = "run"
        if not os.path.exists(dir):
            os.makedirs(dir)
        args.save_dir = os.path.join(dir, args.exp_name)
        args.save_dir2 = os.path.join(dir, "synthesis")

    args = vars(args)
    Helper = Cifar_helper(args)
    train(args)

    end_time = time.time()
    execution_time = (end_time - start_time) / 3600

    print("End time:", time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(end_time)))
    print("Execution time:", round(execution_time, 2), 'h')
