#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running personalized federated learning experiments.
Ensures full determinism across runs.
"""

# ------------------------------
# 1️⃣ Standard library imports
# ------------------------------
import os
import random
import time
import json
import copy
import warnings
import logging
import argparse
from collections import defaultdict

# ------------------------------
# 2️⃣ Third-party imports
# ------------------------------
import numpy as np
import torch
import torch._dynamo
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
# import whisper  # If needed for speech tasks

# ------------------------------
# 3️⃣ Custom imports
# ------------------------------
from flcore.servers.serveravg import FedAvg
from flcore.trainmodel.models import *
from flcore.trainmodel.whisper import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

# ------------------------------
# 4️⃣ Logger & warnings
# ------------------------------
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")

# ------------------------------
# 5️⃣ Set deterministic behavior
# ------------------------------
def set_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g



# ------------------------------
# 6️⃣ Torch Dynamo config
# ------------------------------
torch._dynamo.config.suppress_errors = True

# ------------------------------
# 7️⃣ Hyperparameters for text tasks
# ------------------------------
VOCAB_SIZE = 98635  # 98635 for AG_News, 399198 for Sogou_News
MAX_LEN = 200
EMB_DIM = 32




def run(args):
    # First prepare the dataset
    #prepare_dataset(args)

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model based on dataset
        if model_str == 'resnet18':
            
            args.encoder = None
            seed = args.seed
            g = set_determinism(seed)
            args.g =g
            args.model = ResNet18_dpp(  num_tasks=args.tasks,num_classes_per_task=args.num_classes)
        else:
            raise NotImplementedError(f"Model {model_str} not implemented")
        
        

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            # args.head = copy.deepcopy(args.model.heads)  # Copy the task-specific heads
            # args.model.heads = nn.ModuleList([nn.Identity() for _ in range(args.model.num_tasks)])  # Replace heads with identity layers
            # args.model = BaseHeadSplit(args.model, args.head)  # Ensure compatibility
            server = FedAvg(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="CIFAR10"
                       )
    parser.add_argument('-tasks', "--tasks", type=int, default=10)
    parser.add_argument('-delta', "--delta", type=float, default=0)
    parser.add_argument('-memory_size', "--memory_size", type=int, default=10)
    parser.add_argument('-memory_batch_size', "--memory_batch_size", type=int, default=20)
    parser.add_argument('-task_batch_size', "--task_batch_size", type=int, default=10)
    parser.add_argument('-V', "--V", type=float, default=1)
    parser.add_argument('-ppe', "--patterns_per_experience", type=int, default=50)
    parser.add_argument('-ss', "--sample_size", type=int, default=50)
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-seed', "--seed", type=int, default=42)
    parser.add_argument('-m', "--model", type=str, default="cnn",
                        choices=["cnn", "resnet18","whisper","alexnet"])
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")

    parser.add_argument('-alpha', "--alpha", type=float, default=0.1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
    
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
