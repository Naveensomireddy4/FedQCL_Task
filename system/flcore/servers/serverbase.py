import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
import math
from utils.data_utils import read_client_data
from utils.dlg import DLG
import copy
import torch
import torch.nn as nn
import numpy as np
import os
import math
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from torch.utils.data import WeightedRandomSampler



def construct_balanced_sampler(dataset):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    # deterministic generator
    g = torch.Generator()
    g.manual_seed(0)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g
    )
    return sampler


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        # convert args.device string into torch.device and store
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        # move global model to server device
        self.global_model = copy.deepcopy(args.model)
        self.global_model.to(self.device)

        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        # self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        self.round = 0
        self.classes_per_task = args.num_classes
        self.num_clients = args.num_clients
        self.round_per_task = math.floor((args.global_rounds+1)/args.tasks)
        #print(f"in serbase {self.classes_per_task}  {self.round_per_task} ")
        
    
    
    def load_train_data(self,id, batch_size=None,round=0):
        if batch_size == None:
            batch_size = self.batch_size
        #datasets=["task_0","task_1","task_2","task_3","task_4"]
        idx = math.floor(round)
        task_folder = f"task_{idx}"
        # Set self.dataset to include the full path
        dataset = os.path.join(self.dataset, task_folder) 
        #self.dataset = datasets[idx]
        # print("Constructed dataset path for loading train data:", dataset)
        train_data = read_client_data(dataset, id, is_train=True, few_shot=self.few_shot)
        sampler = construct_balanced_sampler(train_data)
        g = torch.Generator().manual_seed(0)
        return DataLoader(
            train_data,
            batch_size,
            drop_last=True,
            sampler=sampler,
            shuffle=False,
            generator=g
        )



    def load_test_data(self,id, batch_size=None,round =0):
        if batch_size == None:
            batch_size = self.batch_size
        #datasets=["task_0","task_1","task_2","task_3","task_4"]
        idx = math.floor(round)
        task_folder = f"task_{idx}"
        # Set self.dataset to include the full path
        dataset = os.path.join(self.dataset, task_folder) 
        #self.dataset = datasets[idx]    
        # print("Constructed dataset path for loading test data:", dataset)
        test_data = read_client_data(dataset, id, is_train=False)
        g = torch.Generator().manual_seed(0)
        return DataLoader(
            test_data,
            batch_size,
            drop_last=False,
            shuffle=False,
            generator=g
        )
  

    def set_clients(self, clientObj ,idx):
        self.clients = []
        print("Hello currently in serverbase")
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            print("currently in setclients in servebase")
            task_folder = f"task_{idx}"
            dataset = os.path.join(self.dataset, task_folder)
            print("Constructed dataset path in set clients:", dataset)
            train_data = read_client_data(dataset, i, is_train=True)
            test_data = read_client_data(dataset, i, is_train=False)
            print(f"Length of train samples for client {i} is {len(train_data)}")
            # trainloader = self.load_train_data(id=i,round=0)
            # testloader = self.load_test_data(id=i,round=0)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
        print("Number of clients:", len(self.clients))
        

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        np.random.seed(self.round)   # or fixed seed

        selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )


        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self,req_clients=None):
        assert (len(req_clients) > 0)
      # when droupout=0 use temp_selected_clients else use selectde_clients

        active_clients = random.sample(
        req_clients, int(len(req_clients)))#droup_ration=0.2
        
    
        
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        
        print("*"*25,"length of uploaded models","*"*25,len(self.uploaded_models))

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("Global_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model.state_dict(), model_path)  # save only weights
        #torch.save(self.global_model, model_path)
    def load_model(self):
        model_path = os.path.join("Global_models", self.dataset, self.algorithm + "_server.pt")
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        
        # Load only the weights into the model
        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.global_model.to(self.device)
        self.global_model.eval()  # optional, set to eval mode if you just want inference


    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self,round = 0):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients(round = round)
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        for c in self.clients:
            acc, num_samples, loss = c.test_metrics(round)
            total_correct += acc * num_samples
            total_samples += num_samples
            total_loss += loss * num_samples

        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples

        return avg_acc, avg_loss

    def train_metrics(self,round =0):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        total_samples = 0
        total_loss = 0.0

        for c in self.clients:
            loss, num_samples = c.train_metrics(round)
            total_samples += num_samples
            total_loss += loss * num_samples

        avg_loss = total_loss / total_samples
        return avg_loss

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None,round = 0):
        self.round = round
        idx =math.floor( round/self.round_per_task)+1   #10 is the number of rounds for rach task
        task_wise_acc = []
        for i in range(idx):
            print(f"evaluation for task {i}")
            avg_acc, test_loss = self.test_metrics(round = i)
            #train_loss = self.train_metrics(round = i)

            if acc is None:
                self.rs_test_acc.append(avg_acc)
            else:
                acc.append(avg_acc)

            # if loss is None:
            #     self.rs_train_loss.append(train_loss)
            # else:
            #     loss.append(train_loss)

            #print(f"Averaged Train Loss: {train_loss:.4f}")
            print(f"Averaged Test Accuracy: {avg_acc*100:.2f}%")
            print(f"Averaged Test Loss: {test_loss:.4f}")
            task_wise_acc.append(avg_acc)
        print(f"Task-wise accuracies: {[f'{acc*100:.2f}%' for acc in task_wise_acc]}")
        print("Mean accuracy across tasks: {:.2f}%".format(np.mean(task_wise_acc)*100))

            

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self,round = 0):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics(round = 0)
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
    
     
    def get_global_weights(self):
    # Initialize an empty dictionary to store the weights
        global_weights_dict = {}
        
        # Iterate over named parameters in the global model
        for name, param in self.global_model.named_parameters():
            # Add parameter name and its data to the dictionary
            global_weights_dict[name] = param.data
        
        # Return the dictionary containing global weights
        return global_weights_dict
