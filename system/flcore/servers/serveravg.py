import time
from threading import Thread, Lock
import random
import numpy as np
from collections import defaultdict
from flcore.clients.clientavg_tl import clientAVG
from flcore.servers.serverbase import Server
from .helper import *
import wandb
import math
#wandb.init(project='fl', entity='naveen2112')
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.temp = 1
        self.temp_dropout_ratio = 0
        self.count = 0
        self.total_len = 0
        self.d = 100000
        self.k = 100000  # Example value for k; set this to your desired number of clients
        self.client_times = defaultdict(list)
        self.lock = Lock()
        self.aggregated_client_count = 0
        self.classes_per_task = args.num_classes
        self.round_per_task = math.floor((args.global_rounds+1)/args.tasks)
        #print(f"in seravg {self.classes_per_task}  {self.round_per_task} ")
        self.has_aggregated_first_k_clients = False

        # Initialize round_dropout_clients to track dropout clients for each round
        self.round_dropout_clients = {}

        # Select slow clients
        self.set_clients(clientObj= clientAVG,idx = 0)
        self.set_slow_clients()
       

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    
    def set_random_seed(self, round_number):
        seed = 42 + round_number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def select_clients(self):
        """Select a fraction of clients based on join_ratio and dropout_rate (deterministic)."""
        # Ensure randomness for this round is seeded BEFORE any random ops.
        # This makes selection deterministic for a given round.
        self.set_random_seed(self.current_round)

        # Use num_selected_clients as you had (keeps logic unchanged)
        num_selected_clients = max(1, int(self.num_clients))

        # Sample deterministically now that we've seeded RNG
        available_clients = random.sample(self.clients, num_selected_clients)

        selected_clients = []
        dropped_clients = []
        dropout_rate = 1  # Keep original logic (you may want to adjust this separately)

        # Pre-calc threshold and store temp dropout ratio
        threshold_val = dropout_rate * self.temp * 20
        self.temp_dropout_ratio = dropout_rate * self.temp

        # Deterministic iteration order: use the order from `available_clients`
        for i, client in enumerate(available_clients):
            if i > threshold_val:
                selected_clients.append(client)
            else:
                dropped_clients.append(client)

        if len(selected_clients) == 0:
            selected_clients = available_clients
            dropped_clients = []
            self.temp_dropout_ratio = 0

        # Store dropout clients for the current round
        self.round_dropout_clients[self.current_round] = [client.id for client in dropped_clients]

        # Keep sorted order for deterministic downstream processing
        selected_clients.sort(key=lambda client: client.id)
        dropped_clients.sort(key=lambda client: client.id)
        available_clients.sort(key=lambda client: client.id)

        print("Dropped clients for round", self.current_round, ":", self.round_dropout_clients[self.current_round])
        return selected_clients, dropped_clients, available_clients


    def ordered_dict_to_array(self, ordered_dict):
        return np.concatenate([value.flatten() for value in ordered_dict.values()])

    def train(self):
        self.set_clients(clientObj=clientAVG, idx=0)
        self.Budget = []

        for i in range(self.global_rounds + 1):
            print(f"\n-------------Round number: {i}-------------")

            s_t = time.time()
            self.current_round = i

            # Client selection
            self.selected_clients, dropped_clients, _ = self.select_clients()
            print("Selected clients:", len(self.selected_clients))

            # ================================
            # PARALLEL CLIENT TRAINING ✅
            # ================================
            threads = []
            for client in self.selected_clients:
                t = Thread(target=client.train, kwargs={"round": i})
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # ================================
            # AGGREGATION (only once)
            # ================================
            self.receive_models(req_clients=self.selected_clients)
            self.aggregate_parameters()
            self.send_models()

            # ================================
            # Evaluation
            # ================================
            print("\nEvaluate global model")
            self.evaluate(round=i)

            # ================================
            # Time tracking
            # ================================
            round_time = time.time() - s_t
            self.Budget.append(round_time)
            print('-' * 25, 'Time cost', '-' * 25, round_time)

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], 
                top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("Average time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # Optional: evaluate new clients
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print("\n-------------Fine tuning round-------------")
            self.evaluate()


    
