import os
import copy
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ..trainmodel.resnet import ResNet18_dpp
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# If your project uses Client base class, keep this import
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from torch.utils.data import WeightedRandomSampler


class clientAVG(Client):
    """
    Federated continual-learning client with per-task episodic memory and Q-weighted replay.
    Key behaviors:
      - Save theta_prev at the START of a task's first round (so it is available for that task's rounds).
      - Save a per-task memory file: ./client_memory/client{client_id}_task{task_id}_memory.pt
      - During training: use one random batch from each previous task's stored memory.
      - At the end of a task (task boundary), compute Q updates by iterating through the entire stored memory.
      - Persist client state (Q, Q_history, theta_prev) in ./req_data/client_{client_id}_state.pt
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainloader = None
        self.testloader = None
        self.classes_per_task = args.num_classes
        self.round_per_task = math.floor((args.global_rounds + 1) / args.tasks)
        self.num_tasks = args.tasks
        self.encoder = args.encoder
        self.delta = args.delta
        self.memory_size = args.memory_size
        self.memory_batch_size = args.memory_batch_size
        self.task_batch_size = args.batch_size
        self.epochs = args.local_epochs
        self.lr = args.local_learning_rate
        self.V = args.V
        self.Q =[0.0 for _ in range(self.num_tasks)]
        self.Q_history= []
        self.theta_prev= None

        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.strategy = "reservoir_balanced"
        # Directories for memory
        self.memory_dir = "./client_memory"
        self.req_data_dir = "./req_data"
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.req_data_dir, exist_ok=True)
        # path to client state file
        # --------------------------
        self._seed = 42 + int(self.id)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # per-client torch Generator to use for samplers / randperm, seeded by client and will also incorporate task id when used
        self._gen = torch.Generator()
        self._gen.manual_seed(self._seed)

        # iterator cache for replay loaders (avoid recreating iter(loader) each time)
        self._mem_iterators = {}


        self.state_path = os.path.join(self.req_data_dir, f"client_{self.id}_state.pt")

        if not hasattr(self, "train_time_cost"):
            self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        print(f"[Client {self.id}] Initialized clientAVG: "
            f"round_per_task={self.round_per_task}, num_tasks={self.num_tasks}, "
            f"memory_size={self.memory_size}")

        # ----------------------------------------------------------
        # 🔥 Load ALL task dataloaders ONCE at initialization
        # ----------------------------------------------------------
        self.task_memory = {t: [] for t in range(self.num_tasks)}
        self.task_memoryloaders = {t: None for t in range(self.num_tasks)}

        print(f"Loading data from client base for client{self.id}")
        for task in range(self.num_tasks):
            labels = []
            trainloader = self.task_trainloaders[task]
            for _, y in trainloader:
                labels.extend(y.cpu().tolist())
            print(f"Task {task} → Unique labels: {sorted(set(labels))}")


    def save_task_memory_from_loader(
            self,
            client_id,
            task_id,
            trainloader,
            memory_size
        ):

        if not hasattr(self, "task_memories"):
            self.task_memories = {}   # {task_id: [(x,y), ...]}

        dataset = trainloader.dataset
        dataset_len = len(dataset)

        if dataset_len == 0:
            print(f"[Client {client_id}] Warning: Empty dataset for task {task_id}")
            return

        # --------------------------
        # Debug BEFORE
        # --------------------------
        print(f"\n[DEBUG] BEFORE update — Client {client_id}, Task {task_id}")
        total_before = sum(len(self.task_memories.get(t, [])) for t in self.task_memories)
        for t in sorted(self.task_memories.keys()):
            print(f"  • Task {t}: {len(self.task_memories[t])} samples")
        print(f"  → Total stored samples BEFORE update = {total_before}\n")

        # --------------------------
        # STRATEGY
        # --------------------------
        if self.strategy == "random":
            indices = torch.randperm(dataset_len)[:memory_size]
            mem = [dataset[idx] for idx in indices]
            self.task_memories[task_id] = mem
            print(f"[Client {client_id}] (random) Stored {len(mem)} samples for task {task_id}")

        elif self.strategy == "reservoir":
            M = self.task_memories
            total_mem = sum(len(M.get(t, [])) for t in M.keys())
            j = 0
            for (x, y) in dataset:
                if total_mem < memory_size:
                    M.setdefault(task_id, []).append((x, y))
                else:
                    i = random.randint(0, total_mem + j)
                    if i < memory_size:
                        nonempty = [t for t, arr in M.items() if len(arr) > 0]
                        if not nonempty:
                            continue
                        t_remove = random.choice(nonempty)
                        idx_r = random.randint(0, len(M[t_remove]) - 1)
                        M[t_remove].pop(idx_r)
                        M.setdefault(task_id, []).append((x, y))
                j += 1
            self.task_memories = M
            print(f"[Client {client_id}] Reservoir updated for task {task_id}")

        elif self.strategy == "reservoir_balanced":
            M = self.task_memories
            total_tasks = len(M.keys()) + 1
            base = memory_size // total_tasks
            rem = memory_size % total_tasks
            print(f"[DEBUG] Balanced Reservoir: base={base}, remainder={rem}")

            # Trim old tasks
            for t in sorted(M.keys()):
                M[t] = M[t][:base]

            # Current task memory
            curr_limit = base + rem
            perm_gen = torch.Generator()
            perm_gen.manual_seed(self._seed + task_id)
            perm = torch.randperm(dataset_len, generator=perm_gen)

            curr_mem = []
            for idx in perm:
                curr_mem.append(dataset[idx])
                if len(curr_mem) >= curr_limit:
                    break
            M[task_id] = curr_mem
            self.task_memories = M
            print(f"[Client {client_id}] Balanced reservoir stored {len(curr_mem)} samples for task {task_id}")

        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        # --------------------------
        # Create DataLoader for memory
        # --------------------------
        # use num_workers=0 for determinism and to avoid multiprocessing temp dir issues
        gen = torch.Generator()
        gen.manual_seed(self._seed + 100 * task_id)

        self.task_memoryloaders[task_id] = DataLoader(
            self.task_memories[task_id],
            batch_size=self.memory_batch_size,
            shuffle=False,
            generator=gen,
            num_workers=0,
            persistent_workers=False
        )


        print(f"[Client {client_id}] Memory loader for task {task_id} now has {len(self.task_memoryloaders[task_id].dataset)} samples")

        # --------------------------
        # Debug AFTER
        # --------------------------
        total_after = sum(len(self.task_memories.get(t, [])) for t in self.task_memories)
        print(f"\n[DEBUG] AFTER update — Client {client_id}, Task {task_id}")
        for t in sorted(self.task_memories.keys()):
            print(f"  • Task {t}: {len(self.task_memories[t])} samples")
        print(f"  → Total stored samples AFTER update = {total_after}")
        print("---------------------------------------------------------\n")
        
    def _get_next_from_memloader(self, k):
        # return next batch (xs, ys) from memory loader k deterministically
        loader = self.task_memoryloaders.get(k)
        if loader is None:
            return None, None
        it = self._mem_iterators.get(k)
        if it is None:
            it = iter(loader)
            self._mem_iterators[k] = it
        try:
            return next(it)
        except StopIteration:
            # restart iterator deterministically
            it = iter(loader)
            self._mem_iterators[k] = it
            return next(it)

    # -------------------------
    # Main training method
    # -------------------------
    def train(self, round):
        """
        Local training for one federated round.
        Behavior changes at task boundaries:
          - On the first round of a task (round % round_per_task == 0), save theta_prev BEFORE training.
          - At the final round of a task ((round+1) % round_per_task == 0), save memory and update Q-values using full stored memories.
        """
        # --------------------------------------------------
        # Deterministic per-client, per-round seeding
        # --------------------------------------------------

        seed = 42 + self.id * 1000 + int(round)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        
        start_time = time.time()

        #print(f"[Client {self.id}] Starting train() for round={round}")

        # Which task are we in?
        task_number = math.floor(round / self.round_per_task)
        task_id = task_number
        #print(f"[Client {self.id}] Determined task_number={task_number}, task_id={task_id}")

        # Load (or init) persistent client state
        #client_state = self.load_or_create_client_state()
        Q = self.Q
        Q_history = self.Q_history
        theta_prev = self.theta_prev  # may be None
        print(f"Q:{Q}");
        # print(f"Q_history:{Q_history}");

        # Local training loop
        
        trainloader = self.task_trainloaders[task_id]
        for epoch in range(self.epochs):
            
            total_loss = 0.0
            correct = 0
            total = 0

            self.model.train()
            #print(f"[Client {self.id}] Starting epoch {epoch+1}/{self.epochs} for task {task_id}")

            for  batch in trainloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                self.optimizer.zero_grad()

                # Use label within-task if neces
                

                # Current task loss
                y %= self.classes_per_task
                features =  x
                features = self.encoder(x)[0] if self.encoder else x
                outputs = self.model(features, task_id=task_id)
                
                main_loss = (self.loss(outputs, y))
              
                        
                total_batch_loss = main_loss
                # Backpropagate and step
                total_batch_loss.backward()
                self.optimizer.step()
                total_loss += total_batch_loss.item()
                 # accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

          
            # Print epoch stats
            avg_loss = total_loss / (len(trainloader) if len(trainloader) > 0 else 1)
            train_acc = 100.0 * correct / total if total > 0 else 0.0
            print(f"[Client {self.id}] Task {task_id}, Epoch {epoch+1}/{self.epochs}, Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%")

        # ---------------------------
        # End of local epochs: Task-boundary actions
        # ---------------------------
        # If it's the last round of this task, save memory and update Q-values
        #self.model2 = copy.deepcopy(self.model)

        # Update training time bookkeeping
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
        print(f"[Client {self.id}] train() finished for round={round}. Time cost={(time.time() - start_time):.3f}s, total_rounds={self.train_time_cost['num_rounds']}")

    def get_train_accuracy(self):
        """
        Compute training accuracy over the client's local training data.
        """
        self.model.eval()
        correct = 0
        total = 0
        trainloader = self.load_train_data(round=0)  # or adapt as needed
        with torch.no_grad():
            for batch in trainloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                outputs = self.model(x)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        self.model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    # in client class
    def get_weights(self):
        # Return a state_dict mapping name->tensor (tensors can be on client.device).
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict):
        # state_dict tensors may be on server device; move them to the client's device
        sd_on_client = {k: v.to(self.device) for k, v in state_dict.items()}
        self.model.load_state_dict(sd_on_client)



