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

# import os
import shutil
# import time
import atexit
import signal
import sys


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
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.strategy = "reservoir_balanced"
        # Directories for memory
        self.memory_dir = "./client_memory"
        self.req_data_dir = "./req_data"
        # os.makedirs(self.memory_dir, exist_ok=True)
        # os.makedirs(self.req_data_dir, exist_ok=True)

        run_id = f"{int(time.time())}_{os.getpid()}"

        base_dir = f"./runs/{run_id}"

        self.memory_dir = os.path.join(base_dir, "client_memory")
        self.req_data_dir = os.path.join(base_dir, "req_data")

        # create directories
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.req_data_dir, exist_ok=True)

        # --- cleanup function ---
        def cleanup():
            print(f"[Cleanup] Removing {base_dir}")
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)

        # normal exit
        atexit.register(cleanup)

        # handle Ctrl+C / kill
        def handle_signal(sig, frame):
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
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
            perm = torch.randperm(dataset_len)
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
        for t in range (task_id+1):
            self.task_memoryloaders[t] = DataLoader(
                self.task_memories[t],
                batch_size=self.memory_batch_size,
                shuffle=True,
                num_workers=0
            )
            print(f"[Client {client_id}] Memory loader for task {t} now has {len(self.task_memoryloaders[t].dataset)} samples")

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

        # === If this is the FIRST round for this task, store theta_prev BEFORE training ===
        # We detect the first round of a task as: round % round_per_task == 0
        if ((round % self.round_per_task) == 0)  and round > 0:
            # Save current model as theta_prev (store CPU copy) this the model from server
            theta_prev_to_save = {k: v.cpu() for k, v in self.model.state_dict().items()}
            print("saving the weights in theta_prev for task{task_id} ")
            # Keep Q and Q_history as they are (no change here)
            #self.save_client_state(Q, Q_history, theta_prev_to_save)
            # Make sure theta_prev variable in memory is the one we saved
            theta_prev = theta_prev_to_save
            print(f"[Client {self.id}] Saved theta_prev BEFORE training for task {task_id}")
            
                
        old_model = None
        if task_number > 0 and theta_prev is not None:
            # Use a deep copy of the current model architecture and load theta_prev
            try:
                old_model = ResNet18_dpp(num_tasks=self.num_tasks, num_classes_per_task=self.classes_per_task).to(self.device)
                old_model.load_state_dict(theta_prev)
                old_model.eval()
                print(f"[Client {self.id}] old_model prepared from theta_prev for task {task_id}")
            except Exception as e:
                print(f"[Client {self.id}] Warning: couldn't prepare old_model: {e}")
                old_model = None


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
                
                main_loss = (self.V )* (self.loss(outputs, y))
              
                        
                total_batch_loss = main_loss


                # Memory loss: sample one batch per previous task
                if task_id > 0:
                    memory_loss = torch.tensor(0.0, device=self.device)
                    for k in range(task_id):
                        mem_loader = self.task_memoryloaders.get(k)
                      

                        if mem_loader is None:
                            #print(f"[Client {self.id}] No memory loader for previous task {k}; skipping")
                            continue
                        # get one random batch from memory loader
                        try:
                            mem_x, mem_y = self._get_next_from_memloader(k)
                            if mem_x is None:
                                continue

                        except Exception as e:
                            #print(f"[Client {self.id}] Warning: couldn't fetch batch from mem_loader for task {k}: {e}")
                            continue

                        mem_x = mem_x.to(self.device)
                        mem_y = mem_y.to(self.device)
                        mem_y %= self.classes_per_task
                        features =  mem_x
                        x=mem_x
                        features = self.encoder(x)[0] if self.encoder else x
                        # self.model.eval()
                        loss_k_current = self.loss(self.model(features, k), mem_y)
                        # self.model.train()
                        with torch.no_grad():
                            if old_model is not None:
                               
                                loss_k_prev = self.loss(old_model(features, k), mem_y)
                            else:
                                loss_k_prev = torch.tensor(0.0, device=self.device)

                        memory_loss += Q[k] * (loss_k_current - loss_k_prev)
                        #print(f"[Client {self.id}] mem task {k}: loss_cur={loss_k_current.item():.4f}, loss_prev={loss_k_prev.item() if isinstance(loss_k_prev, torch.Tensor) else loss_k_prev:.4f}, Q[{k}]={Q[k]:.4f}")

                    total_batch_loss = total_batch_loss + memory_loss

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
        if (round > 0) and (((round + 1) % self.round_per_task) == 0):
            print(f"[Client {self.id}] Detected end of task {task_id} at round {round}; saving memory ")
            #Save current task memory (from trainloader)
            #fresh_loader = self.load_train_data(round=math.floor(round/self.round_per_task ))
            seed_mem = 50000 + self.id * 100 + task_id
            random.seed(seed_mem)
            np.random.seed(seed_mem)
            torch.manual_seed(seed_mem)
            self.save_task_memory_from_loader(
            client_id=self.id,
            task_id=task_id,
            trainloader=self.task_trainloaders[task_id],
            memory_size=self.memory_size
        )

        # Update Q for all previous tasks using full stored memory
        if (task_id > 0) :
            print("Updating Q Values")
            for k in range(task_id):
                print(f"[Client {self.id}] Updating Q for previous task {k}")
                mem_loader_k = self.task_memoryloaders.get(k)
                if mem_loader_k is None:
                    print(f"[Client {self.id}] No stored memory for task {k}; skipping Q update")
                    continue

                loss_current = 0.0
                loss_prev = 0.0

                # Sum losses over the entire stored memory for task k
                with torch.no_grad():
                    for x_mem, y_mem in mem_loader_k:
                        x_mem = x_mem.to(self.device)
                        y_mem = y_mem.to(self.device)
                        y_mem %= self.classes_per_task
                        features =  x_mem
                        x=x_mem
                        features = self.encoder(x)[0] if self.encoder else x
                        pred_cur = self.model(features, k)
                        
                        loss_current += self.loss(pred_cur, y_mem).item()
                        if old_model is not None:
                            pred_prev = old_model(features, k)
                            loss_prev += self.loss(pred_prev, y_mem).item()


                # Update Q[k]
                old_Qk = Q[k]
                Q[k] = max(Q[k] + loss_current - loss_prev - self.delta, 0.0)
                print(f"[Client {self.id}] Q-update task {k}: loss_cur={loss_current:.4f}, loss_prev={loss_prev:.4f}, delta={self.delta:.4f}, Q[{k}] from {old_Qk:.4f} -> {Q[k]:.4f}")

            # record Q history and save client state (theta_prev is current model after finishing this task)
            Q_history.append(Q.copy())
            theta_prev_to_save =  theta_prev #{k: v.cpu() for k, v in self.model.state_dict().items()}#theta_prev
            self.Q = Q
            self.Q_history = Q_history
            self.theta_prev = theta_prev 
            #self.save_client_state(Q, Q_history, theta_prev_to_save)
            
            print(f"[Client {self.id}] Finished end-of-task actions for task {task_id}. Q_history length={len(Q_history)}")

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



