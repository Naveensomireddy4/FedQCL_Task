# Federated Continual Learning as a Distributed Drift-Plus-Penalty Control Problem

Official code for our paper published at the **5th Conference on Lifelong Learning Agents (CoLLAs), 2026**:

> **Federated Continual Learning as a Distributed Drift-Plus-Penalty Control Problem**
> Nazreen Shah¹, Naveen Kumar Reddy Somireddy¹, Zubair Shaban¹, Bharath B. N.², Ranjitha Prasad¹
> ¹IIIT Delhi, ²IIT Dharwad

## Overview

This repository implements **FEDQCL (Federated Queue-regulated Continual Learning)**, a federated
continual learning (FCL) algorithm that casts the stability–plasticity trade-off across clients as
a **Lyapunov drift-plus-penalty (DPP) control problem**. 


## Repository structure

```
FedQCL_Task/
├── dataset.py                        # Builds per-task, per-client federated CL datasets
└── system/
    ├── main.py                       # Entry point / experiment driver
    ├── flcore/
    │   ├── clients/
    │   │   ├── clientbase.py         # Base FL client (data loading, train/test metrics)
    │   │   ├── clientavg_tl.py       # Drift-plus-penalty client (Q-weighted replay) — main method
    │   │   └── clientavg_er.py       # Experience-replay baseline client
    │   ├── servers/
    │   │   ├── serverbase.py         # Base FL server (aggregation, evaluation, logging)
    │   │   ├── serveravg.py          # FedAvg server with parallel client training
    │   │   └── helper.py             # Client/server weight comparison utilities
    │   ├── optimizers/
    │   │   └── fedoptimizer.py
    │   └── trainmodel/
    │       ├── resnet.py             # ResNet18 with per-task classification heads (ResNet18_dpp)
    │       ├── resnet_cil.py
    │       ├── custom_resnet18.py
    │       ├── models.py
    │       └── whisper.py
    └── utils/
        ├── data_utils.py             # Reads per-client/per-task .npz shards
        ├── privacy.py                # Differential privacy (Opacus)
        ├── dlg.py                    # Deep leakage from gradients evaluation
        ├── mem_utils.py              # GPU memory reporting
        └── result_utils.py           # Aggregation of results across runs
```

## Requirements

- Python 3.8+
- PyTorch, torchvision
- numpy, scikit-learn, h5py, matplotlib
- wandb
- calmsize (GPU memory reporting)
- Pillow

Install with:

```bash
pip install -r requirements.txt
```

## 1. Prepare the dataset

`dataset.py` downloads a vision dataset, splits its classes into sequential tasks, and partitions
each task's samples across clients with a Dirichlet non-IID split. Supported datasets: `CIFAR10`,
`CIFAR100`, `TinyImageNet`.

Edit the config block at the bottom of `dataset.py` (dataset name, number of clients, number of
tasks, Dirichlet `alpha`) and run:

```bash
python dataset.py
```

This writes per-task, per-client `.npz` shards plus a `config.json` per task under
`./dataset/<DATASET>_NONIID_alpha<alpha>_clients<N>_tasks<T>/task_<i>/{train,test}/<client_id>.npz`.

## 2. Run federated continual learning

The easiest way to run the method is via [run.sh](run.sh):

```bash
./run.sh cifar10          # Split-CIFAR-10:  5 tasks, V=20
./run.sh cifar100         # Split-CIFAR-100: 5 tasks, V=200
./run.sh tinyimagenet     # Split-TinyImageNet: 10 tasks, V=220

# override any hyperparameter via env vars or extra args, e.g.:
V=50 DELTA=0 ./run.sh cifar10 -t 3   # sweep V/delta, average over 3 seeds
```

Or call `system/main.py` directly:

```bash
cd system
python main.py \
  -data ../dataset/CIFAR10_NONIID_alpha10.0_clients5_tasks5 \
  -m resnet18 -algo FedAvg \
  -tasks 5 -nb 2 -nc 5 -jr 1.0 \
  -gr 49 -lbs 13 -lr 0.005 -ls 5 \
  -V 20 -delta 1 -memory_size 500 -memory_batch_size 13
```

### Key arguments

| Flag | Meaning | Default |
|---|---|---|
| `-data` | Path to the prepared task/client dataset | `CIFAR10` |
| `-m` | Backbone (`resnet18` is the DPP-ready model) | `cnn` |
| `-algo` | Federated aggregation algorithm | `FedAvg` |
| `-tasks` | Number of sequential tasks | `10` |
| `-nb` | Number of classes per task | `10` |
| `-nc` | Total number of clients (paper uses `5`) | `20` |
| `-jr` | Fraction of clients sampled per round | `1.0` |
| `-gr` | Total global communication rounds (split evenly across tasks) | `2000` |
| `-ls` | Local epochs per round | `1` |
| `-lbs` | Local batch size | `10` |
| `-lr` | Local learning rate | `0.005` |
| `-V` | Penalty weight on the current-task loss | `1` |
| `-delta` | Target drift / slack in the queue update | `0` |
| `-memory_size` | Per-client episodic memory budget (samples, shared across seen tasks) | `10` |
| `-memory_batch_size` | Replay batch size sampled from memory per previous task | `20` |
| `-dp` | Enable differential privacy | `False` |
| `-dlg_eval` | Enable DLG (gradient leakage) evaluation | `False` |

Run `python main.py -h` for the full list of options.

Results (accuracy/loss per run) are written under the folder given by `-go`/results utilities and
averaged across `-t` repeated runs via `utils/result_utils.average_data`.


## Acknowledgements

The federated learning system code (client/server abstractions, FedAvg, privacy and DLG utilities)
builds on [PFLlib](https://github.com/TsingZ0/PFLlib) (GPL-2.0).

## Citation

```bibtex
@inproceedings{shah2026fedqcl,
  title     = {Federated Continual Learning as a Distributed Drift-Plus-Penalty Control Problem},
  author    = {Shah, Nazreen and Somireddy, Naveen Kumar Reddy and Shaban, Zubair and B. N., Bharath and Prasad, Ranjitha},
  booktitle = {5th Conference on Lifelong Learning Agents (CoLLAs)},
  year      = {2026}
}
```
