import os
import json
import numpy as np
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset          # added for TinyImageNet
from PIL import Image                         # added for TinyImageNet
import urllib.request                         # added for TinyImageNet download
import zipfile                                # added for TinyImageNet download


# ─────────────────────────────────────────────────────────────────────────────
# TINY IMAGENET  (new)
# ─────────────────────────────────────────────────────────────────────────────

def download_tinyimagenet(root="./data/tiny-imagenet-200"):
    """Download and extract TinyImageNet-200 if not already present."""
    url      = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(os.path.dirname(root), "tiny-imagenet-200.zip")
    data_dir = os.path.dirname(root)

    if os.path.isdir(os.path.join(root, "train")):
        print(f"TinyImageNet already present at {root}, skipping download.")
        return

    if not os.path.isfile(zip_path):
        print(f"Downloading TinyImageNet-200 (~240 MB) -> {zip_path}")
        os.makedirs(data_dir, exist_ok=True)

        def _progress(count, block_size, total):
            pct = count * block_size / total * 100
            print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()
    else:
        print(f"Zip already downloaded: {zip_path}")

    print(f"Extracting -> {data_dir}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    print(f"TinyImageNet ready at {root}")


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []
        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes  = len(classes)
        for cls in classes:
            img_dir = os.path.join(root, cls, 'images')
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[cls]))
        self.targets = [s[1] for s in self.samples]   # needed by split_dataset_into_tasks

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


class TinyImageNetTestDataset(Dataset):
    def __init__(self, img_dir, annotation_file, class_to_idx, transform=None):
        self.transform = transform
        self.samples   = []
        with open(annotation_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, cls = parts[0], parts[1]
                if cls in class_to_idx:
                    self.samples.append((os.path.join(img_dir, fname), class_to_idx[cls]))
        self.targets = [s[1] for s in self.samples]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


def load_tinyimagenet(root="./data/tiny-imagenet-200"):
    download_tinyimagenet(root)
    mu, sd = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)

    # train: augmentation + normalise  |  test: normalise only (no random ops)
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mu, sd),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, sd),
    ])

    train_dir = os.path.join(root, 'train')
    val_dir   = os.path.join(root, 'val', 'images')
    ann_file  = os.path.join(root, 'val', 'val_annotations.txt')
    print("Loading TinyImageNet dataset...")
    train_dataset = TinyImageNetTrainDataset(train_dir, transform=train_transform)
    test_dataset  = TinyImageNetTestDataset(val_dir, ann_file, train_dataset.class_to_idx, transform=test_transform)
    print(f"  Train: {len(train_dataset)} | Test: {len(test_dataset)} | Classes: {train_dataset.num_classes}")
    return train_dataset, test_dataset, train_dataset.num_classes


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL CODE BELOW — only load_cifar updated to add augmentation
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar(dataset_name="CIFAR10"):
    print(f"Loading {dataset_name} dataset...")

    if dataset_name == "CIFAR10":
        mu, sd = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset_name == "CIFAR100":
        mu, sd = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("dataset_name must be either 'CIFAR10' or 'CIFAR100'")

    # train: augmentation + normalise  |  test: normalise only (no random ops)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mu, sd),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, sd),
    ])

    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
        test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
        num_classes   = 10
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root="./data", train=True,  download=True, transform=train_transform)
        test_dataset  = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
        num_classes   = 100

    return train_dataset, test_dataset, num_classes


def split_dataset_into_tasks(dataset, classes_per_task=2):
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    tasks = []
    for i in range(0, num_classes, classes_per_task):
        task_classes = list(range(i, min(i + classes_per_task, num_classes)))
        task_indices = np.where(np.isin(labels, task_classes))[0].tolist()
        tasks.append({"classes": task_classes, "indices": task_indices})
    return tasks


def distribute_data_dirichlet(tasks, num_clients, dataset, alpha=0.5):
    client_data = {i: [] for i in range(num_clients)}
    targets = np.array(dataset.targets)

    for task_id, task in enumerate(tasks):
        for cls in task["classes"]:
            class_indices = np.where(targets == cls)[0].tolist()
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet([alpha] * num_clients)
            counts = (proportions * len(class_indices)).astype(int)

            remainder = len(class_indices) - counts.sum()
            for i in range(remainder):
                counts[i % num_clients] += 1

            start = 0
            for client_id in range(num_clients):
                end = start + counts[client_id]
                client_data[client_id].extend(class_indices[start:end])
                start = end

    total_samples = sum(len(v) for v in client_data.values())
    desired = total_samples // num_clients
    task_indices = [i for cls in task["classes"] for i in np.where(targets == cls)[0]]

    for client_id in range(num_clients):
        if len(client_data[client_id]) > desired:
            client_data[client_id] = random.sample(client_data[client_id], desired)
        elif len(client_data[client_id]) < desired:
            shortage = desired - len(client_data[client_id])
            extra = random.sample(task_indices, shortage)
            client_data[client_id].extend(extra)

    # Return summary distribution instead of verbose logging
    summary = {}
    for client_id in range(num_clients):
        lbls = targets[client_data[client_id]]
        unique, counts = np.unique(lbls, return_counts=True)
        summary[client_id] = dict(zip(unique, counts))

    return client_data, summary


def save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, tasks):
    task_folder = os.path.join(output_dir, f"task_{task_id}")
    os.makedirs(task_folder, exist_ok=True)

    train_folder = os.path.join(task_folder, "train")
    test_folder = os.path.join(task_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    task_config = {
        "task_id": task_id,
        "num_classes": len(tasks[task_id]["classes"]),
        "classes": tasks[task_id]["classes"],
        "train_data": {},
        "test_data": {}
    }

    for client_id, indices in train_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = train_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        client_data = {"x": np.array(images), "y": np.array(labels)}
        np.savez(os.path.join(train_folder, f"{client_id}.npz"), data=client_data)
        task_config["train_data"][f"{client_id}"] = len(indices)

    for client_id, indices in test_client_data.items():
        images, labels = [], []
        for idx in indices:
            img, label = test_dataset[idx]
            images.append(img.numpy())
            labels.append(label)
        client_data = {"x": np.array(images), "y": np.array(labels)}
        np.savez(os.path.join(test_folder, f"{client_id}.npz"), data=client_data)
        task_config["test_data"][f"{client_id}"] = len(indices)

    with open(os.path.join(task_folder, "config.json"), "w") as f:
        json.dump(task_config, f, indent=4)


if __name__ == "__main__":
    dataset_name = "TinyImageNet"   # "CIFAR10" | "CIFAR100" | "TinyImageNet"
    num_clients  = 10
    num_tasks    = 10                # ← number of tasks to split the dataset into
    alpha        = 10.0

    # load dataset first so we know the total number of classes
    if dataset_name == "TinyImageNet":
        train_dataset, test_dataset, num_classes = load_tinyimagenet()
    else:
        train_dataset, test_dataset, num_classes = load_cifar(dataset_name)

    # classes_per_task derived from num_tasks — must divide evenly
    assert num_classes % num_tasks == 0, \
        f"{dataset_name} has {num_classes} classes, not divisible by num_tasks={num_tasks}"
    classes_per_task = num_classes // num_tasks  # ← auto-computed from num_tasks

    # output_dir = f"./dataset/{dataset_name}_T{num_tasks}_C{num_clients}"
    output_dir = f"./dataset/{dataset_name}_NONIID_alpha{alpha}_clients{num_clients}_tasks{num_tasks}"

    print(f"\nDataset      : {dataset_name}")
    print(f"Total classes: {num_classes}")
    print(f"Num tasks    : {num_tasks}")
    print(f"Classes/task : {classes_per_task}")
    print(f"Num clients  : {num_clients}")
    print(f"Alpha        : {alpha}")
    print(f"Output dir   : {output_dir}\n")

    tasks = split_dataset_into_tasks(train_dataset, classes_per_task)

    for task_id, task in enumerate(tasks):
        print(f"\nTask {task_id}")

        train_client_data, train_summary = distribute_data_dirichlet([task], num_clients, train_dataset, alpha)
        test_labels = np.array(test_dataset.targets)
        test_indices = np.where(np.isin(test_labels, task["classes"]))[0].tolist()
        test_task = {"classes": task["classes"], "indices": test_indices}
        test_client_data, test_summary = distribute_data_dirichlet([test_task], num_clients, test_dataset, alpha)

        for client_id in range(num_clients):
            print(f"Client {client_id} {train_summary[client_id]}")

        save_dataset_structure(task_id, train_client_data, test_client_data, train_dataset, test_dataset, output_dir, tasks)