#!/usr/bin/env bash
#
#
# Usage:
#   ./run.sh <cifar10|cifar100|tinyimagenet> [extra python main.py args...]
#
# Example:
#   ./run.sh cifar100 -t 3          # 3 repeated runs
#   ./run.sh cifar10 -V 50 -delta 0 # override V / delta
#
set -euo pipefail

DATASET="${1:-cifar10}"
if [[ $# -gt 0 ]]; then shift; fi

# Shared defaults (Table 3 in the paper)
CLIENTS="${CLIENTS:-5}"
ALPHA="${ALPHA:-10.0}"
LR="${LR:-0.005}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
DELTA="${DELTA:-1}"
BATCH="${BATCH:-13}"
MEM_BATCH="${MEM_BATCH:-13}"
MEMORY_SIZE="${MEMORY_SIZE:-500}"
ROUNDS_PER_TASK="${ROUNDS_PER_TASK:-10}"
SEED="${SEED:-42}"

# Per-dataset settings (task count, classes per task, penalty weight V)
case "$DATASET" in
  cifar10)
    DATASET_NAME=CIFAR10
    TASKS="${TASKS:-5}"
    NB="${NB:-2}"
    V="${V:-20}"
    ;;
  cifar100)
    DATASET_NAME=CIFAR100
    TASKS="${TASKS:-5}"
    NB="${NB:-20}"
    V="${V:-200}"
    ;;
  tinyimagenet)
    DATASET_NAME=TinyImageNet
    TASKS="${TASKS:-10}"
    NB="${NB:-20}"
    V="${V:-220}"
    ;;
  *)
    echo "Usage: $0 <cifar10|cifar100|tinyimagenet> [extra main.py args...]" >&2
    exit 1
    ;;
esac

# global_rounds is chosen so that round_per_task = floor((global_rounds+1)/tasks) == ROUNDS_PER_TASK
GLOBAL_ROUNDS=$(( TASKS * ROUNDS_PER_TASK - 1 ))

DATA_PATH="${DATA_PATH:-../dataset/${DATASET_NAME}_NONIID_alpha${ALPHA}_clients${CLIENTS}_tasks${TASKS}}"

echo "=================================================="
echo "Dataset          : $DATASET_NAME"
echo "Data path         : $DATA_PATH"
echo "Tasks             : $TASKS (classes/task: $NB)"
echo "Clients           : $CLIENTS"
echo "Rounds/task       : $ROUNDS_PER_TASK (global_rounds=$GLOBAL_ROUNDS)"
echo "V / delta         : $V / $DELTA"
echo "Memory size/batch : $MEMORY_SIZE / $MEM_BATCH"
echo "=================================================="

cd "$(dirname "$0")/system"

python main.py \
  -data "$DATA_PATH" \
  -m resnet18 -algo FedAvg \
  -tasks "$TASKS" -nb "$NB" -nc "$CLIENTS" -jr 1.0 \
  -gr "$GLOBAL_ROUNDS" -lbs "$BATCH" -lr "$LR" -ls "$LOCAL_EPOCHS" \
  -V "$V" -delta "$DELTA" \
  -memory_size "$MEMORY_SIZE" -memory_batch_size "$MEM_BATCH" \
  -seed "$SEED" \
  "$@"
