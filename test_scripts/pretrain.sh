#!/usr/bin/env bash
# Fail fast and show commands as they run
set -euo pipefail

# ------------------- Path setup -------------------
# Move to repo root (this script is expected in a subfolder)
cd "$(dirname "$0")/.."

# ------------------- Defaults ---------------------
# Allow overriding via environment variables before calling the script
: "${MODEL:=vgg13}"   # default model if not provided
: "${BATCH_SIZE:=1024}"         # default batch size if not provided
: "${PYTHON_BIN:=python}"      # default python executable
: "${GAMMA:=0.99}"              # default gamma if not provided
: "${NUM_TRAIN:=None}"         # default num_train; can be "None" or an integer string

# ------------------- Info -------------------------
echo "Date: $(date)"
echo "Model: ${MODEL}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gamma: ${GAMMA}"
echo "Num train: ${NUM_TRAIN}"
echo "Python: $(command -v "${PYTHON_BIN}")"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "----------------------------------------"

# ------------------- Run training -----------------
# Note: NUM_TRAIN can be "None" or an integer; your Python script should handle parsing accordingly.
"${PYTHON_BIN}" SupCon_SimCLR_Pretrain.py \
    --dataset emnist \
    --model "${MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --pretrain_method combined \
    --gamma "${GAMMA}" \
    --num_train "${NUM_TRAIN}"

# ------------------- Done -------------------------
echo "All experiments finished."
echo "Job completed on $(date)"
