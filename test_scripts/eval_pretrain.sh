#!/usr/bin/env bash
cd ..

set -euo pipefail

# English comments only in code per your request.

# ---------------- CLI args ----------------
DATASET="emnist"   # default dataset
CKPT_EPOCH="500"   # default checkpoint epoch (controls filename only)
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)
      DATASET="${2:-cifar10}"
      shift 2
      ;;
    --epoch|-e)
      # Epoch number used in checkpoint filename: pretrain_joint_ckpt_epoch_<E>.pth
      CKPT_EPOCH="${2:-1000}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--dataset <name>] [--epoch <num>] [--dry-run]"
      echo "Defaults: --dataset cifar10  --epoch 1000"
      echo
      echo "Note: --epoch here only affects the checkpoint filename"
      echo "      (pretrain_joint_ckpt_epoch_<num>.pth), not the training args."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --------------- Path / grids ---------------
BASE_DIR="save/PreTrain_combined"
BSZ="512"
AUG="ssaug_strong"
# Use the epoch chosen from CLI for checkpoint filename
CKPT_FILE="pretrain_joint_ckpt_epoch_${CKPT_EPOCH}.pth"

MODELS=("resnet18" "preactresnet18")
NUM_TRAINS=("2256" "22560")
GAMMAS=("0.99")

# --------------- Common args ----------------
# Important: --epoch below is the script's original training/runtime flag for FullySup.py,
# NOT the checkpoint epoch. We keep it as-is (0) unless you want to wire it to CKPT_EPOCH too.
COMMON_ARGS=(
  --epoch 0
  --dataset "${DATASET}"
  --plot_freq_ss 25
  --cosine
  --sup_train_type gl
  --epsilon 1
)

# --------------- Loop & run -----------------
for CUR_MODEL in "${MODELS[@]}"; do
  for NUM_TRAIN in "${NUM_TRAINS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
      # <BASE_DIR>/<DATASET>_<MODEL>_bsz_<BSZ>_<NUM_TRAIN>_<AUG>_gamma_<GAMMA>/<CKPT_FILE>
      CP_PATH="${BASE_DIR}/${DATASET}_${CUR_MODEL}_bsz_${BSZ}_${NUM_TRAIN}_${AUG}_gamma_${GAMMA}/${CKPT_FILE}"

      echo "============================="
      echo "Running combo: DATASET=${DATASET}, MODEL=${CUR_MODEL}, num_train=${NUM_TRAIN}, GAMMA=${GAMMA}"
      echo "Checkpoint epoch: ${CKPT_EPOCH}"
      echo "CP_PATH: ${CP_PATH}"

      if [[ ! -f "${CP_PATH}" ]]; then
        echo "WARNING: Checkpoint not found -> ${CP_PATH}"
        echo "Skipping this combo."
        continue
      fi

      CMD=(python3 evaluate.py
           "${COMMON_ARGS[@]}"
           --model "${CUR_MODEL}"
           --num_train "${NUM_TRAIN}"
           --cp_load_path "${CP_PATH}"
      )

      echo "Command: ${CMD[*]}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "(dry run) Not executing."
      else
        "${CMD[@]}"
      fi
      echo
    done
  done
done
