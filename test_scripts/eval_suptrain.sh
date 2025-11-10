#!/usr/bin/env bash
# English comments only.

set -euo pipefail

# ---------------- Resolve repo root robustly ----------------
# This makes the script work no matter where you run it from.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${REPO_ROOT}"

# ---------------- CLI args ----------------
DATASET="emnist"     # default dataset
SUP_TRAIN="mlp"      # gl | mlp (default: mlp)
DRY_RUN="${DRY_RUN:-0}"
EPOCHS=()            # will be filled by --epoch

print_help() {
  echo "Usage: $0 [--dataset <name>] [--epoch <num[,num2,...]>] [--sup <gl|mlp>] [--dry-run]"
  echo
  echo "Options:"
  echo "  -d, --dataset <name>        Dataset name (default: emnist)"
  echo "  -e, --epoch <nums>          One or more epoch numbers, e.g.:"
  echo "                              - \"100,200,500\" (comma-separated)"
  echo "                              - -e 100 -e 200 (repeatable)"
  echo "  -s, --sup, --sup-train      'gl' or 'mlp' (default: mlp)"
  echo "      --dry-run               Print commands without executing"
  echo "  -h, --help                  Show this help"
  echo
  echo "Notes:"
  echo "  * Each epoch only affects the checkpoint filename (ckpt_epoch_<E>.pth)."
  echo "  * --sup selects SupTrain folder and sets --sup_train_type for evaluate.py."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)
      DATASET="${2:-emnist}"
      shift 2
      ;;
    --epoch|-e)
      RAW_EPOCHS="${2:-}"
      if [[ -z "${RAW_EPOCHS}" ]]; then
        echo "Error: --epoch requires a value." >&2
        exit 1
      fi
      IFS=',' read -r -a TMP_ARR <<< "${RAW_EPOCHS}"
      for E in "${TMP_ARR[@]}"; do
        E_TRIM="${E//[[:space:]]/}"
        if [[ -n "${E_TRIM}" ]]; then
          if [[ "${E_TRIM}" =~ ^[0-9]+$ ]]; then
            EPOCHS+=("${E_TRIM}")
          else
            echo "Warning: ignoring non-integer epoch '${E_TRIM}'." >&2
          fi
        fi
      done
      shift 2
      ;;
    --sup|-s|--sup-train)
      SUP_TRAIN="${2:-mlp}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo
      print_help
      exit 1
      ;;
  esac
done

# Default epoch if none provided.
if [[ "${#EPOCHS[@]}" -eq 0 ]]; then
  EPOCHS=("500")
fi

# Validate SUP_TRAIN
if [[ "${SUP_TRAIN}" != "gl" && "${SUP_TRAIN}" != "mlp" ]]; then
  echo "Error: --sup must be 'gl' or 'mlp' (got '${SUP_TRAIN}')." >&2
  exit 1
fi

# --------------- Path / grids ---------------
BASE_DIR="save/SupTrain_${SUP_TRAIN}"

BSZ="1250"
AUG="ssaug_strong"

# IMPORTANT: removed the space in "resnet110"
MODELS=("resnet18" "preactresnet18" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "vgg11" "vgg13")
NUM_TRAINS=("2256" "22560" "None")
GAMMAS=("0.5")

# --------------- Common args ----------------
COMMON_ARGS=(
  --epoch 0
  --dataset "${DATASET}"
  --plot_freq_ss 25
  --cosine
  --sup_train_type "${SUP_TRAIN}"
  --epsilon 1
  # You can uncomment the next line to force a save directory:
  # --save_folder "save/eval_outputs"
)

# --------------- Loop & run -----------------
for CKPT_EPOCH in "${EPOCHS[@]}"; do
  CKPT_FILE="ckpt_epoch_${CKPT_EPOCH}.pth"

  echo "########################################"
  echo "# Processing epoch: ${CKPT_EPOCH}"
  echo "########################################"

  for CUR_MODEL in "${MODELS[@]}"; do
    for NUM_TRAIN in "${NUM_TRAINS[@]}"; do
      for GAMMA in "${GAMMAS[@]}"; do
        CP_PATH="${BASE_DIR}/${DATASET}_${CUR_MODEL}_bsz_${BSZ}_${NUM_TRAIN}_${AUG}_gamma_${GAMMA}_cosine/${CKPT_FILE}"

        echo "============================="
        echo "SupTrain: ${SUP_TRAIN}"
        echo "Running combo: DATASET=${DATASET}, MODEL=${CUR_MODEL}, NUM_TRAIN=${NUM_TRAIN}, GAMMA=${GAMMA}"
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
done
