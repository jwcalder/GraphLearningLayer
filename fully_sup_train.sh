#!/usr/bin/env bash
# Run FullySup.py locally for preset (MODEL, NUM_TRAIN[, CP_PATH]) tuples.
# Edit PAIRS below, then: ./fullysup_train_batch_local.sh

set -euo pipefail

# -------- Edit here --------
PAIRS=(
#   "vgg11 None"
  # "vgg13 None"
  # "resnet20 None"
  # "resnet32 None"
  # "resnet44 None"
  # "resnet56 None"
  # "resnet110 None"
  "resnet18 None"
  # "preactresnet18 None"
  # "wrn-28-2 None"
)
SLEEP_BETWEEN_JOBS=0
CONDENV=gll_compat
# ---------------------------

# Try to locate FullySup.py (current dir or parent)
if [[ ! -f "FullySup.py" && -f "../FullySup.py" ]]; then cd ..; fi
if [[ ! -f "FullySup.py" ]]; then echo "FullySup.py not found."; exit 1; fi

# Optional: conda activation
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh" || true
fi
conda activate "$CONDENV" 2>/dev/null || true

# Conservative CPU/BLAS settings (safer on older CPUs)
export ATEN_CPU_CAPABILITY=avx2
export ONEDNN_MAX_CPU_ISA=AVX2
export MKL_DEBUG_CPU_TYPE=5
export OPENBLAS_CORETYPE=HASWELL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one() {
  local MODEL="$1"
  local NUM_TRAIN="$2"
  local CP_PATH="${3:-save/PreTrain_SimCLR/${MODEL}_bsz_512_None_ssaug_strong/ckpt_epoch_1000.pth}"

  if [[ -z "$MODEL" || -z "$NUM_TRAIN" ]]; then
    echo "[SKIP] Invalid tuple: '$MODEL' '$NUM_TRAIN'"; return 0; fi
  if [[ "$NUM_TRAIN" != "None" && ! "$NUM_TRAIN" =~ ^[0-9]+$ ]]; then
    echo "[SKIP] NUM_TRAIN must be integer or 'None'."; return 0; fi
  if [[ ! -f "$CP_PATH" ]]; then
    echo "[SKIP] Checkpoint not found: $CP_PATH"; return 0; fi

  local EXTRA_ARGS=()
  [[ "$NUM_TRAIN" != "None" ]] && EXTRA_ARGS+=(--num_train "$NUM_TRAIN")

  echo "==> MODEL=$MODEL NUM_TRAIN=$NUM_TRAIN"
  set -x
  python3 FullySup.py \
    --model "${MODEL}" \
    --dataset cifar10 \
    --plot_freq_ss 50 \
    --cosine \
    --sup_train_type gl \
    --cp_load_path "${CP_PATH}" \
    "${EXTRA_ARGS[@]}"
  local ec=$?
  set +x
  echo "<== Exit code: $ec"
  (( SLEEP_BETWEEN_JOBS > 0 )) && sleep "$SLEEP_BETWEEN_JOBS"
}

for t in "${PAIRS[@]}"; do
  MODEL=""; NUM_TRAIN=""; CP_PATH=""
  # shellcheck disable=SC2086
  read -r MODEL NUM_TRAIN CP_PATH <<< $t
  run_one "$MODEL" "$NUM_TRAIN" "${CP_PATH:-}"
done

echo "Done."
