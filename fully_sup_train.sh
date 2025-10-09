#!/usr/bin/env bash
# Run FullySup.py locally for preset trials: MODEL, NUM_TRAIN, PRETRAIN_METHOD, GAMMA, SUP_TRAIN_TYPE
# Edit TRIALS below, then: ./fullysup_train_batch_local.sh

set -euo pipefail

# -------- Edit here --------
# Format: "MODEL NUM_TRAIN PRETRAIN_METHOD GAMMA SUP_TRAIN_TYPE"
TRIALS=(
  "preactresnet18 None combined 0.5 gl"
  "preactresnet18 10000 combined 0.5 gl"
  "preactresnet18 1000 combined 0.5 gl"
)
SLEEP_BETWEEN_JOBS=0
CONDENV=gll_compat
# ---------------------------

# Try to locate FullySup.py (current dir or parent)
if [[ ! -f "FullySup.py" && -f "../FullySup.py" ]]; then cd ..; fi
if [[ ! -f "FullySup.py" ]]; then echo "FullySup.py not found."; exit 1; fi

# # Optional: conda activation
# if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
#   # shellcheck source=/dev/null
#   source "${HOME}/miniconda3/etc/profile.d/conda.sh" || true
# fi
# conda activate "$CONDENV" 2>/dev/null || true

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
  local PRETRAIN_METHOD="$3"
  local GAMMA="$4"
  local SUP_TRAIN_TYPE="$5"

  if [[ -z "$MODEL" || -z "$NUM_TRAIN" || -z "$PRETRAIN_METHOD" || -z "$GAMMA" || -z "$SUP_TRAIN_TYPE" ]]; then
    echo "[SKIP] Invalid trial: '$MODEL' '$NUM_TRAIN' '$PRETRAIN_METHOD' '$GAMMA' '$SUP_TRAIN_TYPE'"
    return 0
  fi

  if [[ "$NUM_TRAIN" != "None" && ! "$NUM_TRAIN" =~ ^[0-9]+$ ]]; then
    echo "[SKIP] NUM_TRAIN must be integer or 'None'."; return 0
  fi

  local CP_PATH="save/PreTrain_${PRETRAIN_METHOD}/${MODEL}_bsz_512_${NUM_TRAIN}_ssaug_strong_gamma_${GAMMA}/pretrain_joint_ckpt_epoch_1000.pth"
  if [[ ! -f "$CP_PATH" ]]; then
    echo "[SKIP] Checkpoint not found: $CP_PATH"; return 0
  fi

  local EXTRA_ARGS=()
  [[ "$NUM_TRAIN" != "None" ]] && EXTRA_ARGS+=(--num_train "$NUM_TRAIN")

  echo "==> MODEL=$MODEL NUM_TRAIN=$NUM_TRAIN PRETRAIN_METHOD=$PRETRAIN_METHOD GAMMA=$GAMMA SUP_TRAIN_TYPE=$SUP_TRAIN_TYPE"
  set -x
  python3 FullySup.py \
    --model "${MODEL}" \
    --dataset cifar10 \
    --plot_freq_ss 50 \
    --cosine \
    --sup_train_type "${SUP_TRAIN_TYPE}" \
    --cp_load_path "${CP_PATH}" \
    --epsilon 1 \
    "${EXTRA_ARGS[@]}"
  local ec=$?
  set +x
  echo "<== Exit code: $ec"
  (( SLEEP_BETWEEN_JOBS > 0 )) && sleep "$SLEEP_BETWEEN_JOBS"
}

for t in "${TRIALS[@]}"; do
  MODEL=""; NUM_TRAIN=""; PRETRAIN_METHOD=""; GAMMA=""; SUP_TRAIN_TYPE=""
  # shellcheck disable=SC2086
  read -r MODEL NUM_TRAIN PRETRAIN_METHOD GAMMA SUP_TRAIN_TYPE <<< $t
  run_one "$MODEL" "$NUM_TRAIN" "$PRETRAIN_METHOD" "$GAMMA" "$SUP_TRAIN_TYPE"
done

echo "Done."
