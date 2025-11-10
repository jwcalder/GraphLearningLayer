#!/usr/bin/env bash
# Fail fast and surface errors
set -Eeuo pipefail

# --- Paths & Logging ---
cd ..  # keep your original behavior
LOGDIR="logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$LOGDIR/train_${TS}.log"

# Checkpoint path (fallback to your default)
CP_PATH="${1:-save/_Sup_and_SS/SupCE_resnet110_bsz_512_method_SupCE_Sup_and_SS_supaug_strong_ssaug_strong/ckpt_epoch_1000.pth}"

# --- Helpful environment knobs (safe defaults) ---
# Make Python output line-buffered so logs flush immediately
export PYTHONUNBUFFERED=1
# Avoid oversubscribing CPU BLAS threads which can bloat memory
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Reduce glibc arena fragmentation in multi-threaded allocs
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# Slightly friendlier CUDA allocator behavior (if GPU exists)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:true}"

# --- Utility: print memory & cgroup limits snapshot ---
print_limits() {
  echo "========== MEMORY SNAPSHOT ($(date)) =========="
  # System memory
  { free -h || true; } 2>/dev/null
  { grep -E 'MemTotal|MemFree|SwapTotal|SwapFree' /proc/meminfo || true; } 2>/dev/null

  echo "---- ulimit (soft/hard) ----"
  ulimit -a || true

  # cgroup v2
  if [[ -f /sys/fs/cgroup/memory.max ]]; then
    echo "---- cgroup v2 ----"
    printf "memory.max: "; cat /sys/fs/cgroup/memory.max
    printf "memory.current: "; cat /sys/fs/cgroup/memory.current
    [[ -f /sys/fs/cgroup/memory.swap.max ]] && { printf "memory.swap.max: "; cat /sys/fs/cgroup/memory.swap.max; }
  fi
  # cgroup v1
  if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    echo "---- cgroup v1 ----"
    printf "memory.limit_in_bytes: "; cat /sys/fs/cgroup/memory/memory.limit_in_bytes
    printf "memory.usage_in_bytes: "; cat /sys/fs/cgroup/memory/memory.usage_in_bytes
    [[ -f /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes ]] && { printf "memory.memsw.limit_in_bytes: "; cat /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes; }
  fi

  # GPU snapshot (if available)
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "---- GPU (nvidia-smi) ----"
    nvidia-smi || true
  fi
  echo "==============================================="
}

# --- Trap: on exit, capture OOM footprints & exit code ---
on_exit() {
  rc=$?
  echo ""
  echo "========== TRAINING FINISHED ($(date)) =========="
  echo "Exit code: $rc"
  print_limits
  # Kernel OOM/Kill traces (may require privileges on some systems)
  if command -v dmesg >/dev/null 2>&1; then
    echo "---- dmesg (last OOM/Kill lines) ----"
    dmesg -T | egrep -i 'killed process|out of memory|oom|cgroup' | tail -n 50 || true
  fi
  # Heuristic hint for cgroup/OS kill
  if [[ "$rc" -eq 137 || "$rc" -eq 9 ]]; then
    echo "[HINT] Likely SIGKILL (OOM or cgroup memory limit). See snapshots above and the log at: $LOGFILE"
  fi
  exit "$rc"
}
trap on_exit EXIT

# --- Pre-run snapshot ---
echo "========== TRAINING START ($(date)) =========="
print_limits

# --- Run training with resource profiling ---
# /usr/bin/time -v prints Max RSS (peak resident set size)
# tee writes a full log to disk for later inspection
if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v python3 FullySup.py \
    --model resnet110 \
    --dataset cifar10 \
    --plot_freq_ss 25 \
    --cosine \
    --sup_train_type gl \
    --cp_load_path "$CP_PATH" \
    --num_train None 2>&1 | tee "$LOGFILE"
  # Preserve Python's exit code despite the pipe to tee
  rc=${PIPESTATUS[0]}
  exit "$rc"
else
  # Fallback if /usr/bin/time -v is unavailable
  python3 FullySup.py \
    --model resnet110 \
    --dataset cifar10 \
    --plot_freq_ss 25 \
    --cosine \
    --sup_train_type gl \
    --cp_load_path "$CP_PATH" \
    --num_train None 2>&1 | tee "$LOGFILE"
  rc=${PIPESTATUS[0]}
  exit "$rc"
fi

# --- Optional flags if your CLI supports them ---
# You can try reducing peak memory by passing lower-concurrency loader params.
# Uncomment IF parse_option supports these flags:
#   --num_workers 2 --prefetch_factor 2 --pin_memory False
# Also consider: --gl_update_base_epochs 5 (update base set less frequently)
