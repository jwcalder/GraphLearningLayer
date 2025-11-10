#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English comments only.

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CLI args ----------------
def _parse_epoch_list(raw: str) -> List[int]:
    """Parse a comma-separated epoch list like '100,200,500' into [100, 200, 500].
    Empty strings and non-integers are ignored."""
    if raw is None:
        return []
    out: List[int] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok:
            continue
        if tok.isdigit():
            out.append(int(tok))
        else:
            print(f"[WARN] Ignoring non-integer epoch token: '{tok}'", file=sys.stderr)
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-class accuracies across methods, with optional within-run averaging over epochs. "
            "Reports mean per-class accuracy, lists classes under 10% * N (N=1..10), "
            "and saves two CSVs plus one histogram image."
        )
    )
    parser.add_argument(
        "--dataset", "-d",
        default="emnist",
        help="Dataset name (default: emnist)"
    )
    parser.add_argument(
        "--epoch", "-e",
        default="500",
        help="One or more epochs to read, comma-separated (e.g., '100,200,500'). "
             "For each run directory, files 'epoch<E>_per_class_acc.npy' will be loaded and averaged."
    )
    parser.add_argument(
        "--sup", "--sup-train", "-s",
        dest="sup_train",
        default="mlp",
        choices=["gl", "mlp"],
        help="Choose SupTrain type: gl or mlp (default: mlp)"
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Override base directory. If unset, uses ../save/SupTrain_<sup>."
    )
    parser.add_argument(
        "--out-dir",
        default="aggregate_out",
        help="Directory to save CSVs and histogram (default: aggregate_out)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose logs."
    )
    parser.add_argument(
        "--strict-shape",
        action="store_true",
        help="Abort on arrays with mismatched length instead of skipping."
    )
    args = parser.parse_args()
    # Parse epochs now into a list of ints
    args.epochs: List[int] = _parse_epoch_list(args.epoch)
    return args

# --------------- Grid (mirrors the .sh) ---------------
BSZ = "1250"
AUG = "ssaug_strong"
GAMMAS = ["0.5"]
MODELS = ["resnet18", "preactresnet18", "resnet20", "resnet32",
          "resnet44", "resnet56", "resnet110", "vgg11", "vgg13"]
NUM_TRAINS = ["2256", "22560", "None"]

def build_run_dir(base_dir: Path, dataset: str, model: str, num_train: str, gamma: str) -> Path:
    """Return the directory expected to contain per_class_acc.npy (or epoch<E>_per_class_acc.npy) for a run."""
    # Pattern: <BASE_DIR>/<DATASET>_<MODEL>_bsz_<BSZ>_<NUM_TRAIN>_<AUG>_gamma_<GAMMA>_cosine/
    return base_dir / f"{dataset}_{model}_bsz_{BSZ}_{num_train}_{AUG}_gamma_{gamma}_cosine"

def _load_np(path: Path, quiet: bool) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        if not quiet:
            print(f"[WARN] Failed to load {path}: {e}", file=sys.stderr)
        raise
    return arr

def collect_per_class_arrays(
    base_dir: Path,
    dataset: str,
    models: List[str],
    num_trains: List[str],
    gammas: List[str],
    epochs: List[int],
    quiet: bool,
    strict_shape: bool
) -> Tuple[List[np.ndarray], List[List[Path]]]:
    """
    Walk the grid and collect arrays. For each run_dir:
      - If 'epochs' is non-empty: load 'epoch<E>_per_class_acc.npy' for each E in epochs,
        average them (axis=0), and append that mean to the returned arrays.
      - If 'epochs' is empty: fall back to 'per_class_acc.npy' (legacy, single file).
    Returns:
      arrays: list of 1D arrays (one per run after possible epoch-averaging)
      sources: parallel list; each item is the list of source paths used for that run
    """
    arrays: List[np.ndarray] = []
    sources: List[List[Path]] = []
    ref_len: int = -1

    for model in models:
        for nt in num_trains:
            for g in gammas:
                run_dir = build_run_dir(base_dir, dataset, model, nt, g)

                # Collect epoch-specific files
                run_paths: List[Path] = []
                run_arrays: List[np.ndarray] = []

                if epochs:  # preferred: read epoch-prefixed files
                    for e in epochs:
                        p = run_dir / f"epoch{e}_per_class_acc.npy"
                        if not p.is_file():
                            if not quiet:
                                print(f"[WARN] Missing file: {p}", file=sys.stderr)
                            continue
                        try:
                            arr = _load_np(p, quiet=quiet)
                        except Exception:
                            continue

                        if arr.ndim != 1:
                            print(f"[WARN] Expected 1D array in {p}, got shape {arr.shape}. Skipping.", file=sys.stderr)
                            continue

                        if ref_len < 0:
                            ref_len = arr.shape[0]
                        elif arr.shape[0] != ref_len:
                            msg = (
                                f"[ERROR] Mismatched class count in {p}: "
                                f"expected {ref_len}, got {arr.shape[0]}"
                            )
                            if strict_shape:
                                raise ValueError(msg)
                            else:
                                print(f"{msg}. Skipping this file.", file=sys.stderr)
                                continue

                        run_paths.append(p)
                        run_arrays.append(arr.astype(np.float64, copy=False))

                    # If we intended epoch averaging but found nothing, optionally warn and skip this run
                    if len(run_arrays) == 0:
                        if not quiet:
                            print(f"[WARN] No epoch files found for run dir: {run_dir}", file=sys.stderr)
                        continue

                    # Average across epochs for this run
                    run_mean = np.vstack(run_arrays).mean(axis=0)
                    arrays.append(run_mean)
                    sources.append(run_paths)

                    if not quiet:
                        names = ", ".join(p.name for p in run_paths)
                        print(f"[OK] Averaged {len(run_paths)} epoch files for run: {run_dir.name} -> shape {run_mean.shape}")

                else:
                    # Legacy single-file behavior
                    p = run_dir / "per_class_acc.npy"
                    if not p.is_file():
                        if not quiet:
                            print(f"[WARN] Missing file: {p}", file=sys.stderr)
                        continue
                    try:
                        arr = _load_np(p, quiet=quiet)
                    except Exception:
                        continue

                    if arr.ndim != 1:
                        print(f"[WARN] Expected 1D array in {p}, got shape {arr.shape}. Skipping.", file=sys.stderr)
                        continue

                    if ref_len < 0:
                        ref_len = arr.shape[0]
                    elif arr.shape[0] != ref_len:
                        msg = (
                            f"[ERROR] Mismatched class count in {p}: "
                            f"expected {ref_len}, got {arr.shape[0]}"
                        )
                        if strict_shape:
                            raise ValueError(msg)
                        else:
                            print(f"{msg}. Skipping.", file=sys.stderr)
                            continue

                    arrays.append(arr.astype(np.float64, copy=False))
                    sources.append([p])

                    if not quiet:
                        print(f"[OK] Loaded {p} (classes={arr.shape[0]})")

    return arrays, sources

def save_mean_per_class_artifacts(mean_per_class: np.ndarray, save_dir: str):
    """Save mean per-class accuracy as two CSVs and one histogram image.

    Files produced in save_dir:
      - mean_per_class_acc.csv              (columns: class_id,mean_acc_percent)
      - mean_per_class_acc_thresholds.csv   (<= 10% * N for N=1..10: count and indices)
      - mean_per_class_acc_hist.png         (histogram of per-class accuracies in %, NaN ignored)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Normalize to percent scale for reporting/plotting.
    arr = mean_per_class.astype(float)
    if np.nanmax(arr) <= 1.0 + 1e-6:
        arr_pct = arr * 100.0
    else:
        arr_pct = arr

    # CSV 1: per-class mean accuracies (percent)
    csv_path = os.path.join(save_dir, "mean_per_class_acc.csv")
    with open(csv_path, "w") as f:
        f.write("class_id,mean_acc_percent\n")
        for cid, acc in enumerate(arr_pct):
            if np.isnan(acc):
                f.write(f"{cid},nan\n")
            else:
                f.write(f"{cid},{acc:.6f}\n")

    # PNG: histogram of per-class accuracies (percent), NaN ignored
    hist_png_path = os.path.join(save_dir, "mean_per_class_acc_hist.png")
    valid = arr_pct[~np.isnan(arr_pct)]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # high DPI for print clarity
    bins = np.linspace(0, 100, 21)  # 0..100 with 5-point step
    ax.hist(valid, bins=bins)

    # Fonts and styling (match your previous function)
    label_fs = 22   # axis label font size
    tick_fs = 20    # tick label font size
    ax.set_xlabel("Per-class accuracy (%)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(hist_png_path, bbox_inches="tight")
    plt.close(fig)

    # CSV 2: thresholds (<= 10% * N, N=1..10) on percent scale
    thr_csv_path = os.path.join(save_dir, "mean_per_class_acc_thresholds.csv")
    with open(thr_csv_path, "w") as f:
        f.write("threshold_percent,count,class_indices\n")
        mask_valid = ~np.isnan(arr_pct)
        idx_valid = np.where(mask_valid)[0]
        acc_valid = arr_pct[mask_valid]
        for N in range(1, 11):
            thr = 10 * N  # percent
            idx_under = idx_valid[acc_valid <= thr]
            idx_str = ";".join(map(str, idx_under.tolist()))
            f.write(f"{thr},{len(idx_under)},{idx_str}\n")

    print(
        "Saved mean per-class artifacts:\n"
        f"  {csv_path}\n"
        f"  {hist_png_path}\n"
        f"  {thr_csv_path}"
    )

def main() -> None:
    args = parse_args()

    # Determine base directory: first go to parent, then save/SupTrain_<sup>
    base_dir = Path(args.base_dir) if args.base_dir else (Path("..") / f"save/SupTrain_{args.sup_train}")
    if not base_dir.exists():
        print(f"[WARN] Base directory does not exist: {base_dir}", file=sys.stderr)

    # Collect arrays (with optional per-run epoch averaging)
    arrays, sources = collect_per_class_arrays(
        base_dir=base_dir,
        dataset=args.dataset,
        models=MODELS,
        num_trains=NUM_TRAINS,
        gammas=GAMMAS,
        epochs=args.epochs,
        quiet=args.quiet,
        strict_shape=args.strict_shape
    )

    if len(arrays) == 0:
        print("[ERROR] No per-class arrays found across the grid. Nothing to aggregate.", file=sys.stderr)
        sys.exit(2)

    # Stack and compute mean per-class acc across runs (each run already averaged over epochs if requested)
    stacked = np.vstack(arrays)  # shape: [num_runs, num_classes]
    mean_per_class = stacked.mean(axis=0)  # shape: [num_classes]

    # Normalize to [0,1] for threshold comparison, but keep original scale for printing
    norm = mean_per_class.astype(float)
    if np.nanmax(norm) > 1.0 + 1e-6:
        norm = norm / 100.0

    num_classes = mean_per_class.shape[0]
    print("\n================ Aggregation Summary ================\n")
    print(f"SupTrain type: {args.sup_train}")
    print(f"Dataset: {args.dataset}")
    if args.epochs:
        print(f"Epochs averaged per run: {args.epochs}")
    else:
        print("Epochs averaged per run: [legacy single file: per_class_acc.npy]")
    print(f"Runs aggregated: {len(arrays)}")
    print(f"Num classes: {num_classes}")
    print(f"Base dir: {base_dir.resolve()}\n")

    # Show mean per-class acc (index and value)
    print("Mean per-class accuracies (index: value):")
    for idx, val in enumerate(mean_per_class):
        # If input files were already in percent, values will remain in percent here
        print(f"{idx}: {val:.6f}")
    print()

    # Threshold analysis: compare on normalized scale, show threshold in original scale
    print("Classes with mean acc below 10% * N (N=1..10):")
    scale = 100.0 if np.nanmax(mean_per_class) > 1.0 + 1e-6 else 1.0
    for N in range(1, 11):
        thr_norm = 0.1 * N
        inds = np.flatnonzero(norm < thr_norm).tolist()
        thr_orig = thr_norm * scale
        print(f"N={N}  threshold={thr_orig:.2f} -> count={len(inds)}  indices={inds}")

    # Save artifacts (two CSVs + one PNG)
    save_mean_per_class_artifacts(mean_per_class, args.out_dir)

    # Show a brief source list if verbose
    if not args.quiet:
        print("\nSources used (grouped per run; epoch-averaged if multiple paths listed):")
        for group in sources:
            if len(group) == 1:
                print(f" - {group[0]}")
            else:
                print(" - " + ", ".join(str(p) for p in group))

if __name__ == "__main__":
    main()
