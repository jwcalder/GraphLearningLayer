#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize subset per-class accuracies across specific epochs for experiment folders.

Behavior:
1) Auto-detect the "save" directory:
   - If --root provided, use it.
   - Else if "./save" exists, use it.
   - Else if "../save" exists, use it.
   - Else fall back to "./save" (may not exist).

2) --class_sub_ind chooses which class indices to average:
   - "auto" (default) depends on dataset:
       * cifar10 -> [0..9]
       * emnist  -> [0,1,9,15,18,21,24,40,41,44]
   - Or pass an explicit CSV list, e.g. "0,1,2".

3) For each experiment directory:
   - Look for files named "epoch{E}_per_class_acc.npy" for each E in --epochs.
   - Each file stores a NumPy array of per-class accuracies (1D C or shape with last dim = C).
   - Select the subset classes, compute the mean for that epoch.
   - Aggregate per-epoch means -> compute overall mean and std across epochs.

Output:
- Prints a CSV-like summary to stdout:
  dataset,model,batch_size,num_train,subset_mean,error_mean,subset_std,n_epochs_used,n_epochs_requested,epochs_used,indices,path_sample
- Optionally saves a CSV via --csv_out.

Example:
python summarize_subset_epochs.py \
  --suptrain_method ERM \
  --datasets cifar10 \
  --models resnet18 \
  --batch_sizes 128 \
  --num_trains 1000 \
  --epochs 300,350,400,450,500 \
  --class_sub_ind auto \
  --csv_out subset_epoch_summary.csv
"""

import argparse
from pathlib import Path
import numpy as np
from typing import Any, List, Tuple

# ----------------------------- CLI & Utilities ----------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Summarize subset per-class accuracies across specified epochs.")
    p.add_argument("--suptrain_method", type=str, default="mlp",
                   help="Value for {suptrain_method} in SupTrain_{suptrain_method}.")
    p.add_argument("--datasets", type=str, default="emnist",
                   help="Comma-separated list of dataset names.")
    p.add_argument("--models", type=str, default="vgg11,vgg13,resnet20,resnet32,resnet44,resnet56,resnet110,resnet18,preactresnet18",
                   help="Comma-separated list of model names.")
    p.add_argument("--batch_sizes", type=str, default="1250",
                   help="Comma-separated list of batch sizes.")
    p.add_argument("--num_trains", type=str, default="2256,22560,None",
                   help="Comma-separated list of num_train.")
    p.add_argument("--root", type=str, default="",
                   help="Root directory containing SupTrain_* folders. If empty, auto-detect './save' or '../save'.")
    p.add_argument("--epochs", type=str, default="300,350,400,450,500",
                   help="Comma-separated list of epochs to aggregate, e.g. '300,350,400,450,500'.")
    p.add_argument("--class_sub_ind", type=str, default="auto",
                   help='Subset of class indices to average. "auto" uses a dataset-based preset; '
                        'or pass a comma-separated list like "0,1,2".')
    p.add_argument("--strict", action="store_true",
                   help="If set, raise error on missing files; otherwise skip with warnings.")
    p.add_argument("--csv_out", type=str, default="script_results/subclass_summary.csv",
                   help="Optional path to save results as CSV. If empty, do not save.")
    return p.parse_args()

def split_list(s: str, cast=str) -> List[Any]:
    """Split a comma-separated string into list, stripping spaces and casting each item."""
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [cast(x) for x in items]

def resolve_root(user_root: str) -> Path:
    """
    Resolve the effective root directory containing SupTrain_* folders.
    Priority:
      1) If user_root provided -> use it.
      2) If ./save exists -> use it.
      3) If ../save exists -> use it.
      4) Otherwise, default to ./save.
    """
    if user_root:
        return Path(user_root).expanduser().resolve()
    cwd = Path.cwd()
    here_save = cwd / "save"
    parent_save = cwd.parent / "save"
    if here_save.exists():
        return here_save.resolve()
    if parent_save.exists():
        return parent_save.resolve()
    return here_save.resolve()

def get_auto_class_indices(dataset: str) -> List[int]:
    """Return default class subset indices for a given dataset."""
    ds = dataset.lower()
    if ds == "cifar10":
        return list(range(10))
    if ds == "emnist":
        return [0, 1, 9, 15, 18, 21, 24, 40, 41, 44]
    return []

def parse_class_indices(arg_val: str, dataset: str) -> List[int]:
    """Parse --class_sub_ind argument."""
    if arg_val.strip().lower() == "auto":
        inds = get_auto_class_indices(dataset)
        return inds  # May be empty for datasets without preset (interpreted as ALL classes).
    return [int(x) for x in split_list(arg_val, int)]

def safe_select_indices(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Extract a 1D vector of per-class accuracies for the requested indices.
    If indices is empty, interpret it as "use all classes".
    Supports shapes:
      - (C,)
      - (T, C) -> uses the last row as per-class accuracies
      - Higher dims -> averages across all but last dim to get shape (C,)
    """
    if arr.ndim == 1:
        base = arr
    elif arr.ndim == 2:
        base = arr[-1, :]
    else:
        base = np.mean(arr, axis=tuple(range(arr.ndim - 1)))
    if len(indices) == 0:
        return np.ravel(base)
    return np.ravel(base)[np.array(indices, dtype=int)]

# ----------------------------- Core Logic ----------------------------- #

def summarize_exp_dir_epochs(
    exp_dir: Path,
    epochs: List[int],
    class_indices: List[int],
    strict: bool
) -> Tuple[float, float, int, int, List[int], str]:
    """
    For a given experiment directory, load epoch-specific per-class files and compute summary.

    Returns:
        subset_mean (float): Mean of per-epoch subset means.
        subset_std  (float): Std of per-epoch subset means (0 if only 1 epoch used).
        n_used      (int)  : Number of epochs successfully loaded.
        n_req       (int)  : Number of requested epochs.
        used_epochs (List[int]): The list of epochs actually used.
        sample_path (str)  : Path of one sample file if available, else "".
    """
    per_epoch_means: List[float] = []
    used_epochs: List[int] = []
    sample_path = ""

    for e in epochs:
        f = exp_dir / f"epoch{e}_per_class_acc.npy"
        if not f.exists():
            msg = f"[WARN] Missing file: {f}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"# {msg}")
            continue
        try:
            arr = np.load(f, allow_pickle=False)
            per_class = safe_select_indices(arr, class_indices)
            epoch_mean = float(np.mean(per_class))
            per_epoch_means.append(epoch_mean)
            used_epochs.append(e)
            if not sample_path:
                sample_path = str(f)
        except Exception as ex:
            if strict:
                raise
            print(f"# [WARN] Failed to process {f}: {ex}")

    n_used = len(per_epoch_means)
    n_req = len(epochs)
    if n_used == 0:
        return float("nan"), float("nan"), 0, n_req, used_epochs, sample_path

    subset_mean = float(np.mean(per_epoch_means))
    subset_std = float(np.std(per_epoch_means, ddof=0)) if n_used > 1 else 0.0
    return subset_mean, subset_std, n_used, n_req, used_epochs, sample_path

def main() -> None:
    """Entry point: summarize subset per-class accuracies across epochs and optionally save CSV."""
    args = parse_args()

    datasets = split_list(args.datasets, str)
    models = split_list(args.models, str)
    batch_sizes = split_list(args.batch_sizes, int)
    num_trains = split_list(args.num_trains, str)
    epochs = split_list(args.epochs, int)

    # Resolve root: user override or auto-detect ./save then ../save
    root = resolve_root(args.root)
    sup_root = root / f"SupTrain_{args.suptrain_method}"

    # Console header
    print("# Subset per-class accuracy across epochs summary")
    print("# Root:", sup_root)
    print("dataset,model,batch_size,num_train,subset_mean,error_mean,subset_std,n_epochs_used,n_epochs_requested,epochs_used,indices,path_sample")

    rows_for_csv = []

    for dataset in datasets:
        # Resolve class indices for this dataset
        class_indices = parse_class_indices(args.class_sub_ind, dataset)
        indices_str = (
            "auto" if args.class_sub_ind.strip().lower() == "auto" and class_indices
            else ("ALL_CLASSES" if len(class_indices) == 0 else args.class_sub_ind)
        )

        for model in models:
            for bsz in batch_sizes:
                for ntr in num_trains:
                    exp_dirname = f"{dataset}_{model}_bsz_{bsz}_{ntr}_ssaug_strong_gamma_0.5_cosine"
                    exp_dir = sup_root / exp_dirname

                    if not exp_dir.exists():
                        msg = f"[WARN] Missing directory: {exp_dir}"
                        if args.strict:
                            raise FileNotFoundError(msg)
                        print(f"# {msg}")
                        continue

                    try:
                        subset_mean, subset_std, n_used, n_req, used_eps, sample_path = summarize_exp_dir_epochs(
                            exp_dir, epochs, class_indices, args.strict
                        )
                        error_mean = 100.0 - subset_mean  # New column: error rate = 100 - subset_mean
                        used_eps_str = ";".join(str(x) for x in used_eps) if used_eps else ""
                        print("{},{},{},{},{:.6f},{:.6f},{:.6f},{},{},{},{},{}".format(
                            dataset, model, bsz, ntr,
                            subset_mean, error_mean, subset_std, n_used, n_req, used_eps_str, indices_str, sample_path
                        ))
                        rows_for_csv.append([
                            dataset, model, bsz, ntr,
                            subset_mean, error_mean, subset_std, n_used, n_req, used_eps_str, indices_str, sample_path
                        ])
                    except Exception as e:
                        if args.strict:
                            raise
                        print(f"# [WARN] Failed to process {exp_dir}: {e}")

    # Optional CSV output
    if args.csv_out:
        try:
            import csv, os
            csv_path = Path(args.csv_out).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            header = [
                "dataset", "model", "batch_size", "num_train",
                "subset_mean", "error_mean", "subset_std", "n_epochs_used", "n_epochs_requested",
                "epochs_used", "indices", "path_sample"
            ]

            # Open file, write header and rows, then flush and fsync for durability
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in rows_for_csv:
                    writer.writerow(list(row))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

            print(f"# Saved CSV to: {csv_path}")
        except Exception as e:
            print(f"# [WARN] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
