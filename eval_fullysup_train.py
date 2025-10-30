#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize test_acc_record from multiple experiment folders.

Usage example:
python summarize_test_acc.py \
  --suptrain_method ERM \
  --datasets cifar10,cifar100 \
  --models resnet18,wrn28x10 \
  --batch_sizes 128,256 \
  --num_trains 1000,2000 \
  --root ../save \
  --tail_k 5

The script will look for folders like:
../save/SupTrain_{suptrain_method}/{dataset}_{model}_bsz_{batch_size}_{num_train}_ssaug_strong_gamma_0.5_cosine
and load: loss_acc_records.npy
"""

import argparse
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Sequence, Tuple, List

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Summarize last-K metrics from test_acc_record.")
    p.add_argument("--suptrain_method", type=str, default='mlp',
                   help="Value for {suptrain_method} in SupTrain_{suptrain_method}.")
    p.add_argument("--datasets", type=str, default="cifar10",
                   help="Comma-separated list of dataset names.")
    p.add_argument("--models", type=str, default="vgg11,vgg13,resnet20,resnet32,resnet44,resnet56,resnet110,resnet18,preactresnet18",
                   help="Comma-separated list of model names.")
    p.add_argument("--batch_sizes", type=str, default="1250",
                   help="Comma-separated list of batch sizes (ints).")
    p.add_argument("--num_trains", type=str, default="1000,10000,None",
                   help="Comma-separated list of num_train (ints).")
    p.add_argument("--root", type=str, default="save",
                   help="Root directory that contains SupTrain_* folders. Default: ../save")
    p.add_argument("--tail_k", type=int, default=20,
                   help="How many last values to average/std from test_acc_record. ")
    p.add_argument("--strict", action="store_true",
                   help="If set, raise error on missing files/keys; otherwise skip with warnings.")
    p.add_argument("--csv_out", type=str, default="fullysup_eval_summary.csv",
                   help="Optional path to save results as CSV. If empty, do not save.")
    return p.parse_args()

def split_list(s: str, cast=str) -> List[Any]:
    """Split a comma-separated string into list, stripping spaces and casting each item."""
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [cast(x) for x in items]

def load_npy_dict(npy_path: Path) -> Dict[str, Any]:
    """Load a numpy .npy file that stores a Python dict."""
    arr = np.load(npy_path, allow_pickle=True)
    # Handle common saving patterns:
    # - np.save(path, dict) -> arr is a numpy array with dtype=object scalar; use .item()
    # - np.save(path, dict) in some setups can directly return a dict after loading
    if isinstance(arr, dict):
        return arr
    # Scalar object array containing a dict
    if isinstance(arr, np.ndarray):
        if arr.shape == () and hasattr(arr, "item"):
            maybe = arr.item()
            if isinstance(maybe, dict):
                return maybe
    # If the content is unexpected, raise an informative error
    raise TypeError(f"Unsupported .npy structure in {npy_path}. Expected a dict or 0-dim object array holding a dict.")

def tail_mean_std(values: Sequence[float], k: int) -> Tuple[float, float, int]:
    """Compute mean and std over the last k values of a sequence."""
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    if n == 0:
        raise ValueError("Empty test_acc_record.")
    k = min(k, n)
    tail = v[-k:]
    return float(np.mean(tail)), float(np.std(tail, ddof=0)), int(k)

def main() -> None:
    # Parse args
    args = parse_args()

    datasets = split_list(args.datasets, str)
    models = split_list(args.models, str)
    batch_sizes = split_list(args.batch_sizes, int)
    num_trains = split_list(args.num_trains, str)

    root = Path(args.root).expanduser().resolve()
    sup_root = root / f"SupTrain_{args.suptrain_method}"

    # Console header with the new mean_err column
    print("# Summary over last {} values from 'test_acc_record'".format(args.tail_k))
    print("# Root:", sup_root)
    print("dataset,model,batch_size,num_train,mean,mean_err,std,used_k,record_len,path")

    rows_for_csv = []

    for dataset in datasets:
        for model in models:
            for bsz in batch_sizes:
                for ntr in num_trains:
                    # Build experiment directory
                    exp_dirname = f"{dataset}_{model}_bsz_{bsz}_{ntr}_ssaug_strong_gamma_0.5_cosine"
                    exp_dir = sup_root / exp_dirname
                    npy_path = exp_dir / "loss_acc_records.npy"

                    if not npy_path.exists():
                        msg = f"[WARN] Missing file: {npy_path}"
                        if args.strict:
                            raise FileNotFoundError(msg)
                        else:
                            print(f"# {msg}")
                            continue

                    try:
                        d = load_npy_dict(npy_path)
                        if "test_acc_record" not in d:
                            msg = f"[WARN] Key 'test_acc_record' not found in: {npy_path}"
                            if args.strict:
                                raise KeyError(msg)
                            else:
                                print(f"# {msg}")
                                continue

                        record = d["test_acc_record"]
                        mean, std, used_k = tail_mean_std(record, args.tail_k)
                        mean_err = 100.0 - mean  # error rate = 100 - accuracy
                        rec_len = len(record)

                        # Print with mean_err
                        print("{},{},{},{},{:.6f},{:.6f},{:.6f},{},{},{}".format(
                            dataset, model, bsz, ntr, mean, mean_err, std, used_k, rec_len, npy_path
                        ))

                        # Collect for CSV
                        rows_for_csv.append([
                            dataset, model, bsz, ntr, mean, mean_err, std, used_k, rec_len, str(npy_path)
                        ])

                    except Exception as e:
                        if args.strict:
                            raise
                        print(f"# [WARN] Failed to process {npy_path}: {e}")

    # Optional CSV output (now includes mean_err)
    if args.csv_out:
        try:
            import csv
            csv_path = Path(args.csv_out).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "dataset", "model", "batch_size", "num_train",
                    "mean", "mean_err", "std", "used_k", "record_len", "path"
                ])
                writer.writerows(rows_for_csv)
            print(f"# Saved CSV to: {csv_path}")
        except Exception as e:
            print(f"# [WARN] Failed to write CSV: {e}")

if __name__ == "__main__":
    main()
