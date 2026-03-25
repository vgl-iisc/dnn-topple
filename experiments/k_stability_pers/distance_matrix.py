"""Build a distance matrix across varying k for a given dataset and epoch."""

import argparse
import os
import glob
import numpy as np
import persim


BASE_DIR = os.path.join(os.path.dirname(__file__), "strees_varying_k")
K_VALUES = [15, 20, 25, 30, 35, 40, 45, 60, 80]


def parse_epoch_values(epoch_arg):
    """Parse epoch input as either a single integer or start,stop,step."""
    if "," not in epoch_arg:
        return [int(epoch_arg)]

    parts = [p.strip() for p in epoch_arg.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid epoch range '{epoch_arg}'. Use start,stop,step (e.g. 0,200,10)."
        )

    start, stop, step = map(int, parts)
    if step <= 0:
        raise ValueError("Epoch range step must be > 0.")
    if stop <= start:
        raise ValueError("Epoch range stop must be greater than start.")

    return list(range(start, stop, step))


def load_diagram(path):
    """Load a persistence diagram from a CSV/text file (two columns: birth, death)."""
    dgm = np.loadtxt(path, delimiter=",")
    if dgm.ndim == 1:
        dgm = dgm.reshape(1, 2)
    return dgm


def find_diagram(base_dir, dataset, epoch, k):
    """Resolve the path to a persistence diagram file."""
    folder = os.path.join(base_dir, f"strees_cross_epoch_{k}", dataset, "trainUval")
    pattern = os.path.join(folder, f"*_e{epoch}_{k}-persistence.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No persistence diagram found for dataset={dataset}, epoch={epoch}, k={k}\n  pattern: {pattern}")
    return matches[0]


def compute_distance_matrix(base_dir, dataset, epoch, metric, k_values):
    """Compute pairwise distance matrix for the given k values."""
    # Load all diagrams
    diagrams = {}
    for k in k_values:
        path = find_diagram(base_dir, dataset, epoch, k)
        diagrams[k] = load_diagram(path)
        print(f"  Loaded k={k}: {path} ({len(diagrams[k])} points)")

    n = len(k_values)
    dist_matrix = np.zeros((n, n))

    dist_fn = persim.bottleneck if metric == "bottleneck" else persim.wasserstein

    total = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            count += 1
            d = dist_fn(diagrams[k_values[i]], diagrams[k_values[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            print(f"  [{count}/{total}] k={k_values[i]} vs k={k_values[j]}: {d:.6f}")

    return dist_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute a distance matrix across varying k for a given dataset and epoch."
    )
    parser.add_argument("dataset", choices=["densenet_cifar", "resnet_cifar"],
                        help="Dataset to use")
    parser.add_argument("epoch", type=str,
                        help="Epoch number (e.g. 50) or range start,stop,step (e.g. 0,200,10)")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR,
                        help=f"Root folder containing strees_cross_epoch_* dirs (default: {BASE_DIR})")
    parser.add_argument("--metric", choices=["bottleneck", "wasserstein"],
                        default="bottleneck",
                        help="Distance metric (default: bottleneck)")
    parser.add_argument("-k", "--k-values", type=int, nargs="+",
                        default=K_VALUES,
                        help=f"k values to compare (default: {K_VALUES})")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: printed to stdout). For ranges, use {epoch} or _e<epoch> is appended.")
    args = parser.parse_args()

    epochs = parse_epoch_values(args.epoch)

    print(f"Base dir: {args.base_dir}")
    print(f"k values: {args.k_values}")
    print(f"Epochs: {epochs}")

    for epoch in epochs:
        print(f"\nComputing {args.metric} distance matrix for {args.dataset}, epoch {epoch}")

        dist_matrix = compute_distance_matrix(
            args.base_dir, args.dataset, epoch, args.metric, args.k_values
        )

        # Pretty-print
        header = "\t".join(f"k={k}" for k in args.k_values)
        print(f"\n\t{header}")
        for i, k in enumerate(args.k_values):
            row = "\t".join(f"{dist_matrix[i, j]:.6f}" for j in range(len(args.k_values)))
            print(f"k={k}\t{row}")

        # Save if requested
        if args.output:
            if "{epoch}" in args.output:
                out_path = args.output.format(epoch=epoch)
            elif len(epochs) > 1:
                root, ext = os.path.splitext(args.output)
                out_path = f"{root}_e{epoch}{ext or '.csv'}"
            else:
                out_path = args.output

            np.savetxt(out_path, dist_matrix, delimiter=",",
                       header=",".join(str(k) for k in args.k_values), comments="")
            print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
