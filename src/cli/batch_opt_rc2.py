# src/cli/batch_opt_rc2.py
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional

# We reuse the logic and dataclass from run_opt_rc2.py
from .run_opt_rc2 import solve_with_rc2, RC2Result, solve_instance


def collect_instances(root: str, exts: List[str]) -> List[str]:
    """
    Recursively collect all files under `root` whose extension
    is in `exts` (e.g., [".cnf", ".wcnf"]).
    """
    root = os.path.abspath(root)
    instances: List[str] = []

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() in exts:
                instances.append(os.path.join(dirpath, name))

    instances.sort()
    return instances


def write_csv(results: List[RC2Result], out_path: str) -> None:
    """
    Write a CSV summary from a list of RC2Result objects.

    Columns are chosen to be useful for MaxSAT experiments.
    """
    # --- ensure parent directory exists ---
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "instance",
        "status",
        "time_s",
        "n_vars",
        "n_hard",
        "n_soft",
        "soft_total_weight",
        "opt_cost",
        "best_soft_weight",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "instance": r.instance,
                    "status": r.status,
                    "time_s": f"{r.time_s:.6f}",
                    "n_vars": r.n_vars,
                    "n_hard": r.n_hard,
                    "n_soft": r.n_soft,
                    "soft_total_weight": r.soft_total_weight,
                    "opt_cost": "" if r.opt_cost is None else r.opt_cost,
                    "best_soft_weight": (
                        "" if r.best_soft_weight is None else r.best_soft_weight
                    ),
                }
            )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RC2 optimally on all CNF/WCNF instances under a directory "
            "and write a CSV summary."
        )
    )

    parser.add_argument(
        "root",
        help="Root directory that contains CNF/WCNF instances (e.g., data/dev_small).",
    )

    parser.add_argument(
        "-o",
        "--out-csv",
        default="rc2_batch_results.csv",
        help="Path to output CSV file (default: %(default)s).",
    )

    parser.add_argument(
        "--solver",
        default="g3",
        help="Underlying SAT solver name for RC2 (default: %(default)s).",
    )

    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".cnf", ".wcnf"],
        help=(
            "File extensions to include (space-separated). "
            "Default: .cnf .wcnf"
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    exts = [e if e.startswith(".") else "." + e for e in args.ext]
    instances = collect_instances(args.root, exts)

    if not instances:
        print(f"[batch_opt_rc2] No instances found under {args.root} with extensions {exts}")
        return

    print(f"[batch_opt_rc2] Found {len(instances)} instances. Solving with RC2...")

    results: List[RC2Result] = []
    for i, path in enumerate(instances, start=1):
        print(f"[{i}/{len(instances)}] {path}")
        #res = solve_with_rc2(path, solver_name=args.solver)
        res = solve_instance(path, solver_name=args.solver)
        results.append(res)

    write_csv(results, args.out_csv)
    print(f"[batch_opt_rc2] Done. Results written to {args.out_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
