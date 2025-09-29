from __future__ import annotations
import argparse, yaml, os
from ..bench.harness import solve_folder, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing .wcnf/.cnf")
    ap.add_argument("--config", required=False, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=False, default="results.csv")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    rows = solve_folder(args.folder, cfg, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_csv(rows, args.out)
    print(f"OK solve_batch: wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
