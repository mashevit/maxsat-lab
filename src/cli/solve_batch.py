from __future__ import annotations
import argparse, yaml, os
from ..bench.harness import solve_folder, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing .wcnf/.cnf")
    ap.add_argument("--config", required=False, default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=False, default="results.csv")
    ap.add_argument("--time_limit_s", type=float, default=None, help="Wall-time cap per instance (seconds)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # If using nested ls:, map a few common keys
    ls = cfg.get("ls", {})
    if "flip_budget" in ls and "max_flips" not in cfg:
        cfg["max_flips"] = int(ls["flip_budget"])
    if "time_limit_s" in ls and "time_limit_s" not in cfg:
        cfg["time_limit_s"] = float(ls["time_limit_s"])
    if "tabu_length" in ls and "tabu_length" not in cfg:
        cfg["tabu_length"] = int(ls["tabu_length"])
    if "noise" in ls and "noise" not in cfg:
        cfg["noise"] = float(ls["noise"])
    if "hard_repair_budget" in ls and "hard_repair_budget" not in cfg:
        cfg["hard_repair_budget"] = int(ls["hard_repair_budget"])

    if args.time_limit_s is not None:
        cfg["time_limit_s"] = float(args.time_limit_s)

    rows = solve_folder(args.folder, cfg, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_csv(rows, args.out)
    print(f"OK solve_batch: wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
