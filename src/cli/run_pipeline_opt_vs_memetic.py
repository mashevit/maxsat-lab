# src/cli/run_pipeline_opt_vs_memetic.py
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any, Sequence


def run_cmd(cmd: Sequence[str]) -> None:
    """Run a command, echoing it to stdout, and fail hard on non-zero exit."""
    print("[pipeline] running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"[pipeline] ERROR: command failed with exit code {result.returncode}: "
            + " ".join(cmd)
        )


def ensure_rc2_csv(bench_dir: Path, rc2_csv: Path) -> None:
    """If rc2_csv doesn't exist, run batch_opt_rc2 to create it."""
    if rc2_csv.exists():
        print(f"[pipeline] RC2 CSV already exists: {rc2_csv}")
        return

    rc2_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.cli.batch_opt_rc2",
        str(bench_dir),
        "-o",
        str(rc2_csv),
    ]
    run_cmd(cmd)


def ensure_memetic_csv(
    bench_dir: Path,
    config: Path,
    seeds: Sequence[int],
    memetic_csv: Path,
    config_id: str | None = None,
) -> None:
    """If memetic_csv doesn't exist, run run_experiment1 to create it."""
    if memetic_csv.exists():
        print(f"[pipeline] memetic CSV already exists: {memetic_csv}")
        return

    memetic_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.cli.run_experiment1",
        "--bench_dir",
        str(bench_dir),
        "-c",
        str(config),
        "--out_csv",
        str(memetic_csv),
    ]

    # seeds
    if seeds:
        cmd += ["--seeds", *[str(s) for s in seeds]]

    # config_id (if not given, use the config path string)
    cfg_id = config_id or str(config)
    cmd += ["--config_id", cfg_id]

    run_cmd(cmd)


def load_rc2_by_instance(rc2_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read RC2 CSV and index rows by basename(instance).

    Expected columns (from your example):
      instance,status,time_s,n_vars,n_hard,n_soft,soft_total_weight,opt_cost,best_soft_weight
    """
    rc2_by_inst: Dict[str, Dict[str, Any]] = {}

    with rc2_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "instance",
            "status",
            "soft_total_weight",
            "opt_cost",
            "best_soft_weight",
        }
        if reader.fieldnames is None:
            raise ValueError(f"[pipeline] RC2 CSV {rc2_csv} has no header")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"[pipeline] RC2 CSV {rc2_csv} missing required columns: {missing}"
            )

        for row in reader:
            inst_basename = os.path.basename(row["instance"])
            # Cast numeric fields:
            row["soft_total_weight"] = float(row["soft_total_weight"])
            row["opt_cost"] = float(row["opt_cost"])
            row["best_soft_weight"] = float(row["best_soft_weight"])
            rc2_by_inst[inst_basename] = row

    return rc2_by_inst


def make_comparison_csv(rc2_csv: Path, memetic_csv: Path, out_csv: Path) -> None:
    """
    Join RC2 optima with memetic results.

    RC2:
      - soft_total_weight  (sum of weights of soft clauses)
      - opt_cost           (min weight of violated softs)
      - best_soft_weight   (optimal satisfied weight)

    Memetic:
      - best_weight        (satisfied weight)
      - soft_unsat         (# of unsatisfied softs, not strictly needed here)
      - status             (OK/TIMEOUT/...)

    We compute:
      memetic_cost = soft_total_weight - best_weight
      opt_gap      = memetic_cost - opt_cost
      rel_gap      = opt_gap / opt_cost (or 0 if opt_cost == 0)
      is_optimal   = (rc2_status == "OPT" and opt_gap == 0)
    """
    print(f"[pipeline] building comparison CSV: {out_csv}")

    rc2_by_inst = load_rc2_by_instance(rc2_csv)

    with memetic_csv.open(newline="") as f_in, out_csv.open("w", newline="") as f_out:
        mem_reader = csv.DictReader(f_in)

        mem_required = {
            "instance",
            "seed",
            "best_weight",
            "status",
            "config_id",
            "soft_unsat",
        }
        if mem_reader.fieldnames is None:
            raise ValueError(f"[pipeline] memetic CSV {memetic_csv} has no header")

        missing = mem_required - set(mem_reader.fieldnames)
        if missing:
            raise ValueError(
                f"[pipeline] memetic CSV {memetic_csv} missing required columns: {missing}"
            )

        fieldnames = [
            "instance",
            "seed",
            "config_id",
            "rc2_status",
            "rc2_opt_cost",
            "soft_total_weight",
            "rc2_best_soft_weight",
            "memetic_status",
            "memetic_best_weight",
            "memetic_soft_unsat",
            "memetic_cost",
            "opt_gap",
            "rel_gap",
            "is_optimal",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for mem_row in mem_reader:
            inst_basename = os.path.basename(mem_row["instance"])
            rc2_row = rc2_by_inst.get(inst_basename)

            if rc2_row is None:
                print(
                    f"[pipeline] WARNING: instance {inst_basename} not found in RC2 CSV, skipping",
                    file=sys.stderr,
                )
                continue

            soft_total = float(rc2_row["soft_total_weight"])
            opt_cost = float(rc2_row["opt_cost"])
            rc2_best_soft_weight = float(rc2_row["best_soft_weight"])

            mem_best_weight = float(mem_row["best_weight"])
            mem_soft_unsat = mem_row.get("soft_unsat", "")
            mem_status = mem_row["status"]

            mem_cost = soft_total - mem_best_weight
            opt_gap = mem_cost - opt_cost
            rel_gap = 0.0 if opt_cost == 0 else opt_gap / opt_cost
            is_optimal = (rc2_row["status"] == "OPT") and (opt_gap == 0)

            out_row = {
                "instance": inst_basename,
                "seed": mem_row["seed"],
                "config_id": mem_row.get("config_id", ""),
                "rc2_status": rc2_row["status"],
                "rc2_opt_cost": opt_cost,
                "soft_total_weight": soft_total,
                "rc2_best_soft_weight": rc2_best_soft_weight,
                "memetic_status": mem_status,
                "memetic_best_weight": mem_best_weight,
                "memetic_soft_unsat": mem_soft_unsat,
                "memetic_cost": mem_cost,
                "opt_gap": opt_gap,
                "rel_gap": rel_gap,
                "is_optimal": int(is_optimal),  # 1 or 0
            }

            writer.writerow(out_row)

    print(f"[pipeline] wrote comparison CSV: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure RC2 and memetic CSVs exist for a benchmark, "
            "then create a comparison CSV with optimality gaps."
        )
    )
    parser.add_argument(
        "--bench_dir",
        required=True,
        type=Path,
        help="Directory with CNF/WCNF instances (e.g. data/exp0)",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config for memetic experiment (e.g. configs/cfg2.yaml)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of seeds to pass to run_experiment1 (e.g. --seeds 1 2 3)",
    )
    parser.add_argument(
        "--rc2_csv",
        required=True,
        type=Path,
        help="Path to RC2 output CSV (will be created if missing)",
    )
    parser.add_argument(
        "--memetic_csv",
        required=True,
        type=Path,
        help="Path to memetic experiment CSV (will be created if missing)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        type=Path,
        help="Path for comparison CSV (RC2 vs memetic)",
    )
    parser.add_argument(
        "--config_id",
        type=str,
        default=None,
        help="Optional config_id to pass to run_experiment1 "
             "(default: use the config path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bench_dir: Path = args.bench_dir
    config: Path = args.config
    seeds = args.seeds
    rc2_csv: Path = args.rc2_csv
    memetic_csv: Path = args.memetic_csv
    out_csv: Path = args.out_csv
    config_id: str | None = args.config_id

    print("[pipeline] bench_dir   :", bench_dir)
    print("[pipeline] config      :", config)
    print("[pipeline] seeds       :", seeds)
    print("[pipeline] rc2_csv     :", rc2_csv)
    print("[pipeline] memetic_csv :", memetic_csv)
    print("[pipeline] out_csv     :", out_csv)

    ensure_rc2_csv(bench_dir, rc2_csv)
    ensure_memetic_csv(bench_dir, config, seeds, memetic_csv, config_id=config_id)
    make_comparison_csv(rc2_csv, memetic_csv, out_csv)


if __name__ == "__main__":
    main()
