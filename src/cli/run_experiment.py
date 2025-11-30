# src/cli/run_experiment.py
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- Flexible imports so you can run as a module or a script ---
try:
    from evo.memetic import run_memetic
except Exception:
    # Add parent-of-src to path when invoked directly
    this = os.path.abspath(__file__)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(this), "..")))
    from evo.memetic import run_memetic  # type: ignore

# Optional YAML support (like in run_ea)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -------- Minimal WCNF fallback loader (same style as run_ea) --------
@dataclass
class Clause:
    lits: List[int]
    weight: float
    is_hard: bool


@dataclass
class WCNF:
    n_vars: int
    clauses: List[Clause]


def _parse_wcnf(path: str) -> WCNF:
    """
    Simple WCNF parser: expects 'p wcnf n m top' and then 'w lit ... 0' lines.
    A clause is 'hard' if weight >= top. Comments ('c ...') ignored.
    """
    top: Optional[float] = None
    n_vars = 0
    clauses: List[Clause] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("c"):
                continue
            if s.startswith("p"):
                # e.g., "p wcnf 3 5 10" or "p wcnf 3 5"
                parts = s.split()
                assert parts[1].lower() == "wcnf", "Only WCNF is supported"
                n_vars = int(parts[2])
                if len(parts) >= 5:
                    try:
                        top = float(parts[4])
                    except Exception:
                        top = None
                continue
            # clause line: weight then literals terminated by 0
            toks = s.split()
            if not toks:
                continue
            w = float(toks[0])
            lits = [int(x) for x in toks[1:] if x != "0"]
            is_hard = (top is not None and w >= top)
            clauses.append(Clause(lits=lits, weight=w, is_hard=is_hard))

    if n_vars == 0:
        # derive from max literal if header missing
        maxv = 0
        for cl in clauses:
            for lit in cl.lits:
                if abs(lit) > maxv:
                    maxv = abs(lit)
        n_vars = maxv
    return WCNF(n_vars=n_vars, clauses=clauses)


def _load_wcnf_generic(path: str) -> Any:
    """
    Try to use the full DIMACS/WCNF parser from sat.cnf (like run_ea),
    fall back to the simple WCNF parser above.
    Supports both .cnf and .wcnf files.
    """
    # Try the project parser first
    try:
        from sat.cnf import WCNF as DimacsWCNF  # type: ignore
    except Exception:
        DimacsWCNF = None  # type: ignore

    if DimacsWCNF is not None:
        try:
            return DimacsWCNF.parse_dimacs(path)
        except Exception as e:
            print(f"[run_experiment] Warning: sat.cnf parser failed on {path}: {e}", file=sys.stderr)

    # Fallback: simple internal WCNF parser (only for real WCNF files)
    return _parse_wcnf(path)


# -------- Config helpers (same spirit as run_ea) --------
def _deep_set(d: Dict[str, Any], dotted_key: str, val: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = val


def _parse_kv_override(s: str) -> tuple[str, Any]:
    """
    Parse "a.b.c=123" -> ("a.b.c", 123) with JSON-ish literal parsing.
    """
    if "=" not in s:
        raise ValueError(f"Expected KEY=VALUE, got: {s}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    try:
        parsed = json.loads(v)
    except Exception:
        parsed = v
    return k, parsed


def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception as e:
        raise SystemExit(f"Failed to parse config {path}: {e}")


# -------- Instance discovery --------
def _iter_instances(bench_dir: str, recursive: bool = False) -> List[str]:
    bench_dir = os.path.abspath(bench_dir)

    # Allow a single file as "bench_dir"
    if os.path.isfile(bench_dir):
        return [bench_dir]

    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(bench_dir):
            for name in files:
                if name.lower().endswith((".wcnf", ".cnf")):
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(bench_dir):
            if name.lower().endswith((".wcnf", ".cnf")):
                paths.append(os.path.join(bench_dir, name))
    paths.sort()
    return paths


# -------- Core experiment logic --------
def run_experiment(
    bench_dir: str,
    cfg: Dict[str, Any],
    seeds: List[int],
    out_csv: str,
    config_id: str,
    time_limit: Optional[float] = None,
    recursive: bool = False,
) -> None:
    # Ensure EA is enabled by default (like run_ea)
    if "ea" not in cfg:
        cfg["ea"] = {}
    cfg["ea"].setdefault("enabled", True)

    # Optional global time limit override
    # NOTE: if your solver expects the limit in a different key
    # (e.g. "ea.time_limit_s" or "ls.time_limit_s"), change this line.
    if time_limit is not None:
        _deep_set(cfg, "time_limit_s", float(time_limit))

    instances = _iter_instances(bench_dir, recursive=recursive)
    if not instances:
        raise SystemExit(f"No .cnf/.wcnf files found under: {bench_dir}")

    # Precompute static info per instance
    instance_info: Dict[str, Dict[str, Any]] = {}
    for path in instances:
        wcnf = _load_wcnf_generic(path)
        # Assume same interface as in run_ea / evo.memetic
        n_vars = int(getattr(wcnf, "n_vars", 0))
        clauses = getattr(wcnf, "clauses", [])

        total_soft_weight = 0.0
        for cl in clauses:
            is_hard = bool(getattr(cl, "is_hard", False))
            if not is_hard:
                total_soft_weight += float(getattr(cl, "weight", 1.0))

        instance_info[path] = {
            "wcnf": wcnf,
            "n_vars": n_vars,
            "n_clauses": len(clauses),
            "total_soft_weight": total_soft_weight,
        }

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    fieldnames = [
        "instance",
        "seed",
        "n_vars",
        "n_clauses",
        "hard_violations",
        "soft_unsat",
        "best_weight",
        "runtime_s",
        "flips",
        "ls_calls",
        "status",
        "config_id",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for inst_path in instances:
            info = instance_info[inst_path]
            inst_name = os.path.basename(inst_path)

            for seed in seeds:
                run_cfg = copy.deepcopy(cfg)
                wcnf = info["wcnf"]
                total_soft_weight = float(info["total_soft_weight"])

                t0 = time.time()
                res = run_memetic(wcnf, run_cfg, rng_seed=seed)
                elapsed = float(res.get("elapsed_sec", time.time() - t0))

                # Basic stats from solver output
                hard_viol = int(res.get("hard_violations", 0))

                # best_weight: prefer "best_total_weight" if present,
                # else fall back to "best_soft_weight".
                best_soft_weight = float(res.get("best_soft_weight", 0.0))
                best_weight = float(res.get("best_total_weight", best_soft_weight))

                # soft_unsat: if solver reports it, use that; otherwise derive.
                if "soft_unsat_weight" in res:
                    soft_unsat = float(res["soft_unsat_weight"])
                elif "unsat_soft_weight" in res:
                    soft_unsat = float(res["unsat_soft_weight"])
                else:
                    # Assume best_soft_weight is satisfied soft weight
                    soft_unsat = max(total_soft_weight - best_soft_weight, 0.0)

                flips = int(res.get("total_flips", 0))
                meta = res.get("meta", {}) or {}
                ls_calls = int(meta.get("ls_calls", 0))
                status = str(res.get("status", "OK"))

                row = {
                    "instance": inst_name,
                    "seed": seed,
                    "n_vars": info["n_vars"],
                    "n_clauses": info["n_clauses"],
                    "hard_violations": hard_viol,
                    "soft_unsat": soft_unsat,
                    "best_weight": best_weight,
                    "runtime_s": elapsed,
                    "flips": flips,
                    "ls_calls": ls_calls,
                    "status": status,
                    "config_id": config_id,
                }
                writer.writerow(row)


# -------- CLI entrypoint --------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="src.cli.run_experiment",
        description="Run memetic EA over a directory of CNF/WCNF instances and write a CSV summary.",
    )
    p.add_argument(
        "--bench_dir",
        required=True,
        help="Directory containing .cnf/.wcnf files OR a single file path",
    )
    p.add_argument(
        "-c",
        "--cfg",
        "--config",
        dest="cfg",
        help="YAML/JSON config file (same format as for run_ea.py)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of RNG seeds, e.g. --seeds 1 2 3 4 5",
    )
    p.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Optional per-run time limit in seconds "
             "(forwarded into config as time_limit_s; tweak in code if you use a different key)",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Path to CSV file to write results into",
    )
    p.add_argument(
        "--config_id",
        help="Identifier to store in the config_id column "
             "(default: stem of config filename or 'default')",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of bench_dir when searching for instances",
    )
    p.add_argument(
        "--override",
        "-D",
        action="append",
        default=[],
        help='Override config key: -D ea.pop_size=80 -D ls.time_limit_s=0.05',
    )

    args = p.parse_args(argv)

    cfg: Dict[str, Any] = _load_cfg(args.cfg)

    # Apply -D overrides (same behavior as run_ea)
    for ov in args.override:
        k, v = _parse_kv_override(ov)
        _deep_set(cfg, k, v)

    # Derive config_id if not provided
    if args.config_id:
        config_id = args.config_id
    elif args.cfg:
        config_id = os.path.splitext(os.path.basename(args.cfg))[0]
    else:
        config_id = "default"

    run_experiment(
        bench_dir=args.bench_dir,
        cfg=cfg,
        seeds=args.seeds,
        out_csv=args.out_csv,
        config_id=config_id,
        time_limit=args.time_limit,
        recursive=args.recursive,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
