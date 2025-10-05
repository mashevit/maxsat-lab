from __future__ import annotations
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict
from ..sat.cnf import WCNF
from ..sat.walksat import WalkSAT, WalkSATConfig

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Run a minimal WalkSAT step (wiring check).")
    ap.add_argument("--cnf", required=True, help="Path to a .cnf or .wcnf instance")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inst = WCNF.parse_dimacs(args.cnf)

    ls_cfg = cfg.get("ls", {})
    ws_cfg = WalkSATConfig(
        max_flips=int(ls_cfg.get("flip_budget", 2000)),
        time_limit_s=float(ls_cfg.get("time_limit_s", 60.0)),
        restarts=int(ls_cfg.get("restarts", 3)),
        patience=int(ls_cfg.get("patience", 150)),
        noise=float(ls_cfg.get("noise", 0.3)),
        tabu_length=int(ls_cfg.get("tabu_length", 12)),
        forbid_break_hard=(ls_cfg.get("hard_clause_policy", "forbid") == "forbid")
        
    )

    solver = WalkSAT(inst, ws_cfg)
    best_assign, best_score = solver.run()
    print(f"[OK] best_satisfied_weight={best_score} on n={inst.n_vars} vars, m={len(inst.clauses)} clauses.")

if __name__ == "__main__":
    main()
