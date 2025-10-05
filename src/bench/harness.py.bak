from __future__ import annotations
import os, csv, time, hashlib, json
from typing import Dict, Any, List

from ..sat import cnf as cnf_mod
from ..sat import walksat


def _config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def solve_folder(folder: str, cfg: Dict[str, Any], seed: int = 1, time_limit_s: float = 60.0) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(os.listdir(folder)):
        if not (name.endswith(".wcnf") or name.endswith(".cnf")):
            continue
        path = os.path.join(folder, name)
        # Your cnf module exposes WCNF with a static parser:
        ins = cnf_mod.WCNF.parse_dimacs(path)
        local_cfg = dict(cfg)
        local_cfg["time_limit_s"] = local_cfg.get("time_limit_s", time_limit_s)
        t0 = time.time()
        stats = walksat.run_satlike(ins, local_cfg, rng_seed=seed)
        stats["instance"] = name
        stats["seed"] = seed
        stats["config_hash"] = _config_hash(local_cfg)
        stats["wall_sec"] = time.time() - t0
        rows.append(stats)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    if not rows:
        return
    keys = ["instance", "seed", "elapsed_sec", "best_soft_weight", "hard_violations",
            "total_flips", "flips_per_sec", "restarts", "final_noise", "config_hash", "wall_sec"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
