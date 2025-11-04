from __future__ import annotations
import os, csv, time, hashlib, json
from typing import Dict, Any, List

from ..sat import cnf as cnf_mod
from ..sat import walksat





def normalize_cfg(cfg: Dict[str, Any], *, n_vars: int | None = None) -> Dict[str, Any]:
    """
    Flattens cfg['ls'] into top-level keys used by the solver.
    If n_vars is provided, computes restart_k from size_rule.
    """
    out = dict(cfg)                    # start with a shallow copy
    ls  = cfg.get("ls", {})

    # --- Direct one-to-one mappings ---
    if "flip_budget" in ls:     out.setdefault("flip_budget", int(ls["flip_budget"]))
    if "tabu_length" in ls:     out.setdefault("tabu_length", int(ls["tabu_length"]))
    if "noise" in ls:           out.setdefault("noise", float(ls["noise"]))
    if "time_limit_s" in ls:    out.setdefault("time_limit_s", float(ls["time_limit_s"]))
    if "hard_repair_budget" in ls:
        out.setdefault("hard_repair_budget", int(ls["hard_repair_budget"]))

    # --- Noise adaptation block ---
    na = ls.get("noise_adapt", {})
    if "min" in na:             out.setdefault("noise_min", float(na["min"]))
    if "max" in na:             out.setdefault("noise_max", float(na["max"]))

    # You have a single 'adapt_delta' in code; YAML has improve & stagnate.
    # Keep both (preferred), and fall back to a single adapt_delta for old code.
    if "improve_delta" in na:   out.setdefault("noise_adapt_improve_delta", float(na["improve_delta"]))
    if "stagnate_delta" in na:  out.setdefault("noise_adapt_stagnate_delta", float(na["stagnate_delta"]))
    if "stagnate_delta" in na and "adapt_delta" not in out:
        # back-compat: use stagnate step as generic adapt_delta
        out["adapt_delta"] = float(na["stagnate_delta"])

    # Some configs use "patience" for how often to adapt; map to adapt_every if not set
    if "patience" in ls:        out.setdefault("adapt_every", int(ls["patience"]))

    # --- Dynamic weights block ---
    dw = ls.get("dynamic_weights", {})
    if "bump" in dw:            out.setdefault("dyn_bump", float(dw["bump"]))
    if "smooth_every" in dw:    out.setdefault("smooth_every", int(dw["smooth_every"]))
    if "rho" in dw:             out.setdefault("rho", float(dw["rho"]))

    # --- Size rule / restart_k ---
    sr = ls.get("size_rule", {})
    k_min = int(sr.get("k_child_min", 500))
    k_max = int(sr.get("k_child_max", 15000))
    k_per = int(sr.get("k_per_var", 5))
    if n_vars is not None:
        restart_k = max(k_min, min(k_max, k_per * n_vars))
        out.setdefault("restart_k", int(restart_k))

    # --- Hard clause policy -> hard_safe flag ---
    # "forbid" = do not allow flips that increase broken hard clauses -> hard_safe=True
    # anything else (e.g., "allow") -> hard_safe=False
    hcp = ls.get("hard_clause_policy")
    if hcp is not None:
        out.setdefault("hard_safe", str(hcp).lower() == "forbid")

    # --- Random seed (optional) ---
    if "random_seed" in cfg:
        out.setdefault("random_seed", int(cfg["random_seed"]))

    return out



def _config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def solve_folder(folder: str, cfg: Dict[str, Any], seed: int = 1, time_limit_s: float = 60.0) -> List[Dict[str, Any]]:
    rows = []
    for name in sorted(os.listdir(folder)):
        if not (name.endswith(".wcnf") or name.endswith(".cnf")):
            continue
        path = os.path.join(folder, name)
        # Load via your WCNF parser
        ins = cnf_mod.WCNF.parse_dimacs(path)

        n = ins.n_vars  # or however you expose #vars
        print(f'n = {n}')
        # Per-instance cfg with derived values (e.g., restart_k)
        local_cfg = normalize_cfg(cfg, n_vars=n)

        # Respect time limit if not already set
        local_cfg["time_limit_s"] = float(local_cfg.get("time_limit_s", time_limit_s))

        # Debug: print effective hard-safe policy
        print(f"hard_safe={bool(local_cfg.get('hard_safe', True))} for {name}")

        #local_cfg = dict(cfg)
        local_cfg["time_limit_s"] = float(local_cfg.get("time_limit_s", time_limit_s))
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
