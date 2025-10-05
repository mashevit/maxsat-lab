from __future__ import annotations
import random
from typing import List, Dict, Any, Tuple
from ..sat import walksat

def tournament(pop: List[List[bool]], fit: List[float], k: int, rng: random.Random) -> int:
    cand = rng.sample(range(len(pop)), k)
    return max(cand, key=lambda i: fit[i])

def clause_aware_crossover(p1: List[bool], p2: List[bool], cnf, rng: random.Random) -> List[bool]:
    # pick per-variable from the parent that satisfies more (weighted) soft clauses
    child = [False] * (len(p1))
    child[0] = False
    for v in range(1, len(p1)):
        # heuristic: favor p1 or p2 by literal occurrence counts as a proxy
        pos = len(getattr(cnf, "pos_adj")[v])
        neg = len(getattr(cnf, "neg_adj")[v])
        child[v] = p1[v] if (p1[v] == p2[v]) else (p1[v] if pos >= neg else p2[v])
    return child

def mutate(assign: List[bool], pmutate: float, rng: random.Random) -> None:
    for v in range(1, len(assign)):
        if rng.random() < pmutate: assign[v] = (not assign[v])

def polish(assign: List[bool], cnf, ls_cfg: Dict[str, Any], rng_seed: int) -> Tuple[List[bool], Dict[str, Any]]:
    cfg = dict(ls_cfg)
    cfg["max_flips"] = int(cfg.get("ls_polish_flips", 700))
    stats = walksat.run_satlike(cnf, cfg, rng_seed=rng_seed)
    # NOTE: run_satlike currently samples its own start; adapt to accept a seed assignment if you add that feature.
    return assign, stats

def run_ea(cnf, cfg: Dict[str, Any], rng_seed: int = 1) -> Dict[str, Any]:
    rng = random.Random(rng_seed)
    ea = cfg.get("ea", {})
    if not ea.get("enabled", False):
        return walksat.run_satlike(cnf, cfg, rng_seed=rng_seed)

    pop_size = ea.get("pop_size", 60)
    k = ea.get("tournament_k", 4)
    pmutate = ea.get("pmutate", 0.02)

    # init random population
    pop = [[False] + [rng.choice((False, True)) for _ in range(cnf.n_vars)] for _ in range(pop_size)]
    fit = [0.0] * pop_size  # TODO: evaluate by calling a quick LS with small budget or a fast proxy

    # main loop skeleton
    for _ in range(100):  # TODO: stop conditions
        i = tournament(pop, fit, k, rng); j = tournament(pop, fit, k, rng)
        c = clause_aware_crossover(pop[i], pop[j], cnf, rng)
        mutate(c, pmutate, rng)
        # optional short polish here
        # replace worst
        worst = min(range(pop_size), key=lambda t: fit[t])
        pop[worst] = c
        fit[worst] = 0.0  # TODO: re-evaluate
    return {"best_soft_weight": 0.0, "hard_violations": 0}
