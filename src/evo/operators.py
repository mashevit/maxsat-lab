from __future__ import annotations
from typing import List
import random
from .population import Individual, evaluate_assignment

def frozen_hard_unit_vars(wcnf) -> set:
    """Variables that must be fixed by hard unit clauses."""
    fr = set()
    for cl in wcnf.clauses:
        if cl.is_hard and len(cl.lits) == 1:
            fr.add(abs(cl.lits[0]))
    return fr


def tournament(pop: List[Individual], k: int, rng: random.Random) -> Individual:
    k = min(k, len(pop))
    cand = rng.sample(pop, k)
    return max(cand, key=lambda x: x.fitness)


def _soft_proxy_scores(wcnf) -> tuple[list[float], list[float]]:
    """
    Precompute soft scores per var for true/false using occurrence lists.
    Score(v=True) ≈ sum of weights of soft clauses where v appears positively.
    """
    n = wcnf.n_vars
    s_true = [0.0] * (n + 1)
    s_false = [0.0] * (n + 1)
    for i, cl in enumerate(wcnf.clauses):
        if cl.is_hard:  # ignore hard in soft proxy
            continue
        w = float(cl.weight)
        for lit in cl.lits:
            v = abs(lit)
            if lit > 0:
                s_true[v] += w
            else:
                s_false[v] += w
    return s_true, s_false


def clause_aware_crossover(p1: Individual, p2: Individual, wcnf, rng: random.Random) -> List[bool]:
    """
    Per-variable choice guided by soft occurrence scores; then hard repair.
    """
    n = wcnf.n_vars
    child = [False] * (n + 1)
    child[0] = False
    s_true, s_false = _soft_proxy_scores(wcnf)
    frozen = frozen_hard_unit_vars(wcnf)

    for v in range(1, n + 1):
        if v in frozen:
            # enforce hard unit later in repair; temporary pick from a parent
            child[v] = p1.assign01[v]
            continue
        a, b = p1.assign01[v], p2.assign01[v]
        if a == b:
            child[v] = a
            continue
        # soft proxy preference; break ties using fitter parent’s bit
        st, sf = s_true[v], s_false[v]
        if st > sf:
            child[v] = True
        elif sf > st:
            child[v] = False
        else:
            # tie -> prefer same value as the better parent
            better = p1 if p1.fitness >= p2.fitness else p2
            child[v] = better.assign01[v]

    # Hard repair: enforce hard units, then satisfy any remaining hard clauses greedily.
    _repair_hard_constraints(child, wcnf, frozen, max_iters=3 * n, rng=rng)
    return child


def _repair_hard_constraints(assign01: List[bool], wcnf, frozen: set, max_iters: int, rng: random.Random) -> None:
    # 1) enforce hard units
    for cl in wcnf.clauses:
        if cl.is_hard and len(cl.lits) == 1:
            lit = cl.lits[0]
            v = abs(lit)
            want_true = (lit > 0)
            assign01[v] = want_true

    # 2) greedy fix: repeatedly choose an unsatisfied hard clause and flip a literal to satisfy it
    it = 0
    while it < max_iters:
        it += 1
        # find first unsatisfied hard clause
        target = None
        for cl in wcnf.clauses:
            if not cl.is_hard:
                continue
            sat = False
            for lit in cl.lits:
                v = abs(lit)
                if (lit > 0 and assign01[v]) or (lit < 0 and not assign01[v]):
                    sat = True
                    break
            if not sat:
                target = cl
                break
        if target is None:
            return  # all hard satisfied

        # try flipping a literal that makes it true (skip frozen if present)
        rng.shuffle(target.lits)
        flipped = False
        for lit in target.lits:
            v = abs(lit)
            want_true = (lit > 0)
            if v in frozen:
                # frozen var already enforced; if still unsat here, try others
                continue
            # Flip v if it would satisfy this clause
            prev = assign01[v]
            if prev != want_true:
                assign01[v] = want_true
                flipped = True
                break
        if not flipped:
            # As a fallback, flip any non-frozen literal in the target
            for lit in target.lits:
                v = abs(lit)
                if v in frozen:
                    continue
                assign01[v] = (not assign01[v])
                flipped = True
                break
        if not flipped:
            # all vars frozen? nothing to do
            return


def mutate(assign01: List[bool], pmutate: float, rng: random.Random, frozen: set) -> None:
    n = len(assign01) - 1
    for v in range(1, n + 1):
        if v in frozen:
            pass#continue
        if rng.random() < pmutate:
            assign01[v] = (not assign01[v])


def short_polish(assign01: List[bool], wcnf, ls_cfg: dict, rng_seed: int) -> List[bool]:
    """
    Placeholder: if your WalkSAT polishing later accepts a start assignment, wire it here.
    For now we simply return the child unchanged to keep the pipeline deterministic.
    """
    # from ..sat import walksat
    # cfg = dict(ls_cfg); cfg['max_flips'] = cfg.get('ls_polish_flips', 700); ...
    # stats = walksat.run_satlike(wcnf, cfg, rng_seed=rng_seed)
    return assign01
