from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import random
import hashlib

# Expect: from src.sat.cnf import WCNF
# WCNF fields used: n_vars, clauses (with .weight, .lits, .is_hard), pos_adj, neg_adj


@dataclass
class Individual:
    """Simple GA individual: 1-based assignment, index 0 unused."""
    assign01: List[bool]
    fitness: float = float("-inf")
    hard_violations: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Individual":
        return Individual(self.assign01.copy(), self.fitness, self.hard_violations, dict(self.meta))


def jw_priors(wcnf) -> List[float]:
    """
    Jeroslow–Wang prior per variable (soft-clause only):
    p(x=True) proportional to sum over soft clauses: weight * 2^(-|C|)
    """
    n = wcnf.n_vars
    p_true = [0.0] * (n + 1)
    p_false = [0.0] * (n + 1)
    for i, cl in enumerate(wcnf.clauses):
        if cl.is_hard:
            continue
        w = float(cl.weight)
        scale = w * (2.0 ** (-max(1, len(cl.lits))))
        for lit in cl.lits:
            v = abs(lit)
            if lit > 0:
                p_true[v] += scale
            else:
                p_false[v] += scale
    pri = [0.5] * (n + 1)
    for v in range(1, n + 1):
        s = p_true[v] + p_false[v]
        pri[v] = (p_true[v] / s) if s > 0 else 0.5
        # clip per spec
        pri[v] = min(0.95, max(0.05, pri[v]))
    return pri


def hash_assign(assign01: List[bool]) -> str:
    """Stable hash for caching. Index 0 ignored."""
    b = bytearray( (1 if x else 0) for x in assign01[1:] )
    return hashlib.blake2b(b, digest_size=12).hexdigest()


def evaluate_assignment(wcnf, assign01: List[bool]) -> Tuple[float, int]:
    """
    Fast fitness proxy: soft weight satisfied & #hard violations.
    Returns (soft_weight, hard_violations).
    """
    soft_w = 0.0
    hard_viol = 0
    for cl in wcnf.clauses:
        sat = False
        for lit in cl.lits:
            v = abs(lit)
            if (lit > 0 and assign01[v]) or (lit < 0 and not assign01[v]):
                sat = True
                break
        if cl.is_hard:
            if not sat:
                hard_viol += 1
        else:
            if sat:
                soft_w += float(cl.weight)
    return soft_w, hard_viol


class Population:
    """
    Minimal population manager with JW seeding and caching.
    """
    def __init__(self, n_vars: int, size: int, rng: random.Random):
        self.n_vars = n_vars
        self.size = size
        self.rng = rng
        self.members: List[Individual] = []
        self.cache: Dict[str, Tuple[float, int]] = {}  # hash -> (fitness, hard_viol)

    def _new_assign_from_priors(self, pri: List[float]) -> List[bool]:
        a = [False] * (self.n_vars + 1)
        a[0] = False
        for v in range(1, self.n_vars + 1):
            a[v] = (self.rng.random() < pri[v])
        return a

    def init_seeds(self, wcnf, cfg: Dict[str, Any]) -> None:
        pri = jw_priors(wcnf)
        cnt = self.size
        for _ in range(cnt):
            a = self._new_assign_from_priors(pri)
            ind = Individual(assign01=a)
            self.members.append(ind)

    def evaluate(self, wcnf, ind: Individual) -> Individual:
        h = hash_assign(ind.assign01)
        if h in self.cache:
            fit, hv = self.cache[h]
        else:
            soft, hv = evaluate_assignment(wcnf, ind.assign01)
            # Fitness: penalize any hard violation heavily
            fit = soft if hv == 0 else (-1e9 - 1e6 * hv)
            self.cache[h] = (fit, hv)
        ind.fitness = fit
        ind.hard_violations = hv
        return ind

    def best(self) -> Individual:
        assert self.members, "Population is empty"
        return max(self.members, key=lambda x: x.fitness)
