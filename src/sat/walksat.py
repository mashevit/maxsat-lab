from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .cnf import WCNF

@dataclass
class WalkSATConfig:
    max_flips: int = 2000
    restarts: int = 3
    patience: int = 150
    noise: float = 0.30
    tabu_length: int = 12
    forbid_break_hard: bool = True

class WalkSAT:
    """
    Simplified WalkSAT/SATLike-style skeleton:
    - Prefers not to break hard clauses.
    - Chooses a random unsatisfied clause and flips a variable using
      a noise / best-gain mixture (very simplified for wiring).
    """
    def __init__(self, cnf: WCNF, cfg: WalkSATConfig, rng: Optional[random.Random] = None):
        self.cnf = cnf
        self.cfg = cfg
        self.rng = rng or random.Random(42)

    def _rand_assign(self) -> List[int]:
        # 1-indexed assignment of 0/1
        return [0] + [self.rng.randint(0, 1) for _ in range(self.cnf.n_vars)]

    def _clause_satisfied(self, cl_idx: int, assign01: List[int]) -> bool:
        cl = self.cnf.clauses[cl_idx]
        return any((lit > 0 and assign01[abs(lit)] == 1) or (lit < 0 and assign01[abs(lit)] == 0) for lit in cl.lits)

    def _would_break_hard(self, var: int, assign01: List[int]) -> bool:
        """Return True if flipping var would make any hard clause unsatisfied."""
        new_val = 1 - assign01[var]
        # Check only hard clauses touching var
        # Clauses satisfied by other lits remain satisfied; we check conservatively by recomputing
        touched = set(self.cnf.pos_adj[var] + self.cnf.neg_adj[var])
        for cid in touched:
            cl = self.cnf.clauses[cid]
            if not cl.is_hard:
                continue
            # simulate satisfaction under flip
            sat = False
            for lit in cl.lits:
                v = abs(lit)
                val = assign01[v] if v != var else new_val
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    sat = True
                    break
            if not sat:
                return True
        return False

    def _pick_var_to_flip(self, unsat_clause_idx: int, assign01: List[int]) -> Optional[int]:
        cl = self.cnf.clauses[unsat_clause_idx]
        candidates = [abs(l) for l in cl.lits] if cl.lits else []
        self.rng.shuffle(candidates)
        # With 'noise', just pick a random allowed variable
        if self.rng.random() < self.cfg.noise:
            for v in candidates:
                if not self.cfg.forbid_break_hard or not self._would_break_hard(v, assign01):
                    return v
            return None
        # Otherwise pick the first that doesn't break hard
        best_v = None
        for v in candidates:
            if not self.cfg.forbid_break_hard or not self._would_break_hard(v, assign01):
                best_v = v
                break
        return best_v

    def run(self, max_flips: Optional[int] = None, restarts: Optional[int] = None) -> Tuple[List[int], int]:
        max_flips = max_flips or self.cfg.max_flips
        restarts = restarts or self.cfg.restarts

        best_assign = None
        best_score = -1

        for _r in range(restarts):
            assign01 = self._rand_assign()
            flips = 0
            while flips < max_flips:
                # Evaluate and track best
                sat_w, hard_v, soft_v = self.cnf.eval_assignment(assign01)
                if hard_v == 0 and sat_w > best_score:
                    best_assign = assign01.copy()
                    best_score = sat_w
                # Collect unsatisfied clauses (both hard and soft)
                unsat = [i for i, cl in enumerate(self.cnf.clauses) if not self._clause_satisfied(i, assign01)]
                if not unsat:
                    # fully satisfied
                    return assign01, sat_w
                # Prefer soft clause to improve objective; fall back to any unsat
                soft_unsat = [i for i in unsat if not self.cnf.clauses[i].is_hard]
                target = self.rng.choice(soft_unsat if soft_unsat else unsat)
                v = self._pick_var_to_flip(target, assign01)
                if v is None:
                    # no safe flip; random restart
                    break
                assign01[v] = 1 - assign01[v]
                flips += 1
        if best_assign is None:
            # just return last assignment
            assign01 = self._rand_assign()
            sat_w, _, _ = self.cnf.eval_assignment(assign01)
            return assign01, sat_w
        return best_assign, best_score
