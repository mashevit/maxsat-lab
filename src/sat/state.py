from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable
import random


@dataclass
class ClauseInfo:
    lits: List[int]
    base_w: int
    is_hard: bool
    dyn_w: float = 0.0  # set at init to base_w
    true_cnt: int = 0
    age: int = 0        # optional: increments when unsatisfied


@dataclass
class SatState:
    """
    Holds the evolving SAT/MaxSAT local-search state.
    - assignment: 1-based variable assignment in {False, True}; index 0 unused
    - clauses: unified list of ClauseInfo, hard first or mixed; indices map to pos/neg occ
    - pos_occ/neg_occ: occurrence lists: for var v -> list of clause indices where v or ~v appears
    """
    nvars: int
    clauses: List[ClauseInfo]
    pos_occ: List[List[int]]
    neg_occ: List[List[int]]
    rng: random.Random
    # counters & book-keeping
    flips: int = 0
    restarts: int = 0
    best_soft_obj: float = float("-inf")
    best_assign: Optional[List[bool]] = None
    hard_violations: int = 0
    # current assignment
    assign: List[bool] = field(default_factory=list)
    # tabu tenure storage: iteration index when var becomes free
    tabu_until: List[int] = field(default_factory=list)
    iter_idx: int = 0

    def __post_init__(self):
        if not self.assign:
            # Initialize random assignment in {False, True}
            self.assign = [False] + [self.rng.choice((False, True)) for _ in range(self.nvars)]
        if not self.tabu_until:
            self.tabu_until = [0] * (self.nvars + 1)
        # Initialize dynamic weights and true-counts
        for c in self.clauses:
            c.dyn_w = float(c.base_w)
            c.true_cnt = self._compute_clause_true_count(c.lits)
        # Compute initial objective & hard violations
        self.hard_violations = self._count_hard_violations()
        self.best_soft_obj = self._soft_objective()
        self.best_assign = self.assign.copy()

    def _lit_val(self, lit: int) -> bool:
        v = abs(lit)
        val = self.assign[v]
        return val if lit > 0 else (not val)

    def _compute_clause_true_count(self, lits: Iterable[int]) -> int:
        return sum(1 for lit in lits if self._lit_val(lit))

    def _count_hard_violations(self) -> int:
        return sum(1 for c in self.clauses if c.is_hard and c.true_cnt == 0)

    def _soft_objective(self) -> float:
        # Sum dynamic weights of satisfied soft clauses
        return sum(c.dyn_w for c in self.clauses if (not c.is_hard) and c.true_cnt > 0)

    def clause_indices_for_var(self, v: int) -> Tuple[List[int], List[int]]:
        return self.pos_occ[v], self.neg_occ[v]

    def var_is_tabu(self, v: int) -> bool:
        return self.iter_idx < self.tabu_until[v]

    def set_tabu(self, v: int, tenure: int) -> None:
        self.tabu_until[v] = self.iter_idx + tenure

    def flip_var_effect(self, v: int) -> Tuple[float, int]:
        """
        Compute gain (delta soft objective using dyn_w) and number of broken hard clauses
        if we flip variable v. O(occ(v)).
        """
        gain = 0.0
        break_hard = 0
        # For clauses where v appears positively
        for ci in self.pos_occ[v]:
            c = self.clauses[ci]
            was_sat = c.true_cnt > 0
            # flipping v from val->~val
            if self.assign[v]:  # v currently True; positive lit contributes 1 to true_cnt
                # becomes false: true_cnt decreases
                t_after = c.true_cnt - 1
            else:
                # was False, becomes True: true_cnt increases
                t_after = c.true_cnt + 1
            if c.is_hard:
                if was_sat and t_after == 0:
                    break_hard += 1
            else:
                if (not was_sat) and t_after > 0:
                    gain += c.dyn_w
                elif was_sat and t_after == 0:
                    gain -= c.dyn_w
        # For clauses where v appears negated
        for ci in self.neg_occ[v]:
            c = self.clauses[ci]
            was_sat = c.true_cnt > 0
            if self.assign[v]:  # v True makes (-v) false; flipping makes it true
                t_after = c.true_cnt + 1
            else:
                t_after = c.true_cnt - 1
            if c.is_hard:
                if was_sat and t_after == 0:
                    break_hard += 1
            else:
                if (not was_sat) and t_after > 0:
                    gain += c.dyn_w
                elif was_sat and t_after == 0:
                    gain -= c.dyn_w
        return gain, break_hard

    def apply_flip(self, v: int) -> None:
        """Apply flip of variable v and update true-counts incrementally. O(occ(v))."""
        self.assign[v] = (not self.assign[v])
        # Update positive occurrences
        for ci in self.pos_occ[v]:
            c = self.clauses[ci]
            if self.assign[v]:
                c.true_cnt += 1
            else:
                c.true_cnt -= 1
        # Update negative occurrences
        for ci in self.neg_occ[v]:
            c = self.clauses[ci]
            if self.assign[v]:
                c.true_cnt -= 1
            else:
                c.true_cnt += 1
        self.flips += 1
        self.iter_idx += 1

    def hard_safe(self, v: int) -> bool:
        _, br = self.flip_var_effect(v)
        return br == 0

    def snapshot_best_if_better(self) -> bool:
        soft = self._soft_objective()
        hv = self._count_hard_violations()
        if hv == 0 and soft > self.best_soft_obj:
            self.best_soft_obj = soft
            self.best_assign = self.assign.copy()
            return True
        return False

    def unsat_soft_indices(self) -> List[int]:
        return [i for i, c in enumerate(self.clauses) if (not c.is_hard) and c.true_cnt == 0]

    def all_unsat_indices(self) -> List[int]:
        return [i for i, c in enumerate(self.clauses) if c.true_cnt == 0]

    def bump_clause(self, idx: int, bump: float) -> None:
        c = self.clauses[idx]
        if not c.is_hard:
            c.dyn_w += bump
            if c.dyn_w < c.base_w:
                c.dyn_w = float(c.base_w)

    def smooth(self, rho: float) -> None:
        """Move dyn_w toward base_w: dyn = base + rho*(dyn-base), 0<rho<1."""
        for c in self.clauses:
            if not c.is_hard and c.dyn_w > c.base_w:
                c.dyn_w = c.base_w + rho * (c.dyn_w - c.base_w)

    def restart_partial_from_best(self, k: int) -> None:
        """Restart from best assignment (if any), then flip k random vars."""
        if self.best_assign is not None:
            self.assign = self.best_assign.copy()
            # recompute true_cnts from scratch for robustness
            for c in self.clauses:
                c.true_cnt = self._compute_clause_true_count(c.lits)
        vars_list = list(range(1, self.nvars + 1))
        self.rng.shuffle(vars_list)
        for v in vars_list[:k]:
            self.apply_flip(v)
        self.restarts += 1
