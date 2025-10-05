from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time, math, random

from .state import SatState, ClauseInfo
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
    time_limit_s: float = 60.0

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


    from typing import Iterable

    def _enforce_hard_units(state: SatState) -> None:
        """
        Make the current assignment satisfy all HARD unit clauses (length 1).
        Recomputes clause true-counts afterward.
        """
        changed = False
        for c in state.clauses:
            if c.is_hard and len(c.lits) == 1:
                lit = c.lits[0]
                v = abs(lit)
                want_true = (lit > 0)
                if state.assign[v] != want_true:
                    state.assign[v] = want_true
                    changed = True
        if changed:
                # recompute all true_cnts once
            for c in state.clauses:
                c.true_cnt = state._compute_clause_true_count(c.lits)


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




def _extract_clauses(cnf) -> Tuple[int, List[ClauseInfo], List[List[int]], List[List[int]]]:
    # Support: nvars / nv / n_vars
    n = getattr(cnf, "nvars", getattr(cnf, "nv", getattr(cnf, "n_vars", None)))
    if n is None:
        raise ValueError("CNF parser must expose nvars (or nv / n_vars)")

    # Support: pos_occ/neg_occ OR pos_adj/neg_adj
    pos_occ = getattr(cnf, "pos_occ", getattr(cnf, "pos_adj", None))
    neg_occ = getattr(cnf, "neg_occ", getattr(cnf, "neg_adj", None))
    if pos_occ is None or neg_occ is None:
        raise ValueError("CNF parser must expose pos_occ/neg_occ or pos_adj/neg_adj")

    clauses: List[ClauseInfo] = []
    # (a) Parser exposes separate hard/soft lists
    if hasattr(cnf, "hard_clauses") and hasattr(cnf, "soft_clauses"):
        for lits in cnf.hard_clauses:
            clauses.append(ClauseInfo(lits=list(lits), base_w=0, is_hard=True))
        for w, lits in cnf.soft_clauses:
            clauses.append(ClauseInfo(lits=list(lits), base_w=int(w), is_hard=False))
    # (b) Unified arrays with weights/top
    elif hasattr(cnf, "clauses") and hasattr(cnf, "weights"):
        top = getattr(cnf, "top", None)
        if top is None:
            for w, lits in zip(cnf.weights, cnf.clauses):
                is_hard = (w == 0)
                bw = 0 if is_hard else int(w or 1)
                clauses.append(ClauseInfo(lits=list(lits), base_w=bw, is_hard=is_hard))
        else:
            for w, lits in zip(cnf.weights, cnf.clauses):
                is_hard = (w >= top)
                bw = 0 if is_hard else int(w or 1)
                clauses.append(ClauseInfo(lits=list(lits), base_w=bw, is_hard=is_hard))
    # (c) Your WCNF class: list of Clause(weight, lits, is_hard)
    elif hasattr(cnf, "clauses") and len(getattr(cnf, "clauses")) > 0 and \
         hasattr(cnf.clauses[0], "lits") and hasattr(cnf.clauses[0], "weight") and hasattr(cnf.clauses[0], "is_hard"):
        for cl in cnf.clauses:
            is_hard = bool(cl.is_hard)
            bw = 0 if is_hard else int(cl.weight or 1)
            clauses.append(ClauseInfo(lits=list(cl.lits), base_w=bw, is_hard=is_hard))
    else:
        # Fallback: treat all as soft weight 1
        for lits in getattr(cnf, "clauses", []):
            clauses.append(ClauseInfo(lits=list(lits), base_w=1, is_hard=False))
        if not clauses:
            raise ValueError("CNF parser interface not recognized.")

    return n, clauses, pos_occ, neg_occ


def _cfg(cfg: Dict[str, Any], key: str, default):
    return cfg.get(key, default)



def _freeze_hard_units(state: SatState) -> None:
    """
    Identify hard unit clauses, set their variables to satisfy the unit,
    recompute true-counts once, and mark those vars as frozen.
    """
    changed = False
    frozen = set()
    for c in state.clauses:
        if c.is_hard and len(c.lits) == 1:
            lit = c.lits[0]
            v = abs(lit)
            want_true = (lit > 0)
            frozen.add(v)
            if state.assign[v] != want_true:
                state.assign[v] = want_true
                changed = True
    state.frozen_vars |= frozen
    if changed:
        for c in state.clauses:
            c.true_cnt = state._compute_clause_true_count(c.lits)

def run_satlike(cnf, cfg: Dict[str, Any], rng_seed: int = 1) -> Dict[str, Any]:
    """
    SATLike-ish local search with:
      - O(occ(v)) incremental gains
      - dynamic soft weights (bump + smoothing)
      - tabu with aspiration
      - adaptive noise
      - hard-first repair (pre-phase + in-loop), optional
      - restarts by size rule
    """
    n, clauses, pos_occ, neg_occ = _extract_clauses(cnf)
    rng = random.Random(rng_seed)
    state = SatState(nvars=n, clauses=clauses, pos_occ=pos_occ, neg_occ=neg_occ, rng=rng)
    # Freeze variables from hard unit clauses (e.g., 10 1 0, 10 2 0)
    _freeze_hard_units(state)   
    _enforce_hard_units(state)
    # Knobs
    max_flips        = int(_cfg(cfg, "max_flips", 1_000_000))
    noise            = float(_cfg(cfg, "noise", 0.20))
    noise_min        = float(_cfg(cfg, "noise_min", 0.05))
    noise_max        = float(_cfg(cfg, "noise_max", 0.50))
    adapt_delta      = float(_cfg(cfg, "noise_adapt_delta", 0.05))
    adapt_every      = int(_cfg(cfg, "adapt_every", 2000))
    tabu_len         = int(_cfg(cfg, "tabu_length", 12))
    bump             = float(_cfg(cfg, "dyn_bump", 1.0))
    smooth_every     = int(_cfg(cfg, "smooth_every", 5000))
    rho              = float(_cfg(cfg, "rho", 0.5))  # 0<rho<1
    restart_after    = int(_cfg(cfg, "restart_after", 20000))
    restart_k        = int(max(500, min(15000, 5 * n)))
    hard_safe        = bool(_cfg(cfg, "hard_safe", True))
    time_limit_s     = float(_cfg(cfg, "time_limit_s", 60.0))
    hard_repair_budget = int(_cfg(cfg, "hard_repair_budget", 8000))

    start_t = time.time()
    last_improve_iter = 0

    def time_up() -> bool:
        return (time.time() - start_t) >= time_limit_s

    def pick_unsat_clause_index() -> int:
        hard_unsat = state.unsat_hard_ids()
        if hard_unsat:
            return rng.choice(hard_unsat)
        soft_unsat = state.unsat_soft_indices()
        if soft_unsat:
            return rng.choice(soft_unsat)
        return -1

    # Global hard-repair pre-phase
    if state._count_hard_violations() > 0 and hard_repair_budget > 0:
        for _ in range(hard_repair_budget):
            if time_up():
                break
            hard_unsat = state.unsat_hard_ids()
            if not hard_unsat:
                break
            cand = state.vars_adjacent_to(hard_unsat)
            best_v = None
            best_dh = 0
            best_gain = float("-inf")
            for v in cand:
                dh = state.flip_var_hard_delta(v)
                if dh < best_dh or (dh == best_dh and dh < 0):
                    g, _ = state.flip_var_effect(v)
                    if dh < 0 and g > best_gain:
                        best_gain = g
                        best_dh = dh
                        best_v = v
            if best_v is None:
                break
            state.apply_flip(best_v)
            if state._count_hard_violations() == 0:
                state.snapshot_best_if_better()
                break

    # Main loop
    while state.flips < max_flips and not time_up():
        target = pick_unsat_clause_index()
        if target == -1:
            state.snapshot_best_if_better()
            break

        if not state.clauses[target].is_hard:
            state.bump_clause(target, bump)

        clause = state.clauses[target]
        hv_now = state._count_hard_violations()

        if hv_now > 0:
            cand_vars = list(state.vars_adjacent_to(state.unsat_hard_ids()))
        else:
            cand_vars = [abs(l) for l in clause.lits]
        # Never consider frozen vars (hard units)
        cand_vars = [v for v in cand_vars if v not in state.frozen_vars]
        rng.shuffle(cand_vars)

        chosen_v = None
        chosen_gain = float("-inf")

        explore = (rng.random() < noise)

        if explore:
            if hv_now > 0:
                for v in cand_vars:
                    if state.flip_var_hard_delta(v) < 0:
                        chosen_v = v
                        chosen_gain, _ = state.flip_var_effect(v)
                        break
            else:
                for v in cand_vars:
                    if (not hard_safe) or state.hard_safe(v):
                        chosen_v = v
                        chosen_gain, _ = state.flip_var_effect(v)
                        break
        else:
            best_v = None
            best_gain = float("-inf")
            best_break = math.inf
            best_dh = math.inf
            aspirant_v = None
            aspirant_gain = float("-inf")
            aspirant_dh = math.inf

            for v in cand_vars:
                gain, br = state.flip_var_effect(v)
                dh = state.flip_var_hard_delta(v)
                is_tabu = state.var_is_tabu(v)

                if hv_now > 0:
                    if dh >= 0:
                        continue
                    if not is_tabu:
                        if (dh < best_dh) or (dh == best_dh and (gain > best_gain or (gain == best_gain and br < best_break))):
                            best_dh, best_gain, best_break = dh, gain, br
                            best_v = v
                    else:
                        if (dh < aspirant_dh) or (dh == aspirant_dh and gain > aspirant_gain):
                            aspirant_dh, aspirant_gain, aspirant_v = dh, gain, v
                else:
                    if hard_safe and br > 0:
                        continue
                    if not is_tabu:
                        if (gain > best_gain) or (gain == best_gain and br < best_break):
                            best_gain, best_break = gain, br
                            best_v = v
                    else:
                        if gain + state._soft_objective() > state.best_soft_obj and gain > aspirant_gain:
                            aspirant_gain = gain
                            aspirant_v = v

            chosen_v, chosen_gain = (aspirant_v, aspirant_gain) if aspirant_v is not None else (best_v, best_gain)

        if chosen_v is not None:
            state.apply_flip(chosen_v)
            state.set_tabu(chosen_v, tabu_len)

        if smooth_every > 0 and state.flips > 0 and (state.flips % smooth_every == 0):
            state.smooth(rho)

        improved = state._count_hard_violations() == 0 and state._soft_objective() > state.best_soft_obj
        if improved:
            last_improve_iter = state.iter_idx
            noise = max(noise_min, noise - adapt_delta)
            state.snapshot_best_if_better()
        elif state.iter_idx - last_improve_iter >= adapt_every:
            noise = min(noise_max, noise + adapt_delta)
            last_improve_iter = state.iter_idx

        if restart_after > 0 and state.iter_idx > 0 and (state.iter_idx % restart_after == 0):
            state.restart_partial_from_best(restart_k)

    elapsed = max(1e-9, time.time() - start_t)
    hv = state._count_hard_violations()
    soft = state._soft_objective()
    if state.best_assign is not None and hv == 0:
        soft = state.best_soft_obj
        hv = 0

    # Debug dump of unsatisfied hard clauses (first few)
    if hv > 0:
        uns = state.unsat_hard_ids()
        try:
            print(f"[DEBUG] hard violations: {hv}; unsatisfied hard clause ids: {uns[:10]}", flush=True)
            for cid in uns[:10]:
                print(f"[DEBUG] clause[{cid}] lits={state.clauses[cid].lits}", flush=True)
        except Exception:
            pass

    return {
        "best_soft_weight": float(soft),
        "hard_violations": int(hv),
        "total_flips": int(state.flips),
        "flips_per_sec": float(state.flips / elapsed),
        "restarts": int(state.restarts),
        "elapsed_sec": float(elapsed),
        "final_noise": float(noise),
    }
