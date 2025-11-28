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





def clause_aware_crossover1(
    p1: Individual, 
    p2: Individual, 
    wcnf, 
    rng: random.Random
) -> List[bool]:
    """
    Per-variable choice guided by soft occurrence scores + hard-clauses awareness.
    
    For each variable v:
      1. Candidate bits come from the two parents (a = p1[v], b = p2[v]).
      2. For each candidate bit, estimate how many hard clauses it would newly violate
         and how many it would newly satisfy.
      3. Prefer the bit that:
         - minimizes new violations of hard clauses, and
         - among ties, maximizes newly satisfied hard clauses.
      4. If still tied, fall back to soft proxy scores, then fitter parent.
    """
    n = wcnf.n_vars
    child = [False] * (n + 1)
    child[0] = False

    # Soft guidance (unchanged logic)
    s_true, s_false = _soft_proxy_scores(wcnf)

    # --- Build hard-clause structures ---------------------------------------
    hard_clauses = []
    for cl in wcnf.clauses:
        if getattr(cl, "is_hard", False):
            hard_clauses.append(cl)

    m_hard = len(hard_clauses)

    # For each hard clause: whether already satisfied, and how many literals still unassigned
    hard_satisfied = [False] * m_hard
    hard_unassigned = [len(cl.lits) for cl in hard_clauses]

    # Occurrence lists for quick per-variable effect computation
    pos_occ = [[] for _ in range(n + 1)]  # pos_occ[v] = [hard_clause_index,...] where literal is +v
    neg_occ = [[] for _ in range(n + 1)]  # neg_occ[v] = [hard_clause_index,...] where literal is -v

    for cid, cl in enumerate(hard_clauses):
        for lit in cl.lits:
            v = abs(lit)
            if v == 0 or v > n:
                continue
            if lit > 0:
                pos_occ[v].append(cid)
            else:
                neg_occ[v].append(cid)

    def eval_candidate(v: int, val: bool) -> tuple[int, int]:
        """
        Evaluate effect of setting variable v := val on hard clauses.

        Returns:
            (delta_violated, delta_satisfied)
            - delta_violated: how many *new* hard clauses become violated
            - delta_satisfied: how many become newly satisfied
        """
        delta_violated = 0
        delta_satisfied = 0

        # Positive occurrences: literal is x_v
        for cid in pos_occ[v]:
            if hard_satisfied[cid]:
                # Clause already satisfied; assigning v can't unsatisfy it (we never flip previous bits)
                continue

            # Before assigning v, this clause has hard_unassigned[cid] unassigned literals,
            # and none of them is True yet (otherwise hard_satisfied[cid] would be True).
            if val:  # literal x_v is True
                # Clause becomes satisfied
                delta_satisfied += 1
            else:
                # literal is False. After assignment, unassigned decreases by 1.
                # If it was 1, then after assignment it becomes 0 with no True literal → violated.
                if hard_unassigned[cid] == 1:
                    delta_violated += 1

        # Negative occurrences: literal is ¬x_v
        for cid in neg_occ[v]:
            if hard_satisfied[cid]:
                continue

            if not val:  # x_v = False ⇒ ¬x_v is True
                delta_satisfied += 1
            else:
                if hard_unassigned[cid] == 1:
                    delta_violated += 1

        return delta_violated, delta_satisfied

    def commit_assignment(v: int, val: bool) -> None:
        """
        Actually set v := val in the child and update hard clause bookkeeping.
        """
        child[v] = val

        # Positive occurrences
        for cid in pos_occ[v]:
            if hard_satisfied[cid]:
                # Still decrement unassigned for consistency, but satisfaction remains.
                hard_unassigned[cid] -= 1
                continue

            if val:  # literal x_v is True
                hard_satisfied[cid] = True
                hard_unassigned[cid] -= 1
            else:
                # literal is False, clause remains unsatisfied but with one fewer unassigned
                hard_unassigned[cid] -= 1

        # Negative occurrences
        for cid in neg_occ[v]:
            if hard_satisfied[cid]:
                hard_unassigned[cid] -= 1
                continue

            if not val:  # x_v = False ⇒ ¬x_v True
                hard_satisfied[cid] = True
                hard_unassigned[cid] -= 1
            else:
                hard_unassigned[cid] -= 1

    # --- Main per-variable crossover loop -----------------------------------
    for v in range(1, n + 1):
        a = p1.assign01[v]
        b = p2.assign01[v]

        # If parents agree, keep that bit (no need to evaluate)
        if a == b:
            commit_assignment(v, a)
            continue

        # Evaluate both candidate bits wrt hard clauses
        dv_a, ds_a = eval_candidate(v, a)
        dv_b, ds_b = eval_candidate(v, b)

        # Prefer fewer new violations
        if dv_a < dv_b:
            chosen = a
        elif dv_b < dv_a:
            chosen = b
        else:
            # Same number of new violations → prefer more newly satisfied clauses
            if ds_a > ds_b:
                chosen = a
            elif ds_b > ds_a:
                chosen = b
            else:
                # Still tied: fall back to soft scores, then fitter parent, then random
                st, sf = s_true[v], s_false[v]

                if st > sf:
                    chosen = True
                elif sf > st:
                    chosen = False
                else:
                    better = p1 if p1.fitness >= p2.fitness else p2
                    chosen = better.assign01[v]

                    # If even that doesn't disambiguate (rare), use RNG
                    if a != b and chosen not in (a, b):
                        chosen = rng.choice([a, b])

        commit_assignment(v, chosen)

    return child
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
