from __future__ import annotations
from typing import Dict, Any, List
import time, math, random

from .population import Population, Individual, evaluate_assignment, build_hard_occurs
from .operators import tournament, clause_aware_crossover, mutate,mutate1, frozen_hard_unit_vars, short_polish, clause_aware_crossover1


def _ea_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ea = cfg.get("ea", {})
    return {
        "enabled": bool(ea.get("enabled", False)),
        "pop_size": int(ea.get("pop_size", 60)),
        "tournament_k": int(ea.get("tournament_k", 4)),
        "pmutate": float(ea.get("pmutate", 0.02)),
        "elitism": bool(ea.get("elitism", True)),
        "ls_polish_flips": int(ea.get("ls_polish_flips", 700)),
    }


def _ls_budget(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Small polishing budget by default
    ls = cfg.get("ls", {})
    return {
        "ls_polish_flips": int(ls.get("ls_polish_flips", 700)),
        "time_limit_s": float(ls.get("time_limit_s", 0.05)),
        "max_flips": int(ls.get("flip_budget", 2000)),
    }


def run_memetic(wcnf, cfg: Dict[str, Any], rng_seed: int = 1) -> Dict[str, Any]:
    """
    Minimal memetic loop:
      - JW seeding
      - tournament selection, clause-aware crossover, mutation
      - (stub) short polish
      - elitist replacement
    Fitness evaluates *assignments directly* (soft weight, penalize hard violations).
    """
    ea = _ea_cfg(cfg)
    pop_size = ea["pop_size"]
    #pop_size = wcnf.n_vars * 2
    k = ea["tournament_k"]
    pmutate = ea["pmutate"]
    elitism = ea["elitism"]

    rng = random.Random(rng_seed)
    pop = Population(n_vars=wcnf.n_vars, size=pop_size, rng=rng)
    pop.init_seeds(wcnf, cfg)

    # initial evaluation
    for ind in pop.members:
        pop.evaluate(wcnf, ind)

    #frozen = frozen_hard_unit_vars(wcnf)#not needed
    best = pop.best().copy()

    # stop conditions
    time_cap = float(cfg.get("time_limit_s", cfg.get("ls", {}).get("time_limit_s", 10.0)))
    start_t = time.time()
    max_gens = int(cfg.get("ea", {}).get("max_gens", 100))

    ls_small = _ls_budget(cfg)
    gen = 0
    total_children = 0
        # 1) derive hard_clauses once
    hard_clauses = [cl for cl in wcnf.clauses if cl.is_hard]

    # 2) precompute occurrences
    hard_occurs = build_hard_occurs(hard_clauses, wcnf.n_vars)
    flips_t =0
    while (time.time() - start_t) < time_cap and gen < max_gens:
        gen += 1
        # Elites
        if elitism:
            elites_cnt = max(1, math.ceil(0.05 * pop_size))
            elites = sorted(pop.members, key=lambda x: x.fitness, reverse=True)[:elites_cnt]
        else:
            elites = []

        new_members: List[Individual] = elites.copy()
        
        # Fill the rest
        while len(new_members) < pop_size:
            p1 = tournament(pop.members, k, rng)
            p2 = tournament(pop.members, k, rng)
            child_bits = clause_aware_crossover1(p1, p2, wcnf, rng)
            #mutate(child_bits, pmutate, rng, frozen=frozen)
            mutate1(child_bits, pmutate, rng, hard_clauses, hard_occurs, ind.hard_satisfied)
            child_bits, flips_t1 = short_polish(child_bits, wcnf, ls_small, rng_seed=rng.randrange(1<<30))
            flips_t += flips_t1
            child = Individual(assign01=child_bits, meta={"gen": gen})
            pop.evaluate(wcnf, child)
            new_members.append(child)
            total_children += 1

        pop.members = new_members
        # track best
        cur_best = pop.best()
        if cur_best.fitness > best.fitness:
            best = cur_best.copy()



    def _assignment_exports(assign01: list[bool]) -> dict:
        # index 0 is unused in your code; export 1..n
        bits = "".join("1" if b else "0" for b in assign01[1:])
        dimacs = "v " + " ".join(str(i if assign01[i] else -i) for i in range(1, len(assign01))) + " 0"
        true_vars = [i for i in range(1, len(assign01)) if assign01[i]]
        return {"assign_bits": bits, "dimacs": dimacs, "true_vars": true_vars}
        #results = {"assign_bits": bits, "dimacs": dimacs, "true_vars": true_vars}
    def clause_satisfied_bits(clause, bits: List[bool]) -> bool:
        """
        clause.lits: list of ints (e.g. [1, -3, 4])
        bits[v]: True/False, 1-based (bits[0] unused)
        """
        for lit in clause.lits:
            v = abs(lit)
            val = bits[v]
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    def count_satisfied_clauses_bits(wcnf, bits: List[bool]) -> int:
        return sum(1 for cl in wcnf.clauses if clause_satisfied_bits(cl, bits))

    results = _assignment_exports(best.assign01)
    # Using your results dict:
    def bits_to_assign01(bits: str) -> List[bool]:
        assert all(c in "01" for c in bits)
        # index 0 unused; bits[0] -> var 1, bits[1] -> var 2, ...
        return [False] + [c == "1" for c in bits]

    assign01 = bits_to_assign01(results["assign_bits"])
    #bits = f'0{results["assign_bits"]}'
    #wcnf = results["dimacs"]

    total_clauses = len(wcnf.clauses)
    sat_clauses = count_satisfied_clauses_bits(wcnf, assign01)
    unsat_clauses = total_clauses - sat_clauses
    elapsed = max(1e-9, time.time() - start_t)
    exports = _assignment_exports(best.assign01)
    soft, hv = evaluate_assignment(wcnf, best.assign01)
    return {
        "best_soft_weight": float(soft),
        "hard_violations": int(hv),
        "elapsed_sec": float(elapsed),
        "total_flips": int(flips_t),          # we didn't call LS yet; wire later if you add polish with flips
        "flips_per_sec": 0.0,
        "restarts": 0,
        "final_noise": 0.0,
        "meta": {"ea_generations": gen, "children": total_children, **exports,},
        "satisfied_clauses": {
            "total": total_clauses,
            "satisfied": sat_clauses,
            "unsatisfied": unsat_clauses,
        },
    }
