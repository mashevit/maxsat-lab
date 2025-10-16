# =========================
# File: src/evo/population.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import random

Assignment = List[int]  # 1-indexed: assignment[v] in {0,1}; index 0 unused

@dataclass
class EvalResult:
    hard_violations: int
    soft_unsat_weight: float
    fitness: float

@dataclass
class Individual:
    genes: Assignment
    eval: EvalResult


def _lit_true(lit: int, assign: Assignment) -> bool:
    v = abs(lit)
    val = assign[v]
    return (val == 1 and lit > 0) or (val == 0 and lit < 0)


def _eval_with_cnf(cnf, assign: Assignment, hard_penalty: Optional[float] = None) -> EvalResult:
    hv = 0
    soft_unsat = 0.0
    if hard_penalty is None:
        # Big-M: larger than any possible soft weight sum
        soft_sum = 0.0
        hard_sum = 0.0
        for cl in cnf.clauses:
            if getattr(cl, "is_hard", False):
                hard_sum += float(getattr(cl, "weight", 1.0) or 1.0)
            else:
                soft_sum += float(getattr(cl, "weight", 1.0) or 1.0)
        hard_penalty = max(1.0, soft_sum + hard_sum) * 10.0
    for cl in cnf.clauses:
        sat = False
        for lit in getattr(cl, "lits", cl):
            if _lit_true(lit, assign):
                sat = True
                break
        if not sat:
            if getattr(cl, "is_hard", False):
                hv += 1
            else:
                soft_unsat += float(getattr(cl, "weight", 1.0) or 1.0)
    fitness = -(soft_unsat + hv * hard_penalty)
    return EvalResult(hv, soft_unsat, fitness)


def _random_assignment(nvars: int, rng: random.Random) -> Assignment:
    a = [0] * (nvars + 1)
    for v in range(1, nvars + 1):
        a[v] = 1 if rng.random() < 0.5 else 0
    return a


def _jw_seed(cnf, rng: random.Random) -> Assignment:
    # Weighted Jeroslow–Wang per literal; soft weights included; shorter clauses favored
    nvars = int(getattr(cnf, "nvars", 0) or len(getattr(cnf, "pos_adj", {})))
    pos_score = [0.0] * (nvars + 1)
    neg_score = [0.0] * (nvars + 1)
    for cl in cnf.clauses:
        w = float(getattr(cl, "weight", 1.0) or 1.0)
        k = max(1, len(getattr(cl, "lits", cl)))
        s = w * (2.0 ** (-k))
        for lit in getattr(cl, "lits", cl):
            v = abs(lit)
            if lit > 0:
                pos_score[v] += s
            else:
                neg_score[v] += s
    a = [0] * (nvars + 1)
    for v in range(1, nvars + 1):
        # pick sign with higher score
        a[v] = 1 if pos_score[v] >= neg_score[v] else 0
        # tiny randomization to diversify equal scores
        if pos_score[v] == neg_score[v] and rng.random() < 0.1:
            a[v] ^= 1
    return a


@dataclass
class EAConfig:
    pop_size: int = 40
    gens: int = 60
    tournament_k: int = 4
    elite: int = 2
    pmutate: float = 0.02
    crossover_bias: float = 0.7  # probability to take gene from fitter parent
    ls_polish_flips: int = 2000  # short polish flips/time budget handed to LS
    seed: int = 1
    use_jw_seed: bool = True


class Population:
    def __init__(self, cnf, ea_cfg: EAConfig, rng: Optional[random.Random] = None, frozen_vars: Optional[set] = None):
        self.cnf = cnf
        self.cfg = ea_cfg
        self.rng = rng or random.Random(ea_cfg.seed)
        self.frozen_vars = frozen_vars or set()
        nvars = int(getattr(cnf, "nvars", 0) or len(getattr(cnf, "pos_adj", {})))
        self.nvars = nvars
        self.individuals: List[Individual] = []
        self._init(nvars)

    def _init(self, nvars: int):
        for _ in range(self.cfg.pop_size):
            if self.cfg.use_jw_seed and self.rng.random() < 0.7:
                genes = _jw_seed(self.cnf, self.rng)
            else:
                genes = _random_assignment(nvars, self.rng)
            # respect frozen vars if any (e.g., hard-core preprocessing)
            for v in self.frozen_vars:
                # assume cnf exposes forced values via cnf.frozen_assign.get(v)
                if hasattr(self.cnf, "frozen_assign") and v in self.cnf.frozen_assign:
                    genes[v] = 1 if self.cnf.frozen_assign[v] else 0
            ev = _eval_with_cnf(self.cnf, genes)
            self.individuals.append(Individual(genes, ev))
        self.individuals.sort(key=lambda ind: ind.eval.fitness, reverse=True)

    def evaluate(self, ind: Individual):
        ind.eval = _eval_with_cnf(self.cnf, ind.genes)

    def select_tournament(self) -> Individual:
        k = min(self.cfg.tournament_k, len(self.individuals))
        picks = self.rng.sample(self.individuals, k)
        return max(picks, key=lambda ind: ind.eval.fitness)

    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.eval.fitness)

    def stats(self) -> Dict:
        best = self.best()
        avg_fit = sum(i.eval.fitness for i in self.individuals) / max(1, len(self.individuals))
        return {
            "best_fitness": best.eval.fitness,
            "best_soft_unsat": best.eval.soft_unsat_weight,
            "best_hard_violations": best.eval.hard_violations,
            "avg_fitness": avg_fit,
        }


# =========================
# File: src/evo/operators.py
# =========================
from __future__ import annotations
from typing import List, Tuple, Optional, Set
import random

from .population import Assignment, Individual, EvalResult, _eval_with_cnf


def weighted_uniform_crossover(parent_a: Individual, parent_b: Individual, rng: random.Random, bias_toward_a: float, frozen: Set[int]) -> Assignment:
    # bias toward fitter parent (assume parent_a is fitter); respect frozen vars
    nvars = len(parent_a.genes) - 1
    child = [0] * (nvars + 1)
    for v in range(1, nvars + 1):
        if v in frozen:
            child[v] = parent_a.genes[v]
            continue
        p = bias_toward_a
        take_a = (rng.random() < p)
        child[v] = parent_a.genes[v] if take_a else parent_b.genes[v]
    return child


def mutate_bitflip(genes: Assignment, rng: random.Random, pmutate: float, frozen: Set[int]) -> None:
    nvars = len(genes) - 1
    for v in range(1, nvars + 1):
        if v in frozen:
            continue
        if rng.random() < pmutate:
            genes[v] ^= 1


def greedy_hard_repair(cnf, assign: Assignment, max_iters: int = 10000) -> Assignment:
    # Greedy hill-climb to eliminate hard violations by local flips.
    # Recomputes hv each probe (O(m)); sufficient for skeleton scale and correctness.
    def hard_unsat_clauses():
        hs = []
        for idx, cl in enumerate(cnf.clauses):
            if getattr(cl, "is_hard", False):
                sat = False
                for lit in cl.lits:
                    v = abs(lit)
                    val = assign[v]
                    if (val == 1 and lit > 0) or (val == 0 and lit < 0):
                        sat = True
                        break
                if not sat:
                    hs.append((idx, cl))
        return hs

    def hard_violations_count():
        return len(hard_unsat_clauses())

    hv = hard_violations_count()
    iters = 0
    while hv > 0 and iters < max_iters:
        iters += 1
        candidates = set()
        for _, cl in hard_unsat_clauses():
            for lit in cl.lits:
                candidates.add(abs(lit))
        best_delta = 0
        best_v = None
        baseline = hv
        for v in candidates:
            assign[v] ^= 1
            new_hv = hard_violations_count()
            delta = baseline - new_hv
            assign[v] ^= 1
            if new_hv < hv and delta > best_delta:
                best_delta = delta
                best_v = v
        if best_v is None:
            # No improving single flip; random kick on a candidate
            v = random.choice(list(candidates))
            assign[v] ^= 1
        else:
            assign[best_v] ^= 1
        hv = hard_violations_count()
    return assign


def short_polish_with_walksat(cnf, cfg_ls: dict, seed: int, start_assignment: Assignment) -> Tuple[Assignment, dict]:
    """Call LS with a starting assignment if supported; otherwise, return start_assignment and empty stats.
    Expected LS return dict may include keys like: best_assignment, hard_violations, soft_cost, flips.
    """
    try:
        from src.sat import walksat as _ls
    except Exception:
        # Fallback: LS not available
        return start_assignment, {"note": "walksat not importable; returned unpolished child"}

    try:
        res = _ls.run_satlike(cnf, cfg_ls, seed=seed, start_assignment=start_assignment)
        assn = res.get("best_assignment", start_assignment)
        return assn, res
    except TypeError:
        # Older run_satlike signature without seeding; just run and ignore seed
        res = _ls.run_satlike(cnf, cfg_ls, seed=seed)
        assn = res.get("best_assignment", start_assignment)
        return assn, {**res, "note": "LS called without start_assignment support"}


# =========================
# File: src/evo/memetic.py
# =========================
from __future__ import annotations
from typing import Dict, Any
import random

from .population import Population, EAConfig, Individual, _eval_with_cnf
from .operators import weighted_uniform_crossover, mutate_bitflip, greedy_hard_repair, short_polish_with_walksat


def memetic_run(cnf, ea_cfg: EAConfig, ls_cfg: Dict[str, Any], rng: random.Random | None = None, frozen_vars=None) -> Dict[str, Any]:
    rng = rng or random.Random(ea_cfg.seed)
    pop = Population(cnf, ea_cfg, rng=rng, frozen_vars=frozen_vars)

    history = []
    for g in range(ea_cfg.gens):
        # Elitism
        pop.individuals.sort(key=lambda ind: ind.eval.fitness, reverse=True)
        next_gen: list[Individual] = pop.individuals[: ea_cfg.elite]

        # Fill the rest
        while len(next_gen) < pop.cfg.pop_size:
            pa = pop.select_tournament()
            pb = pop.select_tournament()
            if pb is pa:
                pb = pop.select_tournament()
            # Ensure pa is fitter for bias semantics
            if pb.eval.fitness > pa.eval.fitness:
                pa, pb = pb, pa
            child_genes = weighted_uniform_crossover(pa, pb, rng, pop.cfg.crossover_bias, pop.frozen_vars)
            mutate_bitflip(child_genes, rng, pop.cfg.pmutate, pop.frozen_vars)
            # Hard repair, then short LS polish
            child_genes = greedy_hard_repair(cnf, child_genes)
            polished, ls_stats = short_polish_with_walksat(
                cnf, {**ls_cfg, "flip_budget": ls_cfg.get("flip_budget", 0) or pop.cfg.ls_polish_flips, "time_limit_s": min(ls_cfg.get("time_limit_s", 1.0), 2.0)},
                seed=rng.randrange(1 << 30),
                start_assignment=child_genes,
            )
            child = Individual(polished, _eval_with_cnf(cnf, polished))
            next_gen.append(child)
        pop.individuals = next_gen
        pop.individuals.sort(key=lambda ind: ind.eval.fitness, reverse=True)
        hist = {"gen": g, **pop.stats()}
        history.append(hist)

    best = pop.best()
    return {
        "best_assignment": best.genes,
        "best_soft_unsat": best.eval.soft_unsat_weight,
        "best_hard_violations": best.eval.hard_violations,
        "best_fitness": best.eval.fitness,
        "history": history,
    }


# =========================
# PATCH: src/sat/walksat.py (enable seeding)
# Apply roughly; adjust to your local signatures if different
# =========================
"""
--- a/src/sat/walksat.py
+++ b/src/sat/walksat.py
@@
-def run_satlike(cnf, cfg: dict, seed: int = 1):
-    """"""Existing signature""""""
-    rng = random.Random(seed)
-    state = State(cnf, rng, cfg)
+def run_satlike(cnf, cfg: dict, seed: int = 1, start_assignment=None):
+    """"""Run SATLike-ish local search. If start_assignment is provided, seed the search with it.
+    """"""
+    rng = random.Random(seed)
+    state = State(cnf, rng, cfg)
+    if start_assignment is not None:
+        try:
+            # If State exposes a setter / direct field, prefer the API you already have
+            if hasattr(state, "set_assignment"):
+                state.set_assignment(start_assignment)
+            else:
+                state.assignment = list(start_assignment)
+                # Recompute true_cnt and any incremental caches
+                if hasattr(state, "recompute_caches"):
+                    state.recompute_caches()
+        except Exception as e:
+            # Fallback: ignore bad seed but continue
+            pass
     # ... rest of your existing function ...
     # make sure to return a dict with at least 'best_assignment', 'hard_violations', 'soft_cost'
"""


# =========================
# PATCH: src/cli/solve.py (wire --ea)
# =========================
"""
--- a/src/cli/solve.py
+++ b/src/cli/solve.py
@@
 parser.add_argument("--ea", action="store_true", help="run memetic EA wrapper")
@@
 if args.ea:
     from src.evo.memetic import memetic_run, EAConfig
     # Split EA vs LS config blocks if you keep YAML under keys {ea: {}, ls: {}}
     ea_cfg_d = cfg.get("ea", {})
     ls_cfg_d = cfg.get("ls", cfg)
     eacfg = EAConfig(
         pop_size=ea_cfg_d.get("pop_size", 40),
         gens=ea_cfg_d.get("gens", 60),
         tournament_k=ea_cfg_d.get("tournament_k", 4),
         elite=ea_cfg_d.get("elite", 2),
         pmutate=ea_cfg_d.get("pmutate", 0.02),
         crossover_bias=ea_cfg_d.get("crossover_bias", 0.7),
         ls_polish_flips=ea_cfg_d.get("ls_polish_flips", 2000),
         seed=args.seed,
     )
     res = memetic_run(cnf, eacfg, ls_cfg_d)
     print(json.dumps(res))
     return
 else:
     # existing LS flow
     pass
"""


# =========================
# Example YAML fragment (configs/default.yaml)
# =========================
"""
ea:
  pop_size: 40
  gens: 60
  tournament_k: 4
  elite: 2
  pmutate: 0.02
  crossover_bias: 0.7
  ls_polish_flips: 2000
ls:
  flip_budget: 300000
  time_limit_s: 20
  noise: 0.10
  tabu_length: 12
  dynamic_weights: { bump: 0.5, smooth_every: 3000, rho: 0.5 }
  hard_repair_budget: 12000
  hard_clause_policy: "forbid"
"""


# =========================
# Minimal smoke test (toy) – replace paths as needed
# =========================
"""
# Plain LS
# PYTHONUNBUFFERED=1 python -u -m src.cli.solve_batch \
#   --folder data/toy --config configs/default.yaml \
#   --time_limit_s 20 --seed 1 --out experiments/toy_results.csv

# EA single run (JSON to stdout)
# python -m src.cli.solve --instance data/toy/sample.wcnf --config configs/default.yaml --ea --seed 1
"""
