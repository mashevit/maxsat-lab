from __future__ import annotations
from typing import Any, Dict, List


def extract_clause_examples(
    *,
    wcnf: Any,
    assign01: List[bool],
    hard_clause_idxs: List[int],
    max_k: int,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts like:
      {"idx": 12, "lits": [1, -3, 7]}
    for violated hard clauses (or any provided idxs).
    Adjust if your WCNF/Clause structure differs.
    """
    examples = []
    clauses = getattr(wcnf, "clauses", [])
    for idx in hard_clause_idxs[:max_k]:
        if 0 <= idx < len(clauses):
            cl = clauses[idx]
            lits = list(getattr(cl, "lits", []))
            examples.append({"idx": idx, "lits": lits})
    return examples


def build_prompt(
    *,
    n_vars: int,
    violated_hard: int,
    clause_examples: List[Dict[str, Any]],
    rng_seed: int,
    extra: Dict[str, Any],
) -> str:
    """
    We force a tiny, machine-parsable JSON response.
    """
    return f"""
You are helping a memetic MaxSAT solver.
We have a candidate assignment over {n_vars} Boolean vars (1-based).
It currently violates {violated_hard} HARD clauses.

Below are examples of violated hard clauses (each is a disjunction of literals).
Literal k means x_k is True, literal -k means x_k is False.
Examples:
{clause_examples}

Extra info (optional):
{extra}

Task:
Propose a SMALL edit (at most ~10 vars) that is likely to reduce hard violations.
Return ONLY valid JSON, one object, in this schema:

{{
  "flip": [3, 17],          // flip these variable indices
  "set_true": [5],          // optionally force vars True
  "set_false": [9],         // optionally force vars False
  "note": "short reason"
}}

No markdown. No additional text.

Seed: {rng_seed}
""".strip()
