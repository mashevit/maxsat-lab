# src/cli/run_opt_rc2.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, List

# --- Try to import PySAT RC2 and WCNF ---------------------------------------


try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF, CNF
except ImportError as e:  # pragma: no cover - import error is handled at runtime
    print(
        "[run_opt_rc2] ERROR: could not import PySAT.\n"
        "Please install it with:\n"
        "    pip install python-sat[pblib,aiger]\n",
        file=sys.stderr,
    )
    raise


# --- Data structures --------------------------------------------------------


@dataclass
class RC2Result:
    """Container for RC2 solution and basic statistics."""

    instance: str
    status: str           # "OPT", "UNSAT", or "ERROR"
    time_s: float
    n_vars: int
    n_hard: int
    n_soft: int
    soft_total_weight: int
    opt_cost: Optional[int] = None           # sum of weights of UNSAT soft clauses
    best_soft_weight: Optional[int] = None   # soft_total_weight - opt_cost
    model: Optional[List[int]] = None        # RC2 model as list of signed ints (DIMACS)


# --- Core solver logic ------------------------------------------------------



import os


def cnf_to_all_soft_wcnf(path: str) -> WCNF:
    """
    Load a plain CNF and convert it to a WCNF where:
    - all original clauses are SOFT with weight 1
    - we add a single HARD tautological clause to satisfy RC2's 'hard part' expectation
    """
    cnf = CNF(from_file=path)

    if cnf.nv == 0:
        raise ValueError(f"CNF {path} has no variables")

    # Let PySAT do the standard "all soft, weight 1" conversion
    wcnf = cnf.weighted()  # all clauses -> soft, weight 1

    # Add a trivial HARD clause so there is at least one hard clause.
    # No weight argument => HARD.
    #wcnf.append([1, -1])

    return wcnf


def load_as_wcnf(path: str) -> WCNF:
    """
    Load an instance as a WCNF for RC2.

    - If the file is already in WCNF/MaxSAT format (header 'p wcnf'),
      use PySAT's WCNF loader directly.
    - If the file is plain CNF (header 'p cnf'), load with CNF and
      wrap it as an unweighted MaxSAT instance where all clauses are
      soft with weight 1.
    """
    # Quick sniff of the header
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                header = line.split()
                if len(header) >= 2 and header[1].lower() == "wcnf":
                    # Proper WCNF / MaxSAT file
                    return WCNF(from_file=path)
                elif len(header) >= 2 and header[1].lower() == "cnf":
                    return cnf_to_all_soft_wcnf(path)
                else:
                    # Unknown 'p' header, let WCNF try and fail loudly
                    break

    # Fallback: just let WCNF try; if it fails, caller will catch
    return WCNF(from_file=path)





import time
from pysat.examples.rc2 import RC2

def solve_instance(path: str, solver_name: str = "g3") -> RC2Result:
    start = time.time()

    status = "OK"
    n_vars = n_hard = n_soft = soft_total_weight = 0
    opt_cost = None
    best_soft_weight = None

    try:
        wcnf = cnf_to_all_soft_wcnf(path)

        n_vars = wcnf.nv
        n_hard = len(wcnf.hard)
        n_soft = len(wcnf.soft)
        soft_total_weight = sum(wcnf.wght)

        # Sanity checks before we touch RC2
        assert n_vars > 0
        assert n_soft > 0
        assert len(wcnf.wght) == n_soft

        # ---- RC2 call ----
        with RC2(wcnf) as rc2:
            rc2.compute()           # compute optimum
            model = rc2.model       # list[int] or None
            opt_cost = rc2.cost     # minimum sum of weights of UNSAT soft clauses

        # For this kind of WCNF (only a tautological hard clause),
        # RC2 should always return a model.
        if model is None:
            status = "UNSAT"
        else:
            status = "OPT"
            # Max soft weight satisfied = total soft weight - cost of violated ones
            if opt_cost is not None:
                best_soft_weight = soft_total_weight - opt_cost

    except AssertionError as e:
        status = f"ERROR: bad WCNF ({e})"
    except Exception as e:
        status = f"ERROR: RC2 failed ({e})"

    time_s = time.time() - start
    return RC2Result(
            instance=path,
            status=status,
            time_s=round(time_s, 6),
            n_vars=n_vars,
            n_hard=n_hard,
            n_soft=n_soft,
            soft_total_weight=soft_total_weight,
            opt_cost=opt_cost,
            best_soft_weight=best_soft_weight,
            model = model,
        )
    return {
        "instance": path,
        "status": status,
        "time_s": round(time_s, 6),
        "n_vars": n_vars,
        "n_hard": n_hard,
        "n_soft": n_soft,
        "soft_total_weight": soft_total_weight,
        "opt_cost": opt_cost,
        "best_soft_weight": best_soft_weight,
    }





def solve_with_rc2(path: str, solver_name: str = "g3") -> RC2Result:
    """
    Load a CNF/WCNF instance from `path` and solve it optimally using RC2.

    * If `path` is a plain CNF, all clauses are treated as hard (standard SAT).
    * If it is a WCNF, hard and soft clauses + weights are respected.

    Returns an RC2Result object with basic stats and the model.
    """
    abs_path = os.path.abspath(path)
    t0 = time.time()

    try:
        # WCNF can also parse plain CNF; in that case all clauses become hard.
        wcnf = cnf_to_all_soft_wcnf(abs_path)
    except Exception as e:
        t1 = time.time()
        return RC2Result(
            instance=abs_path,
            status=f"ERROR: load failed ({e})",
            time_s=t1 - t0,
            n_vars=0,
            n_hard=0,
            n_soft=0,
            soft_total_weight=0,
        )

    n_vars = wcnf.nv  # number of variables
    n_hard = len(wcnf.hard)
    n_soft = len(wcnf.soft)
    soft_total_weight = int(sum(wcnf.wght)) if wcnf.wght else 0
    start = time.time()
    try:
        with RC2(wcnf, solver=solver_name) as rc2:
            model = rc2.compute()  # optimal model or None
            cost = rc2.cost        # sum of weights of unsatisfied soft clauses
    except Exception as e:
        t1 = time.time()
        return RC2Result(
            instance=abs_path,
            status=f"ERROR: RC2 failed ({e})",
            time_s=t1 - t0,
            n_vars=n_vars,
            n_hard=n_hard,
            n_soft=n_soft,
            soft_total_weight=soft_total_weight,
        )

    t1 = time.time()

    # Standard RC2 semantics:
    #   * cost = sum of weights of UNSAT soft clauses.
    #   * For pure SAT (no soft), cost is usually 0.
    if model is None:
        status = "UNSAT"
        opt_cost = None
        best_soft_weight = None
    else:
        status = "OPT"
        opt_cost = int(cost)
        best_soft_weight = soft_total_weight - opt_cost if n_soft > 0 else None

    return RC2Result(
        instance=abs_path,
        status=status,
        time_s=t1 - t0,
        n_vars=n_vars,
        n_hard=n_hard,
        n_soft=n_soft,
        soft_total_weight=soft_total_weight,
        opt_cost=opt_cost,
        best_soft_weight=best_soft_weight,
        model=model,
    )


# --- I/O helpers ------------------------------------------------------------


def save_model_01(model: List[int], n_vars: int, out_path: str) -> None:
    """
    Save a DIMACS-style model as a 0/1 bitstring to `out_path`.

    * model: list of signed integers, e.g. [1, -2, 3, ...]
    * n_vars: total number of variables in the instance.
    * out_path: where to write the string, e.g. "0001101...".
    """
    # DIMACS models use variable indices 1..n_vars.
    # True  => positive literal present
    # False => negative literal present
    assignment = ["0"] * (n_vars + 1)  # index 0 unused

    for lit in model:
        v = abs(lit)
        if 1 <= v <= n_vars:
            assignment[v] = "1" if lit > 0 else "0"

    bitstring = "".join(assignment[1:])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(bitstring + "\n")


def print_human_readable(res: RC2Result, print_model: bool) -> None:
    """
    Print a human-friendly summary of the RC2 run.

    Optionally, print the full model as a list of integers.
    """
    print(f"Instance:         {res.instance}")
    print(f"Status:           {res.status}")
    print(f"Time (s):         {res.time_s:.4f}")
    print(f"#vars:            {res.n_vars}")
    print(f"#hard clauses:    {res.n_hard}")
    print(f"#soft clauses:    {res.n_soft}")
    print(f"Soft total weight:{res.soft_total_weight}")

    if res.status == "OPT":
        print(f"Optimal cost:     {res.opt_cost}")
        if res.best_soft_weight is not None:
            print(f"Best soft weight: {res.best_soft_weight}")

    if print_model and res.model is not None:
        print("Model (DIMACS literals):")
        print(" ".join(str(l) for l in res.model))


# --- CLI --------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a CNF/WCNF instance optimally using PySAT's RC2 MaxSAT solver.\n"
            "Example:\n"
            "  python -m src.cli.run_opt_rc2 data/dev_small/foo.cnf\n"
            "  python -m src.cli.run_opt_rc2 data/weights/bar.wcnf --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "instance",
        help="Path to the CNF/WCNF instance (DIMACS format).",
    )

    parser.add_argument(
        "--solver",
        default="g3",
        help="Underlying SAT solver name for RC2 (default: %(default)s).",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print a single JSON line with the result summary.",
    )

    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Print the full model as DIMACS literals to stdout (if found).",
    )

    parser.add_argument(
        "--model-out",
        metavar="PATH",
        help=(
            "Optional file path to save the model as a 0/1 bitstring "
            "(variables 1..n written in order)."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    res = solve_with_rc2(args.instance, solver_name=args.solver)

    # Human-readable summary
    print_human_readable(res, print_model=args.print_model)

    # Optional JSON line (easy to parse from scripts)
    if args.json:
        # Do not dump the full model into JSON by default; it can be huge.
        payload = asdict(res)
        if payload.get("model") is not None:
            payload["model"] = f"<{len(res.model)} literals>"  # short placeholder
        print("JSON_RESULT " + json.dumps(payload, ensure_ascii=False))

    # Optional model-as-01-string file
    if args.model_out and res.model is not None and res.n_vars > 0:
        save_model_01(res.model, res.n_vars, args.model_out)


if __name__ == "__main__":  # pragma: no cover
    main()
