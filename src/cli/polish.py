# src/cli/polish.py
from __future__ import annotations

import argparse
import os
import sys
import time
import random

# Make "python -m src.cli.polish" work
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Project imports ---
try:
    # Use the lightweight memetic polish if present; otherwise fall back to run_satlike
    from sat.walksat import walksat_polish  # type: ignore
    _USE_POLISH = True
except Exception:
    from sat.walksat import run_satlike as _run_satlike  # type: ignore
    _USE_POLISH = False

from sat.cnf import WCNF  # your loader


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run WalkSAT polish-only pass on a WCNF (for memetic EA)."
    )
    p.add_argument(
        "--path",
        default=os.path.join("data", "toy", "mini.wcnf"),
        help="Path to WCNF file (default: data/toy/mini.wcnf)",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    p.add_argument(
        "--max-flips",
        type=int,
        default=None,
        help="Flip budget (default: ~10*nvars clamped to [2000, 50000])",
    )
    p.add_argument(
        "--time-limit-s",
        type=float,
        default=0.05,
        help="Wall-clock cap for the polish step (default: 0.05s)",
    )
    p.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="Exploration probability inside polish (default: 0.10)",
    )
    p.add_argument(
        "--no-hard-safe",
        action="store_true",
        help="Allow breaking hard clauses even after feasibility (default: off)",
    )
    p.add_argument(
        "--print-assign",
        action="store_true",
        help="Print final assignment as a 0/1 bitstring (0-based, length=n_vars)",
    )
    args = p.parse_args()

    if not os.path.exists(args.path):
        print(f"[FATAL] WCNF file not found: {args.path}", file=sys.stderr)
        sys.exit(3)

    # Load instance
    cnf = WCNF.parse_dimacs(args.path)
    nvars = cnf.n_vars

    # Seed assignment (0-based list[bool], length = nvars)
    rng = random.Random(args.seed)
    start_assign = [bool(rng.getrandbits(1)) for _ in range(nvars)]

    t0 = time.time()
    if _USE_POLISH:
        res = walksat_polish(
            cnf,
            start_assign=start_assign,
            rng_seed=args.seed,
            max_flips=args.max_flips,
            time_limit_s=args.time_limit_s,
            noise=args.noise,
            hard_safe=not args.no_hard_safe,
        )
    else:
        # Fallback: use run_satlike as a small-budget "polish" (requires start_assign support)
        cfg = {
            "flip_budget": args.max_flips if args.max_flips is not None else max(2000, min(50000, 10 * nvars)),
            "time_limit_s": args.time_limit_s,
            "noise": args.noise,
            "hard_safe": (not args.no_hard_safe),
            "restart_after": 0,  # do not restart in polish mode
        }
        res = _run_satlike(cnf, cfg=cfg, rng_seed=args.seed, start_assign=start_assign)
    wall = time.time() - t0

    # CSV-style single line (compatible with your experiments files)
    # instance,seed,elapsed_sec,best_soft_weight,hard_violations,total_flips,flips_per_sec
    instance = os.path.basename(args.path)
    print(
        f"{instance},{args.seed},{res['elapsed_sec']:.6f},{res['best_soft_weight']:.6f},"
        f"{res['hard_violations']},{res['total_flips']},{res['flips_per_sec']:.6f}"
    )

    if args.print_assign:
        bits = "".join("1" if b else "0" for b in res["final_assign"])
        print(f"# final_assign={bits}")

    # Human-friendly summary
    print(
        f"[OK] hard={res['hard_violations']} soft={res['best_soft_weight']:.6f} "
        f"flips={res['total_flips']} ({res['flips_per_sec']:.0f}/s) wall={wall:.3f}s"
    )


if __name__ == "__main__":
    main()
