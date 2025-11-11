# src/cli/run_ea.py
from __future__ import annotations

import argparse, json, os, sys, time, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# --- Flexible imports so you can run as a module or a script ---
try:
    from evo.memetic import run_memetic
except Exception:
    # Add parent-of-src to path when invoked directly
    this = os.path.abspath(__file__)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(this), "..")))
    from evo.memetic import run_memetic  # type: ignore

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # graceful fallback to JSON or defaults


# -------- Minimal WCNF fallback loader (if you don't have one) --------
@dataclass
class Clause:
    lits: List[int]
    weight: float
    is_hard: bool

@dataclass
class WCNF:
    n_vars: int
    clauses: List[Clause]

def _parse_wcnf(path: str) -> WCNF:
    """
    Simple WCNF parser: expects 'p wcnf n m top' and then 'w lit ... 0' lines.
    A clause is 'hard' if weight >= top. Comments ('c ...') ignored.
    """
    top: Optional[float] = None
    n_vars = 0
    clauses: List[Clause] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("c"):
                continue
            if s.startswith("p"):
                # e.g., "p wcnf 3 5 10" or "p wcnf 3 5"
                parts = s.split()
                assert parts[1].lower() == "wcnf", "Only WCNF is supported"
                n_vars = int(parts[2])
                if len(parts) >= 5:
                    try:
                        top = float(parts[4])
                    except Exception:
                        top = None
                continue
            # clause line: weight then literals terminated by 0
            toks = s.split()
            if not toks:
                continue
            w = float(toks[0])
            lits = [int(x) for x in toks[1:] if x != "0"]
            # If top is unknown, treat no clause as hard (soft-only)
            is_hard = (top is not None and w >= top)
            clauses.append(Clause(lits=lits, weight=w, is_hard=is_hard))

    if n_vars == 0:
        # derive from max literal if header missing
        maxv = 0
        for cl in clauses:
            for lit in cl.lits:
                if abs(lit) > maxv:
                    maxv = abs(lit)
        n_vars = maxv
    return WCNF(n_vars=n_vars, clauses=clauses)


# -------- Config helpers --------
def _deep_set(d: Dict[str, Any], dotted_key: str, val: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = val

def _parse_kv_override(s: str) -> tuple[str, Any]:
    """
    Parse "a.b.c=123" -> ("a.b.c", 123) with JSON-ish literal parsing.
    """
    if "=" not in s:
        raise ValueError(f"Expected KEY=VALUE, got: {s}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    # Try JSON literal for numbers/booleans/null/arrays/objects
    try:
        parsed = json.loads(v)
    except Exception:
        parsed = v  # fallback to raw string
    return k, parsed

def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Try YAML first if available, else JSON
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception as e:
        raise SystemExit(f"Failed to parse config {path}: {e}")


# -------- Pretty printing --------
def _human_summary(res: Dict[str, Any]) -> str:
    ms = 1000.0 * float(res.get("elapsed_sec", 0.0))
    soft = res.get("best_soft_weight", 0.0)
    hv = res.get("hard_violations", 0)
    flips = res.get("total_flips", 0)
    fps = res.get("flips_per_sec", 0.0)
    meta = res.get("meta", {})
    gens = meta.get("ea_generations", "?")
    kids = meta.get("children", 0)
    return (f"[EA] soft={soft:.6f} hard_viol={hv} gens={gens} children={kids} "
            f"time={ms:.1f}ms flips={flips} ({fps:.0f}/s)")


# -------- Main CLI --------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ea",
        description="Run memetic EA on a WCNF and print JSON results.",
    )
    p.add_argument("wcnf", help="Path to .wcnf file")
    p.add_argument("-c", "--cfg", help="YAML/JSON config (ea/ls/time limits, etc.)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    p.add_argument("--override", "-D", action="append", default=[],
                   help="Override config key: -D ea.pop_size=80 -D ls.time_limit_s=0.05")
    p.add_argument("--use-internal-parser", action="store_true",
                   help="Force using built-in WCNF parser (ignore project loader).")
    p.add_argument("--out-json", help="Write result JSON to a file instead of stdout")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress human summary on stderr")

    args = p.parse_args(argv)

    # Load WCNF: use project's loader if available, else built-in
    wcnf = None
    if not args.use_internal_parser:
        try:
            # If you have your own loader, put it here:
            # from sat.cnf import load_wcnf
            # wcnf = load_wcnf(args.wcnf)
            pass
        except Exception:
            wcnf = None
    if wcnf is None:
        wcnf = _parse_wcnf(args.wcnf)

    # Load config and apply overrides
    cfg: Dict[str, Any] = _load_cfg(args.cfg)
    cfg_io = cfg.get("io", {})
    # Ensure EA is on by default for CLI
    if "ea" not in cfg:
        cfg["ea"] = {}
    cfg["ea"].setdefault("enabled", True)
    # Apply -D overrides
    for ov in args.override:
        k, v = _parse_kv_override(ov)
        _deep_set(cfg, k, v)

    # Run EA
    t0 = time.time()
    res = run_memetic(wcnf, cfg, rng_seed=args.seed)
    # If your run_memetic starts returning best assignment, you can write it out here.
    def _format_out(template: str, wcnf_path: str) -> str:
        import os
        stem = os.path.splitext(os.path.basename(wcnf_path))[0]
        dir_ = os.path.dirname(wcnf_path) or "."
        return template.format(stem=stem, name=stem, dir=dir_)

    # Prefer CLI flags; fall back to config
    if not args.out_json and "out_json" in cfg_io:
        args.out_json = _format_out(str(cfg_io["out_json"]), args.wcnf)

    if not args.quiet and "quiet" in cfg_io:
        args.quiet = bool(cfg_io["quiet"])
        # Emit JSON
        if not args.quiet:
            print(_human_summary(res), file=sys.stderr)

    out_json = json.dumps(res, ensure_ascii=False)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
