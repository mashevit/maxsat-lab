This zip contains the SATLike upgrade patch files for your memetic MaxSAT project.

Files:
- src/sat/state.py           (new)  -- state container with O(occ(v)) updates, dyn weights, tabu, restarts
- src/sat/walksat.py         (upd)  -- production SATLike: gains, dyn weights, tabu+aspiration, noise adapt, restarts, stats
- src/bench/harness.py       (new)  -- minimal batch harness that writes CSV
- src/cli/solve_batch.py     (new)  -- CLI for batch solving

Usage (inside your repo, after merging files):
    conda activate maxsat
    python -m src.cli.solve --cnf data/toy/mini.wcnf --config configs/default.yaml

Batch:
    python -m src.cli.solve_batch --folder data/toy --config configs/default.yaml --seed 1 --out experiments/toy_results.csv

Notes:
- Requires the parser to expose pos_occ/neg_occ and nvars, plus either (hard/soft) or (clauses,weights,top).
- All paths are Linux-style and suitable for WSL2.
