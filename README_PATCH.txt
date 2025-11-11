This zip contains the SATLike upgrade patch files for your memetic MaxSAT project.

Files:
- src/sat/state.py           (new)  -- state container with O(occ(v)) updates, dyn weights, tabu, restarts, Δhard helpers
- src/sat/walksat.py         (upd)  -- SATLike with dynamic weights, tabu+aspiration, noise adapt, hard-first repair, restarts
- src/bench/harness.py       (new)  -- minimal batch harness that writes CSV (loads WCNF.parse_dimacs)
- src/cli/solve_batch.py     (new)  -- CLI for batch solving (supports --time_limit_s and ls: mapping)

Usage (inside your repo, after merging files):
    conda activate maxsat
    python -m src.cli.solve --cnf data/toy/mini.wcnf --config configs/default.yaml

Batch:
    python -m src.cli.solve_batch --folder data/toy --config configs/default.yaml --seed 1 --out experiments/toy_results_tmp.csv



python -m src.cli.polish --path data/toy/mini.wcnf --print-assign
# or tweak budgets:
python -m src.cli.polish --path data/toy/mini.wcnf --seed 42 --time-limit-s 0.02 --max-flips 500 --print-assign



# As a module (recommended so imports resolve):
python -m src.cli.run_ea path/to/instance.wcnf -c cfg.yaml -D ea.pop_size=80 -D ls.time_limit_s=0.05 --seed 42

# Or directly (uses flexible import and built-in WCNF parser):
python src/cli/run_ea.py path/to/instance.wcnf --cfg cfg.json --seed 7 --out-json run.json


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m src.cli.solve_batch --folder data/toy --config configs/default.yaml --seed 1 --out experiments/toy_results_tmp.csv

Knobs you may want:
    time_limit_s: 20
    max_flips: 500000
    restart_after: 20000
    dyn_bump: 1.0
    smooth_every: 3000
    rho: 0.5
    tabu_length: 12
    noise: 0.2
    noise_min/noise_max: 0.05/0.45
    hard_repair_budget: 12000
