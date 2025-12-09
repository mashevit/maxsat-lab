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
python -m src.cli.run_ea data/toy/mini.wcnf -c configs/cfg.yaml -D ea.pop_size=80 -D ls.time_limit_s=0.05 --seed 42
python -m src.cli.run_ea data/toy/uf50-05.cnf -c configs/cfg2.yaml -D ea.pop_size=80 -D ls.time_limit_s=0.05 --seed 42
# Or directly (uses flexible import and built-in WCNF parser):
python src/cli/run_ea.py data/toy/mini.wcnf --cfg cfg.json --seed 7 --out-json run.json


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



python -m src.cli.run_experiment \
    --bench_dir data/toy \
    -c configs/cfg2.yaml \
    --seeds 1 2 3 4 5 \
    --time_limit 30 \
    --out_csv results/dev_small_base.csv \
    --config_id cfg2_pop80


python -m src.cli.run_experiment \
    --bench_dir data/toy \
    -c configs/cfg2.yaml \
    --seeds 1 2 3 4 5 \
    --time_limit 30 \
    --out_csv results/dev_small_cfg2_pop80.csv \
    --config_id cfg2_pop80 \
    -D ea.pop_size=80 \
    -D ls.time_limit_s=0.05


If your solver actually expects the per-run time limit under e.g. ea.time_limit_s or ls.time_limit_s, just tweak the _deep_set(cfg, "time_limit_s", ...) line to the dotted key you’re using.



python -m src.cli.run_experiment1 \
  --bench_dir data/dev_small \
  -c configs/cfg2.yaml \
  --seeds 1 2 3 4 5 \
  --total_time_s 300 \
  --out_csv results/dev_small_cfg2_5min.csv \
  --config_id cfg2_5min \
  -D ea.pop_size=80


  אם אתה רוצה במקום זה זמן קבוע פר ריצה (למשל 30 שניות לכל (instance, seed)), תשתמש ב־--per_run_time_s 30 ותשאיר את --total_time_s ריק.

אם ה־time limit שלך צריך ללכת למפתח אחר (למשל time_limit_s או ls.time_limit_s), פשוט תשנה בראש הקובץ:

TIME_LIMIT_KEY = "ea.time_limit_s"

python -m src.cli.run_experiment \
  --bench_dir data/exp0 \
  -c configs/cfg2.yaml \
  --seeds 1 2 3 \
  --total_time_s 300 \
  --out_csv results/dev_small_cfg2_5min.csv \
  --config_id cfg2_5min \
  -D ea.pop_size=80

python -m src.cli.run_experiment1 \
  --bench_dir data/exp0 \
  -c configs/cfg2.yaml \
  --seeds 1 2 3 \
  --out_csv results/dev_small_cfg2_5min.csv \
  --config_id configs/cfg2.yaml \


  python -m src.cli.run_experiment1   --bench_dir data/exp2   -c configs/cfg250.yaml   --seeds 1   --out_csv results/dev_250h.csv   --config_id configs/cfg250.yaml 

  python -m src.cli.batch_opt_rc2 data/exp2 -o results/dev_250h.csv

python -m src.cli.run_experiment \
  --bench_dir data/exp2 \
  -c configs/cfg250.yaml \
  --seeds 1 \
  --total_time_s 300 \
  --out_csv results/dev_250h.csv \
  --config_id cfg2_5min \
  -D ea.pop_size=50

  python -m src.cli.run_pipeline_opt_vs_memetic \
  --bench_dir data/exp0 \
  --config configs/cfg2.yaml \
  --seeds 1 2 3 \
  --rc2_csv results/dev_small_rc2_1.csv \
  --memetic_csv results/dev_small_cfg2_5min.csv \
  --out_csv results/dev_small_cfg2_5min_with_opt.csv \
  --config_id cfg2


   python -m src.cli.run_pipeline_opt_vs_memetic \
  --bench_dir data/exp1 \
  --config configs/cfg250.yaml \
  --seeds 1 2 \
  --rc2_csv results/dev_250.csv \
  --memetic_csv results/dev_250_cfg250.csv \
  --out_csv results/dev_250_with_opt.csv \
  --config_id cfg250