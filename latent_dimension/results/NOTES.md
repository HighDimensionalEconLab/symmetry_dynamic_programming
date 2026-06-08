# Reference results — what each file is

Reference outputs from the runs discussed in `../README.md`. All training on a single CUDA GPU,
`p=1.5`, float32; targets/metrics defined in `../../generalized_mean_simple.py`. `test_rel_error` is
the mean absolute relative error on a held-out test set; tables in the README report the **median
over seeds**. Each grid script is resumable (append one row per run; re-running skips done combos).

## `latent_ladder_results.csv` — Experiment 1 (`latent_ladder.py`)
The latent-dimension ladder. Sweep: `config ∈ {const, mu, mu+sig, mu+sig+eps}` (→ latent dim
`d = 0,1,2,3`) × `L ∈ {0,1,2,3,4,8}` × `N ∈ {128,1024}`, 3 seeds. 1000 train / 3000 test points,
2-layer φ/ρ (width 128), `lr=0.002, batch=256, max_epochs=8000, max_time=300s`.
Columns: `config, d, N, L, run, test_rel_error, test_abs_error, train_loss, epochs_run, stopping_reason`.
Shows `L ≥ d`, the `L=0` constant baseline, and concentration with `N`.

## `d3_compression_results.csv` — Experiment 2 (`d3_compression.py`)
The `d=3` "compression" case with the **plain power-mean** target. Sweep: `mu+sig+eps`,
`L ∈ {1,2,3,4,8}` × `N ∈ {64,128,256,1024}`, 3 seeds. 3-layer φ/ρ (width 128),
`lr=0.002, batch=256, max_epochs=15000, max_time=600s`.
Columns: `d, N, L, run, test_rel_error, test_abs_error, train_loss, epochs_run, stopping_reason`.
`L=2` floors even though three parameters vary — ε is collinear with location/scale, so `d_eff=2`.

## `d3_skew_results.csv` — Experiment 3 (`d3_skew.py`)
The **skew-augmented** target `Y = power_mean + skew_weight·skewness`. Sweep:
`skew_weight ∈ {0.0, 0.04}` × `L ∈ {1,2,3,4,8}` × `N ∈ {64,128,256,1024}`, 3 seeds, `epsilon_alpha=1`
(uniform ε). Same nets/budget as Exp 2.
Columns add `skew_weight, epsilon_alpha`. NOTE: the `skew_weight=0.04` rows all hit `max_epochs`
(under-trained — the skew target trains slowly); the definitive read is the trajectory logs below.

## `trajectory_*.log` — Diagnostic trajectories (`skew_trajectory.py`)
Single `N=1024`, `L ∈ {2,3,4}`, `test_rel_error` printed every few thousand epochs. The three runs:
| log | detector | λ | ε draw | epochs |
|---|---|---|---|---|
| `trajectory_skewness_uniform.log` | standardized skewness | 0.04 | Uniform[−1,1] (α=1) | 50000 |
| `trajectory_semivariance_uniform.log` | signed semivariance | 0.14 | Uniform[−1,1] (α=1) | 50000 |
| `trajectory_semivariance_ushaped.log` | signed semivariance | 0.14 | U-shaped (α=0.5, corners) | 20000 |
Show: not under-training (plateaus); skewness ≡ semivariance; `L=2` capacity wall vs `L≥3`
optimization band; U-shaped ε sharpens the wall.

## `free_vars_results.csv` — Auxiliary broad sweep (`free_vars_experiment.py`)
The earlier, **wider** in-distribution grid (kept for breadth; superseded for the clean capacity
story by Experiments 1–3). Sweep: `config ∈ {mu, mu+sigma, mu+sigma+eps, mu+sigma+eps+delta}`
(`n_free = 1,2,3,4`) × `N ∈ {16,32,64,128,256,512,1024}` × `num_points ∈ {5,10,100,1000}` ×
`L ∈ {1,2,3,4,8}`, 7 seeds; 2000 test points, 2-layer φ/ρ, **short budget**
`lr=0.005, batch=256, max_epochs=2000, max_time=120s`.
Columns: `config, n_free, N, num_points, L, run, seed, train_data_seed, test_data_seed,
test_rel_error, test_abs_error, test_loss, train_loss, epochs_run, stopping_reason, train_time`.
CAVEAT: the short ep2000 budget **under-trains** the harder cells (many hit `max_time`/`max_epochs`),
so absolute levels for large `N` / many free vars are optimization-limited, not capacity floors.
`free_vars_run.log` is the corresponding progress log.
