#!/usr/bin/env bash
#
# Recreate every table/experiment in latent_dimension/README.md.
#
# Requires a CUDA (or Apple MPS) GPU and the repo .venv (`uv sync`).  Run from the REPO ROOT:
#     bash latent_dimension/run_all.sh
#
# The grid scripts (latent_ladder / d3_compression / d3_skew) are RESUMABLE: they append to a CSV
# next to the script and skip combos already present, so a killed run can just be re-launched.
# Heavy runtimes (single CUDA GPU) are noted; comment out what you do not need.  Per-experiment
# knobs (grids, layer counts, optimizer budget, eps Beta alpha) live as constants at the top of each
# script; the diagnostic trajectory exposes them as CLI flags (every variant we used is listed below).
#
set -euo pipefail
cd "$(dirname "$0")/.."        # repo root, so `generalized_mean_simple` imports

# =====================================================================================
# Monte-Carlo motivation (no training; ~1-2 min each on GPU)
# =====================================================================================
# Why a skew term is needed: eps's effect on Y is collinear with location (mu) and scale (sigma);
# its location/scale-orthogonal content in the experiment regime is ~3e-4 (at the learned floor).
uv run python latent_dimension/collinearity_mc.py

# Which detector: standardized skewness and the signed-semivariance ratio are clean (odd, monotone,
# location/scale-invariant) and interchangeable; Gini is EVEN in eps (skew-blind); quantile ratios
# are muddied with variance.
uv run python latent_dimension/detector_mc.py

# =====================================================================================
# Experiment 1 -- latent ladder:  d in {0,1,2,3} x L in {0,1,2,3,4,8} x N in {128,1024}
#   ~1.5-2 h.  Knobs: CONFIGS, L_VALUES, N_VALUES, N_SEEDS, OPT (lr0.002/bs256/ep8000/300s).
# =====================================================================================
uv run python latent_dimension/latent_ladder.py

# =====================================================================================
# Experiment 2 -- d=3 compression (deeper nets):  mu+sig+eps, L in {1,2,3,4,8}, N in {64,128,256,1024}
#   ~2 h.  3-layer phi/rho, ep15000/600s.  Shows L=2 floors even though 3 parameters vary.
# =====================================================================================
uv run python latent_dimension/d3_compression.py

# =====================================================================================
# Experiment 3 -- d=3 skew-augmented grid:  skew_weight in {0.0, 0.04}, same L/N as Exp 2
#   ~5 h.  NOTE: skew_weight=0.04 UNDER-TRAINS at ep15000 (all runs hit max_epochs); use the
#   trajectories below for the definitive read.  Edit EPSILON_ALPHA at the top (1.0 uniform / 0.5
#   U-shaped) to change the eps draw.
# =====================================================================================
uv run python latent_dimension/d3_skew.py

# =====================================================================================
# Diagnostic trajectories -- single N=1024, L in {2,3,4}, test_rel printed periodically.
# These are the runs behind the README "trajectory" tables and give the capacity-vs-optimization
# read.  ALL detector / eps-alpha / lambda / epoch-budget options used this session:
# =====================================================================================
# (a) literal skewness, uniform eps  (soft staircase; plateaus ~10x above floor; ~1.7 h)
uv run python latent_dimension/skew_trajectory.py \
    --detector skewness --lam 0.04 --eps-alpha 1.0 --N 1024 --L 2 3 4 --max-epochs 50000 --eval-interval 5000

# (b) signed semivariance, uniform eps  (statistically identical to (a) -> detector form not the bottleneck)
uv run python latent_dimension/skew_trajectory.py \
    --detector semivariance --lam 0.14 --eps-alpha 1.0 --N 1024 --L 2 3 4 --max-epochs 50000 --eval-interval 5000

# (c) signed semivariance, U-shaped eps (alpha=0.5, mass at the +/-1 corners)
#     -> sharper L=2 capacity wall (~2x the L>=3 band); 20k budget (it plateaus by ~15-20k); ~35 min
uv run python latent_dimension/skew_trajectory.py \
    --detector semivariance --lam 0.14 --eps-alpha 0.5 --N 1024 --L 2 3 4 --max-epochs 20000 --eval-interval 2500

# Other knobs the trajectory accepts: --eps-min/--eps-max, --lr, --batch-size, --phi-layers,
# --rho-layers, --hidden, --num-train, --num-test, --p, --seed.  (`--help` lists them all.)

echo "all experiments complete"
