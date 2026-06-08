"""Experiment 3 -- a genuine d=3 via a skew-augmented target.

Target:  Y = (mean|x|^p)^(1/p) + skew_weight * standardized_sample_skewness(x).

The plain power-mean collapses (mu,sigma,eps) to ~2 effective dimensions (eps collinear with mu; see
d3_compression.py / collinearity_mc.py).  Adding skew_weight * (standardized sample skewness) injects
a location/scale-INVARIANT 3rd-moment term that is a pure function of eps, so the target genuinely
depends on a 3rd latent coordinate -- with mu KEPT in [1,3].

skew_weight is a field on DataSettings (added to generalized_mean_simple.py); skew_weight=0 reproduces
the pure power-mean exactly.  Sweeping skew_weight in {0.0, 0.04}:
  0.00 -> compression baseline: L>=2 floors (== d3_compression.py), L=1 walls.
  0.04 -> L=1 walls; L=2 walls higher; L>=3 best -- a capacity wall at L=2 with mu fixed.
NOTE: the skew target trains much slower than the power-mean; this grid (ep15000) UNDER-TRAINS it.
The capacity-vs-optimization reading comes from skew_trajectory.py (longer, with periodic eval).

Run:  uv run python latent_dimension/d3_skew.py            (resumes from the CSV)
"""

import csv
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generalized_mean_simple import (  # noqa: E402
    DataSettings,
    HCSettings,
    OptimizerSettings,
    generalized_mean_simple,
)

RANGES = dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(-1.0, 1.0), delta=(1.0, 1.0))
SKEW_WEIGHTS = [0.0, 0.04]          # 0 = compression baseline; 0.04 = genuine d=3
EPSILON_ALPHA = 1.0                 # Beta(alpha,alpha) for eps: 1.0 uniform, 0.5 U-shaped (corners)
L_VALUES = [1, 2, 3, 4, 8]
N_VALUES = [64, 128, 256, 1024]
N_SEEDS = 3
NUM_TRAIN = 1000
NUM_TEST = 3000
HC = dict(phi_layers=3, phi_hidden_dim=128, rho_layers=3, rho_hidden_dim=128)
OPT = dict(lr=0.002, batch_size=256, max_epochs=15000, stopping_threshold=1e-6,
           max_time=600.0, print_interval=0)

COLUMNS = ["skew_weight", "epsilon_alpha", "d", "N", "L", "run", "test_rel_error", "test_abs_error",
           "train_loss", "epochs_run", "stopping_reason"]
HERE = Path(__file__).parent
CSV = HERE / "d3_skew_results.csv"
D = sum(lo != hi for lo, hi in RANGES.values())


def run_one(sw, N, L, run):
    ds = DataSettings(
        num_train_points=NUM_TRAIN, num_test_points=NUM_TEST,
        train_data_seed=200 + run, test_data_seed=300 + run, skew_weight=sw,
        mu_min_train=RANGES["mu"][0], mu_max_train=RANGES["mu"][1],
        sigma_min_train=RANGES["sigma"][0], sigma_max_train=RANGES["sigma"][1],
        epsilon_min_train=RANGES["epsilon"][0], epsilon_max_train=RANGES["epsilon"][1],
        epsilon_alpha_train=EPSILON_ALPHA,
        delta_min_train=RANGES["delta"][0], delta_max_train=RANGES["delta"][1],
    )
    try:
        r = generalized_mean_simple(N=N, p=1.5, hc_set=HCSettings(L=L, **HC), data_set=ds,
                                    opt_set=OptimizerSettings(**OPT), seed=100 + run, use_gpu=True,
                                    verbose=False, output_file="/tmp/d3_skew_run.json")
    except Exception as exc:
        print(f"  !! failed sw={sw} N={N} L={L} run={run}: {exc}", flush=True)
        r = None
    row = {"skew_weight": sw, "epsilon_alpha": EPSILON_ALPHA, "d": D, "N": N, "L": L, "run": run}
    for k in ["test_rel_error", "test_abs_error", "train_loss", "epochs_run", "stopping_reason"]:
        row[k] = r[k] if r is not None else float("nan")
    return row


def done_keys():
    if not CSV.exists():
        return set()
    df = pd.read_csv(CSV)
    return {f"{r.skew_weight}|{r.N}|{r.L}|{r.run}" for r in df.itertuples()}


def report():
    df = pd.read_csv(CSV)
    pd.set_option("display.float_format", lambda x: f"{x:.2e}")
    pd.set_option("display.width", 200)
    for sw in SKEW_WEIGHTS:
        sub = df[df.skew_weight == sw]
        if sub.empty:
            continue
        piv = sub.groupby(["L", "N"])["test_rel_error"].median().reset_index().pivot_table(
            index="L", columns="N", values="test_rel_error")
        conv = sub.groupby(["L", "N"]).apply(
            lambda g: (g.stopping_reason == "stopping_threshold").sum(), include_groups=False
        ).reset_index(name="c").pivot_table(index="L", columns="N", values="c")
        tag = "compression (eps redundant)" if sw == 0.0 else "genuine d=3 (eps -> skew)"
        print(f"\n===== skew_weight={sw}  [{tag}]  d={D}, 3-layer phi/rho, ep{OPT['max_epochs']}/"
              f"{int(OPT['max_time'])}s  median test_rel_error vs N [conv/{N_SEEDS}] =====", flush=True)
        out = piv.copy()
        for col in piv.columns:
            out[col] = [f"{piv.loc[i, col]:.2e} ({int(conv.loc[i, col])}/{N_SEEDS})" for i in piv.index]
        print(out.to_string(), flush=True)


def main():
    done = done_keys()
    combos = list(itertools.product(SKEW_WEIGHTS, N_VALUES, L_VALUES))
    print(f"grid: {len(combos)} (skew_weight,N,L) x {N_SEEDS} seeds = {len(combos) * N_SEEDS} runs "
          f"({len(done)} done)", flush=True)
    new = not CSV.exists()
    start = time.perf_counter()
    with open(CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for sw, N, L in combos:
            for run in range(N_SEEDS):
                if f"{sw}|{N}|{L}|{run}" in done:
                    continue
                row = run_one(sw, N, L, run)
                writer.writerow(row)
                f.flush()
            print(f"  sw={sw} N={N} L={L} done  ({time.perf_counter() - start:.0f}s)", flush=True)
    report()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
