"""Experiment 2 -- the d=3 "compression" case (deeper nets, more N, longer training).

The three-parameter DGP mu+sig+eps with the plain power-mean target Y = (mean|x|^p)^(1/p).  With mu
kept in [1,3] (X stays positive) the target is a smooth low-moment functional, so eps's effect on Y
is almost entirely a location shift, COLLINEAR with mu (see collinearity_mc.py).  The three varying
parameters therefore collapse to ~2 effective dimensions (location, scale): L=2 already FLOORS, even
though three parameters vary.  This is latent compression -- L tracks d_eff (the sufficient-statistic
dimension), not the raw parameter count.  Deeper phi/rho (3 layers), N in {64,128,256,1024}, and a
larger optimizer budget so the L>=2 plateau is a capacity result, not under-training.

Run:  uv run python latent_dimension/d3_compression.py            (resumes from the CSV)
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
L_VALUES = [1, 2, 3, 4, 8]
N_VALUES = [64, 128, 256, 1024]
N_SEEDS = 3
NUM_TRAIN = 1000
NUM_TEST = 3000
HC = dict(phi_layers=3, phi_hidden_dim=128, rho_layers=3, rho_hidden_dim=128)
OPT = dict(lr=0.002, batch_size=256, max_epochs=15000, stopping_threshold=1e-6,
           max_time=600.0, print_interval=0)

COLUMNS = ["d", "N", "L", "run", "test_rel_error", "test_abs_error",
           "train_loss", "epochs_run", "stopping_reason"]
HERE = Path(__file__).parent
CSV = HERE / "d3_compression_results.csv"
D = sum(lo != hi for lo, hi in RANGES.values())


def run_one(N, L, run):
    ds = DataSettings(
        num_train_points=NUM_TRAIN, num_test_points=NUM_TEST,
        train_data_seed=200 + run, test_data_seed=300 + run,
        mu_min_train=RANGES["mu"][0], mu_max_train=RANGES["mu"][1],
        sigma_min_train=RANGES["sigma"][0], sigma_max_train=RANGES["sigma"][1],
        epsilon_min_train=RANGES["epsilon"][0], epsilon_max_train=RANGES["epsilon"][1],
        delta_min_train=RANGES["delta"][0], delta_max_train=RANGES["delta"][1],
    )
    try:
        r = generalized_mean_simple(N=N, p=1.5, hc_set=HCSettings(L=L, **HC), data_set=ds,
                                    opt_set=OptimizerSettings(**OPT), seed=100 + run, use_gpu=True,
                                    verbose=False, output_file="/tmp/d3_compression_run.json")
    except Exception as exc:
        print(f"  !! failed N={N} L={L} run={run}: {exc}", flush=True)
        r = None
    row = {"d": D, "N": N, "L": L, "run": run}
    for k in ["test_rel_error", "test_abs_error", "train_loss", "epochs_run", "stopping_reason"]:
        row[k] = r[k] if r is not None else float("nan")
    return row


def done_keys():
    if not CSV.exists():
        return set()
    df = pd.read_csv(CSV)
    return {f"{r.N}|{r.L}|{r.run}" for r in df.itertuples()}


def report():
    df = pd.read_csv(CSV)
    pd.set_option("display.float_format", lambda x: f"{x:.2e}")
    pd.set_option("display.width", 200)
    piv = df.groupby(["L", "N"])["test_rel_error"].median().reset_index().pivot_table(
        index="L", columns="N", values="test_rel_error")
    conv = df.groupby(["L", "N"]).apply(
        lambda g: (g.stopping_reason == "stopping_threshold").sum(), include_groups=False
    ).reset_index(name="c").pivot_table(index="L", columns="N", values="c")
    print(f"\n===== mu+sig+eps (d={D})  3-layer phi/rho, ep{OPT['max_epochs']}/"
          f"{int(OPT['max_time'])}s  median test_rel_error vs N [conv/{N_SEEDS}] =====", flush=True)
    out = piv.copy()
    for col in piv.columns:
        out[col] = [f"{piv.loc[i, col]:.2e} ({int(conv.loc[i, col])}/{N_SEEDS})" for i in piv.index]
    print(out.to_string(), flush=True)


def main():
    done = done_keys()
    combos = list(itertools.product(N_VALUES, L_VALUES))
    print(f"grid: {len(combos)} (N,L) x {N_SEEDS} seeds = {len(combos) * N_SEEDS} runs "
          f"({len(done)} done)", flush=True)
    new = not CSV.exists()
    start = time.perf_counter()
    with open(CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for N, L in combos:
            for run in range(N_SEEDS):
                if f"{N}|{L}|{run}" in done:
                    continue
                row = run_one(N, L, run)
                writer.writerow(row)
                f.flush()
            print(f"  N={N} L={L} done  ({time.perf_counter() - start:.0f}s)", flush=True)
    report()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
