"""Experiment 1 -- the latent-dimension ladder (with the L=0 constant baseline).

For each number of free DGP parameters (0,1,2,3 -> latent-state dimension d), sweep the DeepSet
latent width L INCLUDING L=0 (a constant predictor: the pooled feature is 0-dimensional, so the net
outputs a learned constant = the unconditional-mean baseline) and the set size N.

Shows: (a) L >= d is needed -- below it the error is walled off; (b) L=0 is the constant baseline in
every table and the *correct* rung for the 0-free-vars case; (c) larger N concentrates (lower error
once L is adequate). See ../latent_dimension/README.md for the resulting tables and discussion.

Run:  uv run python latent_dimension/latent_ladder.py            (resumes from the CSV)
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

# (name, ranges); a degenerate (min == max) parameter is held fixed. d = #varying params = latent dim.
CONFIGS = [
    ("const",      dict(mu=(2.0, 2.0), sigma=(0.3, 0.3), epsilon=(0.0, 0.0), delta=(1.0, 1.0))),
    ("mu",         dict(mu=(1.0, 3.0), sigma=(0.3, 0.3), epsilon=(0.0, 0.0), delta=(1.0, 1.0))),
    ("mu+sig",     dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(0.0, 0.0), delta=(1.0, 1.0))),
    ("mu+sig+eps", dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(-1.0, 1.0), delta=(1.0, 1.0))),
]
L_VALUES = [0, 1, 2, 3, 4, 8]      # 0 = constant baseline
N_VALUES = [128, 1024]
N_SEEDS = 3
NUM_TRAIN = 1000
NUM_TEST = 3000
OPT = dict(lr=0.002, batch_size=256, max_epochs=8000, stopping_threshold=1e-6,
           max_time=300.0, print_interval=0)

COLUMNS = ["config", "d", "N", "L", "run", "test_rel_error", "test_abs_error",
           "train_loss", "epochs_run", "stopping_reason"]
HERE = Path(__file__).parent
CSV = HERE / "latent_ladder_results.csv"


def latent_dim(ranges):
    return sum(lo != hi for lo, hi in ranges.values())


def run_one(name, ranges, N, L, run):
    ds = DataSettings(
        num_train_points=NUM_TRAIN, num_test_points=NUM_TEST,
        train_data_seed=200 + run, test_data_seed=300 + run,
        mu_min_train=ranges["mu"][0], mu_max_train=ranges["mu"][1],
        sigma_min_train=ranges["sigma"][0], sigma_max_train=ranges["sigma"][1],
        epsilon_min_train=ranges["epsilon"][0], epsilon_max_train=ranges["epsilon"][1],
        delta_min_train=ranges["delta"][0], delta_max_train=ranges["delta"][1],
    )
    try:
        r = generalized_mean_simple(N=N, p=1.5, hc_set=HCSettings(L=L), data_set=ds,
                                    opt_set=OptimizerSettings(**OPT), seed=100 + run, use_gpu=True,
                                    verbose=False, output_file="/tmp/ll_run.json")
    except Exception as exc:
        print(f"  !! failed {name} N={N} L={L} run={run}: {exc}", flush=True)
        r = None
    row = {"config": name, "d": latent_dim(ranges), "N": N, "L": L, "run": run}
    for k in ["test_rel_error", "test_abs_error", "train_loss", "epochs_run", "stopping_reason"]:
        row[k] = r[k] if r is not None else float("nan")
    return row


def done_keys():
    if not CSV.exists():
        return set()
    df = pd.read_csv(CSV)
    return {f"{r.config}|{r.N}|{r.L}|{r.run}" for r in df.itertuples()}


def report():
    df = pd.read_csv(CSV)
    pd.set_option("display.float_format", lambda x: f"{x:.2e}")
    pd.set_option("display.width", 200)
    for name, ranges in CONFIGS:
        sub = df[df.config == name]
        if sub.empty:
            continue
        piv = sub.groupby(["L", "N"])["test_rel_error"].median().reset_index().pivot_table(
            index="L", columns="N", values="test_rel_error")
        print(f"\n===== {name}  (d = {latent_dim(ranges)})  median test_rel_error vs N =====",
              flush=True)
        print(piv.to_string(), flush=True)


def main():
    done = done_keys()
    combos = list(itertools.product(CONFIGS, N_VALUES, L_VALUES))
    print(f"grid: {len(combos)} (config,N,L) x {N_SEEDS} seeds = {len(combos) * N_SEEDS} runs "
          f"({len(done)} done)", flush=True)
    new = not CSV.exists()
    start = time.perf_counter()
    with open(CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for (name, ranges), N, L in combos:
            for run in range(N_SEEDS):
                if f"{name}|{N}|{L}|{run}" in done:
                    continue
                row = run_one(name, ranges, N, L, run)
                writer.writerow(row)
                f.flush()
            print(f"  {name} N={N} L={L} done  ({time.perf_counter() - start:.0f}s)", flush=True)
    report()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
