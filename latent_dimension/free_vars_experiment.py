"""Auxiliary broad sweep -- in-distribution scan of the generalized-mean DeepSet over set size N,
number of training points, number of free DGP parameters, and latent width L (7 seeds each). Writes
one tidy row per run to a CSV (resumable) and prints median test-relative-error pivot tables per
free-var group.

This is the earlier, WIDER grid (N up to 1024, num_points in {5,10,100,1000}, four free-var configs).
It runs a short per-cell budget (ep2000) and therefore UNDER-TRAINS the harder cells -- it is kept for
its breadth, but the clean capacity story lives in latent_ladder.py / d3_compression.py / d3_skew.py.
Reference output: results/free_vars_results.csv (see results/NOTES.md).

Run:  uv run python latent_dimension/free_vars_experiment.py            (full sweep, resumes)
      uv run python latent_dimension/free_vars_experiment.py --smoke    (tiny grid, sanity)
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

# --- experiment grid (edit here to adapt) ---
N_VALUES = [16, 32, 64, 128, 256, 512, 1024]
NUM_POINTS = [5, 10, 100, 1000]
L_VALUES = [1, 2, 3, 4, 8]
N_SEEDS = 7
NUM_TEST = 2000  # fixed large test set -> stable rel-error estimate
P = 1.5

# (name, ranges) where each param is (min, max); a degenerate (min == max) param is held fixed.
CONFIGS = [
    ("mu", dict(mu=(1.0, 3.0), sigma=(0.3, 0.3), epsilon=(0.0, 0.0), delta=(1.0, 1.0))),
    ("mu+sigma", dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(0.0, 0.0), delta=(1.0, 1.0))),
    ("mu+sigma+eps", dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(-1.0, 1.0), delta=(1.0, 1.0))),
    ("mu+sigma+eps+delta", dict(mu=(1.0, 3.0), sigma=(0.1, 0.5), epsilon=(-1.0, 1.0), delta=(0.7, 1.5))),
]

OPT = dict(lr=0.005, batch_size=256, max_epochs=2000, stopping_threshold=1e-6,
           max_time=120.0, print_interval=0)

COLUMNS = ["config", "n_free", "N", "num_points", "L", "run", "seed", "train_data_seed",
           "test_data_seed", "test_rel_error", "test_abs_error", "test_loss", "train_loss",
           "epochs_run", "stopping_reason", "train_time"]

HERE = Path(__file__).parent


def n_free(ranges):
    return sum(lo != hi for lo, hi in ranges.values())


def run_one(name, ranges, N, npts, L, run):
    seed, train_seed, test_seed = 100 + run, 200 + run, 300 + run
    data_set = DataSettings(
        num_train_points=npts, num_test_points=NUM_TEST,
        train_data_seed=train_seed, test_data_seed=test_seed,
        mu_min_train=ranges["mu"][0], mu_max_train=ranges["mu"][1],
        sigma_min_train=ranges["sigma"][0], sigma_max_train=ranges["sigma"][1],
        epsilon_min_train=ranges["epsilon"][0], epsilon_max_train=ranges["epsilon"][1],
        delta_min_train=ranges["delta"][0], delta_max_train=ranges["delta"][1],
    )
    try:
        r = generalized_mean_simple(
            N=N, p=P, hc_set=HCSettings(L=L), data_set=data_set, opt_set=OptimizerSettings(**OPT),
            seed=seed, use_gpu=True, verbose=False, output_file="/tmp/fv_run.json",
        )
    except Exception as exc:  # one bad run shouldn't abort the sweep
        print(f"  !! failed {name} N={N} npts={npts} L={L} run={run}: {exc}", flush=True)
        r = None
    row = {
        "config": name, "n_free": n_free(ranges), "N": N, "num_points": npts, "L": L, "run": run,
        "seed": seed, "train_data_seed": train_seed, "test_data_seed": test_seed,
    }
    keys = ["test_rel_error", "test_abs_error", "test_loss", "train_loss", "epochs_run",
            "stopping_reason", "train_time"]
    row.update({k: (r[k] if r is not None else float("nan")) for k in keys})
    return row


def done_keys(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {f"{r.config}|{r.N}|{r.num_points}|{r.L}|{r.run}" for r in df.itertuples()}


def report(csv_path, medians_path):
    df = pd.read_csv(csv_path)
    med = (df.groupby(["config", "n_free", "N", "num_points", "L"])["test_rel_error"]
           .median().reset_index())
    med.to_csv(medians_path, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.2e}")
    pd.set_option("display.width", 200)
    for name, ranges in CONFIGS:
        sub = med[med["config"] == name]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=["N", "L"], columns="num_points", values="test_rel_error")
        print(f"\n===== {name}  (n_free={n_free(ranges)})  median test_rel_error =====", flush=True)
        print(pivot.to_string(), flush=True)
    print(f"\nrows: {len(df)}   tidy CSV: {csv_path}   medians: {medians_path}", flush=True)


def main(smoke):
    global N_VALUES, NUM_POINTS, L_VALUES, N_SEEDS, CONFIGS
    if smoke:
        N_VALUES, NUM_POINTS, L_VALUES, N_SEEDS = [16, 256], [5, 100], [1, 4], 2
        CONFIGS = CONFIGS[:1] + CONFIGS[3:]  # an easy and the hardest config
    csv_path = HERE / ("free_vars_results_smoke.csv" if smoke else "free_vars_results.csv")
    medians_path = HERE / ("free_vars_medians_smoke.csv" if smoke else "free_vars_medians.csv")

    done = done_keys(csv_path)
    combos = list(itertools.product(CONFIGS, N_VALUES, NUM_POINTS, L_VALUES))
    total = len(combos) * N_SEEDS
    print(f"grid: {len(combos)} combos x {N_SEEDS} seeds = {total} runs ({len(done)} already done)",
          flush=True)

    new = not csv_path.exists()
    start = time.perf_counter()
    n_done = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for ci, ((name, ranges), N, npts, L) in enumerate(combos):
            rels = []
            for run in range(N_SEEDS):
                if f"{name}|{N}|{npts}|{L}|{run}" in done:
                    continue
                row = run_one(name, ranges, N, npts, L, run)
                writer.writerow(row)
                f.flush()
                rels.append(row["test_rel_error"])
                n_done += 1
            tag = f"[{ci + 1}/{len(combos)}] {name} N={N} npts={npts} L={L}"
            if rels:
                med = pd.Series(rels).median()
                print(f"{tag}  median_test_rel={med:.2e}  ({time.perf_counter() - start:.0f}s, "
                      f"{n_done} new runs)", flush=True)
            else:
                print(f"{tag}  (cached)", flush=True)

    report(csv_path, medians_path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
