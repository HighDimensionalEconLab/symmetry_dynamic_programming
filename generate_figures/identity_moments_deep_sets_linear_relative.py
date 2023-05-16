import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utilities import get_results_by_tag, plot_params

params = plot_params((10, 3.5))
plt.rcParams.update(params)


output_dir = "./figures"
plot_name = "identity_moments_deep_sets_linear_relative"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
tags = ["baseline_identity", "baseline_deep_moments", "baseline_deep_sets"]
title_names = ["Identity", "Moments", "ReLU"]

# grabbing and plotting the tagged sweeps from above
for n, tag in enumerate(tags):
    df = get_results_by_tag(api, project, tag, get_test_results=True)
    assert df.id.nunique() == 100
    df = df[df["retcode"] >= 0]
    quant_result = df.groupby("t")["u_rel_error"].quantile(quantiles).unstack(level=-1)
    quant_result.reset_index(inplace=True)
    quant_result.columns = ["t"] + [f"quantile_{q}" for q in quantiles]

    # need to set first
    if n == 0:
        ax_identity = plt.subplot(130 + n + 1)
        ax_identity.set_yscale("log")
        ax = ax_identity
    else:
        ax = plt.subplot(130 + n + 1, sharey=ax_identity)

    plt.plot(quant_result["t"], quant_result["quantile_0.5"], label=r"Median")
    plt.fill_between(
        quant_result["t"],
        quant_result["quantile_0.1"],
        quant_result["quantile_0.9"],
        color="cornflowerblue",
        alpha=0.2,
        label=r"$10$th and $90$th percentiles",
    )
    plt.fill_between(
        quant_result["t"],
        quant_result["quantile_0.25"],
        quant_result["quantile_0.75"],
        color="cornflowerblue",
        alpha=0.6,
        label=r"$25$th and $75$th percentiles",
    )
    plt.title(rf"""Policy errors ($\epsilon_{{\mathrm{{rel}}}}$) with $\phi(${title_names[n]}$)$""")
    plt.xlabel(r"Time($t$)")
    plt.legend(prop={"size": params["font.size"]}, loc="lower left")
    plt.tight_layout()

plt.savefig(output_path)
