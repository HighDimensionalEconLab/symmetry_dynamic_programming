import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from utilities import get_results_by_tag, plot_params

fontsize = 10
ticksize = 14
figsize = (6, 3.5)
params = plot_params(((6, 3.5)))

output_dir = "./figures"
plot_name = "linear-baseline-theory-vs-predicted"

output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

plt.rcParams.update(params)
fig, ax = plt.subplots()

dfs = {
    "Identity": "baseline_identity_one_run",
    "Moments": "baseline_deep_moments_one_run",
    "ReLU": "baseline_deep_sets_one_run",
}
for name, tag in dfs.items():
    df = get_results_by_tag(api, project, tag, get_test_results=True)
    assert df.id.nunique() == 1  # check one run
    assert df.retcode[0] == 0  # check that its successful
    df = df[df["ensemble"] == 0]
    dfs[name] = df
    if name == "Identity":
        plt.plot(df["t"], df["u_reference"], dashes=[10, 5, 10, 5], label=r"$u(X_t)$, LQ")
    plt.plot(df["t"], df["u_hat"], label=rf"$u(X_t)$, $\phi(${name}$)$")

plt.legend(prop={"size": params["font.size"]})
plt.title(
    r"$u(X_t)$ with $\phi($Identity$)$, $\phi($Moments$)$ and $\phi($ReLU$)$ : Equilibrium Path"
)
plt.tight_layout()
plt.xlabel(r"Time($t$)")
plt.tight_layout()

axins = zoomed_inset_axes(ax, 12, loc="center")
for name, df in dfs.items():
    if name == "Identity":
        plt.plot(df["t"], df["u_reference"], dashes=[10, 5, 10, 5], label=r"$u(X_t)$, LQ")

    plt.plot(df["t"], df["u_hat"], label=rf"$u(X_t)$, $\phi(${name}$)$")

x1, x2, y1, y2 = 42.5, 44.5, 0.03415, 0.03435
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
mark_inset(ax, axins, loc1=2, loc2=4, linewidth="0.7", ls="--", ec="0.5")
plt.show()

plt.savefig(output_path)
