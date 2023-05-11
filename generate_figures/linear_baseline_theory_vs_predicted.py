import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utilities import get_plot_params, first_satisfying_run

params =get_plot_params((6,3.5), 10, 14)

output_dir = "./figures"
plot_name = "linear-baseline-theory-vs-predicted"

output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"


df_deep = first_satisfying_run(project, "baseline_deep_sets_one_run")
df_deep = df_deep[df_deep["ensemble"] == 0]
df_moments = first_satisfying_run(project, "baseline_deep_moments_one_run")
df_moments = df_moments[df_moments["ensemble"] == 0]
df_identity = first_satisfying_run(project, "baseline_identity_one_run")
df_identity = df_identity[df_identity["ensemble"] == 0]


plt.rcParams.update(params)

fig, ax = plt.subplots()
plt.plot(
    df_deep["t"],
    df_deep["u_reference"],
    dashes=[10, 5, 10, 5],
    label=r"$u(X_t)$, LQ",
)
plt.plot(df_identity["t"], df_identity["u_hat"], label=r"$u(X_t)$, $\phi($Identity$)$")
plt.plot(df_moments["t"], df_moments["u_hat"], label=r"$u(X_t)$, $\phi$(Moments$)$")
plt.plot(df_deep["t"], df_deep["u_hat"], label=r"$u(X_t)$, $\phi($ReLU$)$")
plt.legend(prop={"size": params['font.size']})
plt.title(r"$u(X_t)$ with $\phi($Identity$)$, $\phi($Moments$)$ and $\phi($ReLU$)$ : Equilibrium Path")
plt.tight_layout()
plt.xlabel(r"Time($t$)")
plt.tight_layout()

axins = zoomed_inset_axes(ax, 12, loc="center")

plt.plot(
    df_deep["t"],
    df_deep["u_reference"],
    dashes=[10, 5, 10, 5],
    label=r"$u(X_t)$, LQ",
)

plt.plot(df_identity["t"], df_identity["u_hat"], label=r"$u(X_t)$, $\phi($Identity$)$")
plt.plot(df_moments["t"], df_moments["u_hat"], label=r"$u(X_t)$, $\phi$(Moments$)$")
plt.plot(df_deep["t"], df_deep["u_hat"], label=r"$u(X_t)$, $\phi($ReLU$)$")

x1, x2, y1, y2 = 42.5, 44.5, 0.03415, 0.03435
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
mark_inset(ax, axins, loc1=2, loc2=4, linewidth="0.7",ls="--", ec="0.5")

plt.show()
#plt.savefig(output_path)
#plt.clf()
