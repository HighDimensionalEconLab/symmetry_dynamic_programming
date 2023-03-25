import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from matplotlib import cm
import yaml
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

fontsize = 10
ticksize = 14
figsize = (6, 3.5)
params = {
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize": figsize,
    "figure.dpi": 80,
    "figure.edgecolor": "k",
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
}

output_dir = "./figures"
plot_name = "linear-baseline-theory-vs-predicted"

output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
#overall_tag = api.runs(sym_runs, filters={"tags": "baseline_deep_sets"})
sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
def first_satisfying_run(summary_value, threshold, tag):
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    for i in range(len(overall_tag)):
        try: 
            value = overall_tag[i].summary.get(summary_value)
            #print(overall_tag[i].summary.get('seed'))    
        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            if float(value) <= threshold:
                run_id = overall_tag[i].id
                reference_path = f'{sym_runs}/run-{run_id}-test_results:v0'
                artifact = api.artifact(str(reference_path))
                get = artifact.get("test_results")
                data = pd.DataFrame(data = get.data, columns = get.columns)
                return(data)

df_deep = first_satisfying_run("test_u_rel_error", 0.005, "baseline_deep_sets_I_run")
df_deep = df_deep[df_deep["ensemble"] == 0]
df_moments = first_satisfying_run("test_u_rel_error", 0.005, "baseline_deep_moments_I_run")
df_moments = df_moments[df_moments["ensemble"] == 0]
df_identity = first_satisfying_run("test_u_rel_error", 0.005, "baseline_identity_I_run")
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
plt.legend(prop={"size": fontsize})
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

plt.savefig(output_path)
plt.clf()
