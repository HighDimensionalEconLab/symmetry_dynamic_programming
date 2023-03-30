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
plot_name = "deep-sets-nonlinear-var-nu"

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
            
df_nu_150 = first_satisfying_run("test_loss", 1e-5, "deep_sets_nonlinear_nu_150_one_run")
df_nu_150 = df_nu_150[df_nu_150["ensemble"] == 0]

df_nu_130 = first_satisfying_run("test_loss", 1e-5, "deep_sets_nonlinear_nu_130_one_run")
df_nu_130 = df_nu_130[df_nu_130["ensemble"] == 0]

df_nu_100 = first_satisfying_run("test_u_rel_error", 0.005, "baseline_deep_sets_one_run")
df_nu_100 = df_nu_100[df_nu_100["ensemble"] == 0]

plt.rcParams.update(params)

fig, ax = plt.subplots()
plt.plot(df_nu_150["t"], df_nu_150["u_hat"], label=r"$\nu = 1.5$")
plt.plot(df_nu_130["t"], df_nu_130["u_hat"], label=r"$\nu = 1.3$")
plt.plot(df_nu_100["t"], df_nu_100["u_hat"], label=r"$\nu = 1.0$")


plt.legend(prop={"size": fontsize})
plt.title(r"$u(X_t)$ with $\phi($ReLU$)$: Equilibrium Path")
plt.xlabel(r"Time(t)")
plt.tight_layout()

plt.savefig(output_path)
plt.clf()