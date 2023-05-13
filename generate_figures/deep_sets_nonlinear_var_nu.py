import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utilities import get_results_by_tag, plot_params

params = plot_params((6, 3.5))

output_dir = "./figures"
plot_name = "deep-sets-nonlinear-var-nu"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

fig, ax = plt.subplots()
plt.rcParams.update(params)

# looping through the runs with nu variations
nu = {
    "deep_sets_nonlinear_nu_150_one_run": 150,
    "deep_sets_nonlinear_nu_130_one_run": 130,
    "baseline_deep_sets_one_run": 100,
}

for name, n in nu.items():
    df = get_results_by_tag(api, project, name, get_test_results=True)
    assert df.id.nunique() == 1
    assert df.retcode[0] == 0
    df = df[df["ensemble"] == 0]
    label = rf"$\nu = {n/100:.1f}$"
    plt.plot(df["t"], df["u_hat"], label=label)

plt.legend(prop={"size": params["font.size"]}, loc="lower right")
plt.title(r"$u(X_t)$ with $\phi($ReLU$)$: Equilibrium Path")
plt.xlabel(r"Time(t)")
plt.tight_layout()
plt.show()
# plt.savefig(output_path)
# plt.clf()
