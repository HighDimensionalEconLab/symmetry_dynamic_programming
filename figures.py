import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml


## cleaning the lightning logs folder
lightning_logs_path = 'lightning_logs/'

if os.path.exists(lightning_logs_path):
   shutil.rmtree(lightning_logs_path)
else:
    print("Good, your lightning logs folder is empty")

## running the linear prices code 
os.system('python baseline_example.py')
## running the non-linear prices code for \nu = 1.5
os.system('python baseline_example.py --model.nu 1.5')
## running the non-linear prices code for \nu = 1.1
os.system('python baseline_example.py --model.nu 1.1')
## running the non-linear prices code for \nu = 1.05
os.system('python baseline_example.py --model.nu 1.05')


## Plots

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


## Plotting the linear prices setup

output_dir = "./figures"
experiment_root = "./lightning_logs/version_0/"
plot_name = "linear-baseline-theory-vs-predicted"
output_path = output_dir + "/" + plot_name + ".pdf"

df_deep_linear = pd.read_csv(experiment_root + "test_results.csv")
df_deep_linear = df_deep_linear[df_deep_linear["ensemble"] == 0]

plt.rcParams.update(params)
fig, ax = plt.subplots()

plt.plot(
    df_deep_linear["t"],
    df_deep_linear["u_reference"],
    dashes=[10, 5, 10, 5],
    label=r"$u(X_t)$, LQ")
plt.plot(df_deep_linear["t"], df_deep_linear["u_hat"], label=r"$u(X_t)$, $\phi($ReLU$)$")
plt.legend(prop={"size": fontsize})
plt.title(r"$u(X_t)$ with $\phi($ReLU$)$ : Equilibrium Path")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

plt.savefig(output_path)
plt.clf()

## Plotting the non-linear prices setup
experiment_root_nu = "./lightning_logs/version_"

plot_name_nu = "deep-sets-nonlinear-var-nu"
output_path_nu = output_dir + "/" + plot_name_nu + ".pdf"

indices = [1, 2, 3, 0]
for index in indices:
    source_path_yaml = experiment_root_nu+str(index)+"/"+ "config.yaml"
    with open(source_path_yaml) as file:
        info = yaml.full_load(file)
        nu = info['model']['nu']
    source_path = experiment_root_nu+str(index)+"/"+ "test_results.csv"
    df = pd.read_csv(source_path)
    df = df[df["ensemble"] == 0]
    plt.plot(df["t"], df["u_hat"], label=r"$\nu$ = " + str(nu))

plt.legend(prop={"size": fontsize})
plt.title(r"$u(X_t)$ with $\phi($ReLU$)$: Equilibrium Path")
plt.xlabel(r"Time(t)")
plt.tight_layout()

plt.savefig(output_path_nu)
plt.clf()
