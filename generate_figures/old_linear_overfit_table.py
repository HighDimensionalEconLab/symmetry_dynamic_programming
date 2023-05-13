# this is straight up wrong and I think my other one is write
# I checked the number of successful runs for each subtrajectory and they
# dont match up with this table (they do with the new one)
# I also took the median of all the successful subtrajectories of 4 manually
# from WandB and it matches up with my other one but contridicts this one

"""for the test loss of num=4 
numbers = [0.00001372, 0.00001681, 0.000005382, 0.0000007683, 0.000005057, 0.000002397,
           0.00001621, 0.00000009214, 0.000006025, 0.00002953, 0.00006881, 0.0000001243,
           0.00000007725, 0.000004033, 0.0000009828, 0.0000003201, 0.0000002647, 0.000002318,
           0.000001257, 0.0000599, 0.00000002325, 0.00002791, 0.00002524, 0.00000228,
           0.00000009826, 0.00001323, 0.000002198, 0.00004253, 0.000002357, 0.000001247,
           0.00003301, 0.00007047, 0.00001097, 0.000003346, 0.00002906]

numbers.sort()  # Sort the list in ascending order
n = len(numbers)
if n % 2 == 0:
    median = (numbers[n//2 - 1] + numbers[n//2]) / 2
else:
    median = numbers[n//2]

print("Median: ", median)"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import yaml
import pytorch_lightning as pl
import os
import wandb

output_dir = "./figures"

api = wandb.Api()

sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
tag = "deep_sets_overfit"

# same as non linear one
# Downloading the results
overall_tag = api.runs(sym_runs, filters={"tags": tag})
run_num = 0
cols_df = ["Number of data points", "retcode", "test_u_rel_error", "test_loss", "train_loss"]
cols = ["retcode", "test_u_rel_error", "test_loss", "train_loss"]

for i in range(len(overall_tag)):
    x = []
    x.append(float(overall_tag[i].config.get("model.train_subsample_trajectories")))
    for col in cols:
        x.append(float(overall_tag[i].summary.get(col)))
    array_x = np.array(x).reshape(1, len(cols_df))
    if i == 0:
        df = pd.DataFrame(array_x, columns=cols_df)
    else:
        df = pd.concat([df, pd.DataFrame(array_x, columns=cols_df)], axis=0)
    df = df.reset_index(drop=True)

# Preparing the results for the table

# Finding the ones that converged
df_converge = df[df["test_u_rel_error"] < 0.01]  # Convergence criteria : test_u_rel_error < 0.01
# Calculating the succes rates
data_points = sorted(df_converge["Number of data points"].unique().tolist())
data_points = [int(x) for x in data_points]

success_rate = []
for i in data_points:
    success = (
        100
        * len(df_converge[df_converge["Number of data points"] == i])
        / len(df[df["Number of data points"] == i])
    )
    success_rate.append(success)
# Creating the result data frame
df_results = df_converge.groupby("Number of data points").median()
df_results["test_u_rel_error"] = df_results["test_u_rel_error"] * 100
df_results = df_results.drop(["retcode"], axis=1)
df_results["success"] = success_rate
df_results["Number of data points"] = data_points
df_results = df_results[
    ["Number of data points", "success", "train_loss", "test_loss", "test_u_rel_error"]
]
df_results = df_results.set_index(["Number of data points"])
# Creating the latex file


def latex_table(df):
    df = df.rename(
        columns={
            "success": r"\shortstack{Success \\(\%)}",
            "train_loss": r"\shortstack{Train MSE \\ ($\varepsilon$)}",
            "test_loss": r"\shortstack{Test MSE \\ ($\varepsilon$)}",
            "test_u_rel_error": r"\shortstack{Policy Error\\ ($\epsilon_{\mathrm{rel}}$)}",
        }
    )

    latex_str = df.to_latex(
        multicolumn=True,
        multirow=True,
        formatters=["{:0.0f}\%".format, "{:.1e}".format, "{:.1e}".format, "{:.2f}\%".format],
        index_names="number of ",
        longtable=False,
        sparsify=True,
        escape=False,
    )
    latex_list = latex_str.splitlines()
    # latex_list.insert(6, '\midrule')
    latex_new = "\n".join(latex_list)
    return latex_new


# with open(output_dir + "/linear_overfit_table.tex", "w") as file:
# file.write(latex_table(df_results))
