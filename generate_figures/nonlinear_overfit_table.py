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
tag = "deep_sets_nonlinear_overfit"

# Downloading the results
overall_tag = api.runs(sym_runs, filters={"tags": tag})
run_num=0 
cols_df = ['Number of data points', 'retcode',  'test_loss', 'train_loss']
cols = ["retcode", "test_loss", "train_loss"]  

for i in range(len(overall_tag)):
    x= [ ]
    x.append(float(overall_tag[i].config.get("model.train_subsample_trajectories")))
    for col in cols:
        x.append(float(overall_tag[i].summary.get(col)))
    array_x = np.array(x).reshape(1,len(cols_df))
    if i == 0:
        df =  pd.DataFrame(array_x, columns=cols_df)
    else: 
        df= pd.concat([df,pd.DataFrame(array_x, columns=cols_df)], axis =0)
    df = df.reset_index(drop=True)

# Preparing the results for the table

#Finding the ones that converged
df_converge = df[df["retcode"]==0] # Convergence 
# Calculating the succes rates
data_points= sorted(df_converge['Number of data points'].unique().tolist())
data_points = [int(x) for x in data_points]

success_rate = []
for i in data_points:
    success = 100*len(df_converge[df_converge["Number of data points"] == i])/len(df[df["Number of data points"] == i])
    success_rate.append(success)
# Creating the result data frame    
df_results = df_converge.groupby("Number of data points").median()
df_results = df_results.drop(['retcode'], axis=1)
df_results['success'] = success_rate
df_results['Number of data points'] = data_points
df_results = df_results[['Number of data points','success', 'train_loss', 'test_loss']]
df_results = df_results.set_index(['Number of data points'])
# Creating the latex file

def latex_table(df):
    df = df.rename(
            columns={
                "success":r"\shortstack{Success \\(\%)}",
                "train_loss": r"\shortstack{Train MSE \\ ($\varepsilon$)}",
                "test_loss": r"\shortstack{Test MSE \\ ($\varepsilon$)}"
            }
        )

    latex_str = df.to_latex(
        multicolumn=True,
        multirow=True,
        formatters=[
            "{:0.0f}\%".format,
            "{:.1e}".format,
            "{:.1e}".format
        ],
        index_names =  "number of ", 
        longtable=False,
        sparsify=True,
        escape=False,
    )
    latex_list = latex_str.splitlines()
    #latex_list.insert(6, '\midrule')
    latex_new = '\n'.join(latex_list)
    return latex_new

with open(output_dir + "/nonlinear_overfit_table.tex", "w") as file:
    file.write(latex_table(df_results))