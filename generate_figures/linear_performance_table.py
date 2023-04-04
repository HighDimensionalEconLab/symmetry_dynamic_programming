#import symmetry_dp
#from symmetry_dp import experiment_row, reorganize_performance_dataframe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import joypy
from matplotlib import cm
import yaml
import pytorch_lightning as pl
import os
import wandb

output_dir = "./figures"
plot_name = "linear_performance_table"

api = wandb.Api()
sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"

#Need to only adjust this for different runs and descriptions we are grabbing 
#group:[description(...will want to put these in latex format), tag] 
# with group being the key, description and tag being values

networks = {

'Identity': [["Baseline", "baseline_identity"], ["Thin (64 nodes)", "thin_64_identity"]],
'Moments': [["Baseline", "baseline_deep_moments"],
               ["Moments (1,2)","L_2_deep_moments"], ["Thin (64 nodes)", "thin_64_deep_moments"],
                ["Very Shallow (1 layer)", "very_shallow_1_layer_deep_moments"]], 
'Deep Sets': [["Baseline", "baseline_deep_sets"],
            ["L = 2", "L_8_deep_sets"],
            ["L = 16", "L_16_deep_sets"], 
            [r"$\textup{Deep}~(\phi:\textup{2 layers},  \rho:\textup{4 layers})$",  "deep_2_4_deep_sets"],
            [r"$\textup{Thin}~(\phi,\rho:\textup{64 nodes})$", "thin_64_deep_sets"],
            [r"$\textup{Shallow}~(\phi:\textup{1 layer},  \rho:\textup{2 layers})$", "shallow_1_2_deep_sets"], 
            ]
}



def summary_run(group, description, tag):
    run_num=0 
    d=[]
    cols = [
        "retcode",
        "train_time",
        "trainable_parameters",
        "train_loss",
        "test_loss",
        "val_loss",
        "test_u_rel_error"
    ]
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    
    for i in range(len(overall_tag)):
        run_num+=1
        x=[]
        try: 
            for col in cols:
                x.append(float(overall_tag[i].summary.get(col)))     

        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            if run_num == 1:
                df = pd.DataFrame(x, index=cols)
            else: 
                df= pd.concat([df,pd.DataFrame(x, index=cols)], axis=1)
    df = df.T.reset_index(drop=True)
    df["test_u_rel_error"] = df["test_u_rel_error"] * 100
    df.insert(0, "success", df[df['retcode']>=0].count()['retcode'] )
    df_retcode_0 = df[df["retcode"] >= 0]
    df_retcode_0 = df_retcode_0.drop("retcode", axis =1)
    df_retcode_0["trainable_parameters"] = df["trainable_parameters"] / 1000
    

    ##getting the median of the dataframe and creating a new dataframe with it to return median
    new_df = pd.DataFrame(df_retcode_0.quantile(0.5).to_dict(), index = [group])
    new_df['Description'] = description
    return(new_df)

##making the latex table
def linear_performance_table(df):
    df = df.rename(
            columns={
                "success":r"\shortstack{Success \\(\%)}",
                "train_time": r"\shortstack{Time \\ (s)}",
                "trainable_parameters": r"\shortstack{Params\\ (K)}",
                "train_loss": r"\shortstack{Train MSE \\ ($\varepsilon$)}",
                "test_loss": r"\shortstack{Test MSE \\ ($\varepsilon$)}",
                "val_loss": r"\shortstack{Val MSE \\ ($\varepsilon$)}",
                "test_u_rel_error": r"\shortstack{Policy Error\\ $\left(\frac{|u - u_{\text{ref}}|}{u_{\text{ref}}}\right)$}",
            }
        )

    latex_str = df.to_latex(
        multicolumn=True,
        multirow=True,
        formatters=[
            "{:0.0f}\%".format,
            "{:0.0f}".format,
            "{:.1f}".format,
            "{:.1e}".format,
            "{:.1e}".format,
            "{:.1e}".format,
            "{:.2f}\%".format
        ],
        longtable=False,
        sparsify=True,
        escape=False,
    )
    return latex_str


first = 0

#getting all the different networks and types and combining into one dataframe
for group in networks.keys():
    first +=1
    for description in networks[group]:
        try:
            summary_run_one = summary_run(group, description[0], description[1])
        except:
            print("no run yet")
        else: 
            if first == 1:
                first+=1
                summary_run_total = summary_run_one
            else:
                summary_run_total = pd.concat([summary_run_total,summary_run_one])

summary_run_total['success'] = summary_run_total['success'].fillna(0)
#summary_run_total = summary_run_total.replace(np.nan, "--")

summary_run_total = summary_run_total.reset_index()
summary_run_total.rename(columns={'index': 'Group'}, inplace = True)
summary_run_total= summary_run_total.set_index(['Group', 'Description'])

with open(output_dir + "/linear_performance_table.tex", "w") as file:
    file.write(linear_performance_table(summary_run_total))
