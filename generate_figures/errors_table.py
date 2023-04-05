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
#group:[description, tag] 
# with group being the key, description and tag being values

networks = {

'Identity': [["Baseline", "baseline_identity"]],
'Moments': [["Baseline", "baseline_deep_moments"]], 
'Deep Sets': [["Baseline", "baseline_deep_sets"]]
}


#grabbing each element we want i.e in this case only retcode
def summary_run(group, description, tag):
    run_num=0 
    d=[]
    cols = [
        "retcode"
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
    df_retcode_0 = df[df["retcode"] >=0]
    df_retcode_0 = df_retcode_0.drop("retcode", axis =1)
    

    ##getting the median of the dataframe
    # creating a new dataframe with it to return median
    #adding in the aggregated count of
    new_df = pd.DataFrame(df_retcode_0.quantile(0.5).to_dict(), index = [group])
    new_df['Description'] = description
    new_df['retcode = 0'] = df[df['retcode'] ==0].count()['retcode']
    new_df['retcode = -2'] = df[df['retcode'] ==-2].count()['retcode']
    new_df['retcode = -1'] = df[df['retcode'] ==-1].count()['retcode']
    new_df['retcode = -3'] = df[df['retcode'] == -3].count()['retcode']
    return(new_df)

##making the latex table

def error_table(df):
    df = df.rename(
            columns={
                "retcode = 0":r"\shortstack{Success \\(\%)}",
                "retcode = -2": r"\shortstack{Violation of transversality\\ (\%)}",
                "retcode = -1": r"\shortstack{Early stopping failure \\ (\%)}",
                "retcode = -3": r"\shortstack{Overfitting \\ (\%)}",
            }
        )

    latex_str = df.to_latex(
        multicolumn=True,
        multirow=True,
        formatters=[
            "{:0.0f}\%".format,
            "{:0.0f}\%".format,
            "{:0.0f}\%".format,
            "{:0.0f}\%".format
        ],
        longtable=False,
        sparsify=True,
        escape=False,
    )
    return latex_str


first = 0

#getting all the different key value pairs in network and combining into one dataframe

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

summary_run_total = summary_run_total.reset_index()
summary_run_total.rename(columns={'index': 'Group'}, inplace = True)
summary_run_total= summary_run_total.set_index(['Group', 'Description'])

with open(output_dir + "/error_table.tex", "w") as file:
    file.write(error_table(summary_run_total))
