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

output_dir = "../figures"
plot_name = "fig_5"

output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()

sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
def first_satisfying_run(tag):
    run_num=0 
    d=[]
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    print(overall_tag[1].summary)
    print(overall_tag[1].config)
    for i in range(len(overall_tag)):
        run_num+=1
        try: 
            get = dict(overall_tag[i].summary)
            get['seed'] = int(overall_tag[i].config.get('seed'))
            get['N'] = int(overall_tag[i].config.get('N'))
            get['test_u_rel_error']= int(overall_tag[i].summary.get('test_u_rel_error')) 
            get['early_stopping_success']= str(overall_tag[i].summary.get('early_stopping_success'))
            get['train_time']= str(overall_tag[i].summary.get('train_time'))
            get['transversality_check_failed'] = str(overall_tag[i].summary.get('transversality_check_failed'))
        
        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            d.append(get) 
    return(pd.DataFrame.from_records(d))

pd = first_satisfying_run("baseline_deep_sets_N")            
print(pd)
