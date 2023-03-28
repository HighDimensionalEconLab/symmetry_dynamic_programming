import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import matplotlib
from matplotlib import cm
import yaml
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np


fontsize = 10
ticksize = 14
figsize = (8, 3.5)

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
plot_name = "moments-deep-sets-nonlinear-residual"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()


sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
def satisfying_runs(tag):
    run_num=0 #dont know a good way to get the first one so that it works
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    for i in range(len(overall_tag)):
        try: 
            run_id = overall_tag[i].id
            reference_path = f'{sym_runs}/run-{run_id}-test_results:v0'
            artifact = api.artifact(str(reference_path))
            retcode = overall_tag[i].summary.get('retcode') 
            test_loss = float(overall_tag[i].summary.get('test_loss')) 
            
             
        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            if retcode == 0 and np.isnan(test_loss) == False: 
                get = artifact.get("test_results")
                run_num+=1 #was breaking as no run data with first logged run
                data = pd.DataFrame(data = get.data, columns = get.columns)
                data['seed'] = overall_tag[i].config.get('seed')
                data['train_time'] = overall_tag[i].summary.get('train_time')
                data['retcode']=overall_tag[i].summary.get('retcode') 
                if run_num == 1: 
                    all_df = data
                else:
                    all_df = pd.concat([all_df, data])
    return(all_df)


# Preparing the results
quantiles= [0.1,0.25,0.5,0.75,0.9]

#1. deepsets

df_deep = satisfying_runs("baseline_nonlinear_deep_sets") 
df_deep_0 = df_deep[df_deep['retcode']>=0]
quant_result_deep = df_deep_0.groupby('t').quantile(quantiles)['residual'].unstack(level=-1)
quant_result_deep.reset_index(inplace=True)
quant_result_deep.columns = ['t'] + [f'quantile_{q}' for q in quantiles]

#2. moments

df_moments = satisfying_runs("baseline_nonlinear_deep_moments") 

df_moments_0 = df_moments[df_moments['retcode']>=0] #Picking those that converged
quant_result_moments = df_moments_0.groupby('t').quantile(quantiles)['residual'].unstack(level=-1)
quant_result_moments.reset_index(inplace=True)
quant_result_moments.columns = ['t'] + [f'quantile_{q}' for q in quantiles]

plt.rcParams.update(params)

# plotting

ax_moments = plt.subplot(121)
plt.plot(quant_result_moments["t"], quant_result_moments["quantile_0.5"], color= 'black', label = r"Median")
plt.fill_between(quant_result_moments["t"],quant_result_moments["quantile_0.1"], quant_result_moments["quantile_0.9"], color='gray', alpha=0.2, label= r"$10$th and $90$th percentiles")
plt.fill_between(quant_result_moments["t"],quant_result_moments["quantile_0.25"], quant_result_moments["quantile_0.75"], color='gray', alpha=0.6, label= r"$25$th and $75$th percentiles")
plt.title(r"Euler residuals ($\varepsilon$) with $\phi($Moments$)$")
plt.xlabel(r"Time($t$)")
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset= True)
plt.legend(prop={"size": fontsize}, loc='upper right')
plt.tight_layout()

ax_deep = plt.subplot(122, sharey=ax_moments)
plt.plot(quant_result_deep["t"], quant_result_deep["quantile_0.5"], color= 'black', label = r"Median")
plt.fill_between(quant_result_deep["t"],quant_result_deep["quantile_0.1"], quant_result_deep["quantile_0.9"],color='gray', alpha=0.2, label= r"$10$th and $90$th percentiles")
plt.fill_between(quant_result_deep["t"],quant_result_deep["quantile_0.25"], quant_result_deep["quantile_0.75"],color='gray', alpha=0.6, label= r"$25$th and $75$th percentiles")
plt.title(r"Euler residuals ($\varepsilon$) with $\phi($ReLU$)$")
plt.xlabel(r"Time($t$)")
plt.legend(prop={"size": fontsize}, loc='upper right')
plt.tight_layout()

plt.savefig(output_path)
plt.clf()
