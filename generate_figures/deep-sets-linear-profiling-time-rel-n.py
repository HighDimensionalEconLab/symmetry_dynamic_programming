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
plot_name = "deep-sets-linear-profiling-time-rel-n.pdf"

output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()

sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
def satisfying_runs(tag):
    run_num=0 
    d=[]
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    
    for i in range(len(overall_tag)):
        run_num+=1
        try: 
            get = dict(overall_tag[i].summary)
            get['seed'] = int(overall_tag[i].config.get('seed'))
            get['N'] = int(overall_tag[i].config.get('N'))
        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            d.append(get) 
    return(pd.DataFrame.from_records(d))

df_deep = satisfying_runs("baseline_deep_sets_N")            

##once we have retcode will be something like 
#df_successes= df_deep.loc[df_deep['ret_code']==0]


df_successes= df_deep.loc[df_deep['retcode']==0]


#making the dataframe for training time quartiles 
quantiles= [0.05,0.1,0.5,0.9,0.95]

quant_train_time_deep = df_successes.groupby('N').quantile(quantiles)['train_time'].unstack(level=-1)
quant_train_time_deep.reset_index(inplace=True)
quant_train_time_deep.columns = ['N'] + [f'quantile_{q}' for q in quantiles]


#making the dataframe for rel_error quartiles
# 
quant_test_u_rel_error_deep = df_successes.groupby('N').quantile(quantiles)['test_u_rel_error'].unstack(level=-1)
quant_test_u_rel_error_deep.reset_index(inplace=True)
quant_test_u_rel_error_deep.columns = ['N'] + [f'quantile_{q}' for q in quantiles]


plt.rcParams.update(params) 


ax_time = plt.subplot(121)
plt.plot(quant_train_time_deep['N'], quant_train_time_deep['quantile_0.5'])
plt.fill_between(quant_train_time_deep['N'], quant_train_time_deep["quantile_0.1"],quant_train_time_deep["quantile_0.9"], alpha=0.2)
ax_time.xaxis.set_ticks([1, 10, 100, 1000, 10000,100000])
ax_time.set_xscale('log')
plt.title(r"Computation time(seconds)")
plt.xlabel(r"N")

ax_loss = plt.subplot(122)
plt.plot(quant_test_u_rel_error_deep['N'], quant_test_u_rel_error_deep['quantile_0.5'])
plt.fill_between(quant_test_u_rel_error_deep['N'], quant_test_u_rel_error_deep["quantile_0.1"],quant_test_u_rel_error_deep["quantile_0.9"], alpha=0.2)
ax_loss.set_xscale('log')
ax_loss.xaxis.set_ticks([1, 10, 100, 1000, 10000, 100000])
plt.title(r"Relative Test Loss($\varepsilon$)")
plt.xlabel(r"N")

plt.tight_layout()

plt.savefig(output_path)

