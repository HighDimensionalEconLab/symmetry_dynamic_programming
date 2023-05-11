import wandb
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utilities import get_plot_params, get_results_by_tag

output_dir = "./figures"
plot_name = "deep-sets-linear-profiling-var-n"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

params = get_plot_params((8, 3.5), 10, 14)
quantiles= [0.1,0.25,0.5,0.75,0.9]
df_deep = get_results_by_tag(project, "baseline_deep_sets_N", cols_config=['seed', 'N'])
#df_deep = satisfying_runs(project,"baseline_deep_sets_N", ['seed', 'N'])            
df_successes= df_deep.loc[df_deep['retcode']==0]

#making the dataframe for training time quartiles 
df_successes['train_time'] = pd.to_numeric(df_successes['train_time'])
quant_train_time_deep = df_successes.groupby('N')['train_time'].quantile(quantiles).unstack(level=-1)
quant_train_time_deep.reset_index(inplace=True)
quant_train_time_deep.columns = ['N'] + [f'quantile_{q}' for q in quantiles]


#making the dataframe for rel_error quartiles
# 
quant_test_u_rel_error_deep = df_successes.groupby('N')['test_u_rel_error'].quantile(quantiles).unstack(level=-1)
quant_test_u_rel_error_deep.reset_index(inplace=True)
quant_test_u_rel_error_deep.columns = ['N'] + [f'quantile_{q}' for q in quantiles]

plt.rcParams.update(params) 


ax_time = plt.subplot(121)
plt.plot(quant_train_time_deep['N'], quant_train_time_deep['quantile_0.5'], label= r"Median")
plt.fill_between(quant_train_time_deep['N'], quant_train_time_deep["quantile_0.1"],quant_train_time_deep["quantile_0.9"], color ='cornflowerblue', alpha=0.2, label = r"$10$th and $90$th percentiles")
plt.fill_between(quant_train_time_deep['N'], quant_train_time_deep["quantile_0.25"],quant_train_time_deep["quantile_0.75"], color ='cornflowerblue', alpha=0.6, label = r"$25$th and $75$th percentiles")
ax_time.set_xscale('log')
ax_time.xaxis.set_ticks([50, 100, 1000, 10000, 100000])
plt.title(r"Computation time (seconds)")
plt.xlabel(r"N")
plt.legend(prop={"size": params['font.size']}, loc='upper left')
plt.tight_layout()



ax_loss = plt.subplot(122)
plt.plot(quant_test_u_rel_error_deep['N'], quant_test_u_rel_error_deep['quantile_0.5'], label=r"Median")
plt.fill_between(quant_test_u_rel_error_deep['N'], quant_test_u_rel_error_deep["quantile_0.1"],quant_test_u_rel_error_deep["quantile_0.9"], color ='cornflowerblue', alpha=0.2, label = r"$10$th and $90$th percentiles")
plt.fill_between(quant_test_u_rel_error_deep['N'], quant_test_u_rel_error_deep["quantile_0.25"],quant_test_u_rel_error_deep["quantile_0.75"], color ='cornflowerblue', alpha=0.6, label = r"$25$th and $75$th percentiles")
ax_loss.set_xscale('log')
ax_loss.set_yscale('log')
ax_loss.xaxis.set_ticks([50, 100, 1000, 10000, 100000])
ax_loss.yaxis.set_ticks([0.0001, 0.001]) 
plt.title(r"Policy errors ($\epsilon_{\mathrm{rel}}$)")
plt.xlabel(r"N")
plt.legend(prop={"size": params['font.size']}, loc='lower left')
plt.tight_layout()

plt.show()
#plt.savefig(output_path)
