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
from utilities import get_plot_params, get_results_by_tag


params = get_plot_params((8,3.5), 10,14) 
quantiles= [0.1,0.25,0.5,0.75,0.9]

output_dir = "./figures"
plot_name = "moments-deep-sets-nonlinear-residual"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()


project = "highdimensionaleconlab/symmetry_dynamic_programming"


#1. deepsets
df_deep= get_results_by_tag(project, "baseline_nonlinear_deep_sets", cols_config = ['seed'],test_results = True, cols=['train_time', 'retcode']) 
df_deep['residual_squared'] = df_deep['residual']**2
df_deep_0 = df_deep[df_deep['retcode']>=0]
quant_result_deep = df_deep_0.groupby('t').quantile(quantiles)['residual_squared'].unstack(level=-1)
quant_result_deep.reset_index(inplace=True)
quant_result_deep.columns = ['t'] + [f'quantile_{q}' for q in quantiles]

#2. moments
df_moments = get_results_by_tag(project, "baseline_nonlinear_deep_moments", cols_config = ['seed'],test_results = True, cols=['train_time', 'retcode'])
df_moments['residual_squared'] = df_moments['residual']**2
df_moments_0 = df_moments[df_moments['retcode']>=0] #Picking those that converged
quant_result_moments = df_moments_0.groupby('t').quantile(quantiles)['residual_squared'].unstack(level=-1)
quant_result_moments.reset_index(inplace=True)
quant_result_moments.columns = ['t'] + [f'quantile_{q}' for q in quantiles]

plt.rcParams.update(params)

# plotting

ax_moments = plt.subplot(121)
plt.plot(quant_result_moments["t"], quant_result_moments["quantile_0.5"], label = r"Median")
plt.fill_between(quant_result_moments["t"],quant_result_moments["quantile_0.1"], quant_result_moments["quantile_0.9"], color='cornflowerblue', alpha=0.2, label= r"$10$th and $90$th percentiles")
plt.fill_between(quant_result_moments["t"],quant_result_moments["quantile_0.25"], quant_result_moments["quantile_0.75"], color='cornflowerblue', alpha=0.6, label= r"$25$th and $75$th percentiles")
plt.title(r"Euler residuals squared ($\varepsilon^2$) with $\phi($Moments$)$")
plt.xlabel(r"Time($t$)")
ax_moments.set_yscale('log')
plt.legend(prop={"size": params['font.size']}, loc='lower right')
plt.tight_layout()

ax_deep = plt.subplot(122, sharey=ax_moments)
plt.plot(quant_result_deep["t"], quant_result_deep["quantile_0.5"], label = r"Median")
plt.fill_between(quant_result_deep["t"],quant_result_deep["quantile_0.1"], quant_result_deep["quantile_0.9"], color='cornflowerblue', alpha=0.2, label= r"$10$th and $90$th percentiles")
plt.fill_between(quant_result_deep["t"],quant_result_deep["quantile_0.25"], quant_result_deep["quantile_0.75"], color='cornflowerblue', alpha=0.6, label= r"$25$th and $75$th percentiles")
plt.title(r"Euler residuals squared ($\varepsilon^2$) with $\phi($ReLU$)$")
plt.xlabel(r"Time($t$)")
plt.legend(prop={"size": params['font.size']}, loc='upper right')
plt.tight_layout()

plt.show()
#plt.savefig(output_path)
