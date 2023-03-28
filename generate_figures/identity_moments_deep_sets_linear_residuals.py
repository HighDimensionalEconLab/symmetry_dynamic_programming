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
figsize = (10, 3.5)

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

low_quant = 0.025
up_quant = 0.975
median = 0.5

output_dir = "./figures"
plot_name = "identity_moments_deep_sets_linear_residuals"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()

#overall_tag = api.runs(sym_runs, filters={"tags": "baseline_deep_sets"})
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
            


#deep

df_deep = satisfying_runs("baseline_deep_sets")

residuals_deep = (
    df_deep.groupby(["ensemble", "t"]).agg({"residual": "mean"}).reset_index()
)
residuals_deep["error"] = residuals_deep["residual"]
errors_deep = residuals_deep

# removing the residual column 
errors_deep = errors_deep.drop(['residual'], axis = 1)


#preparing the deep table
deep_table = errors_deep.groupby("t").agg({"error": ["mean"]})
deep_table.columns = ["error_mean"]
deep_table["median"] = errors_deep.groupby("t").quantile(median)["error"]
deep_table["up_quant"] = errors_deep.groupby("t").quantile(up_quant)["error"]
deep_table["low_quant"] = errors_deep.groupby("t").quantile(low_quant)["error"]
deep_table["time"] = deep_table.index



#identity: when we get the actual runs for that 

df_identity = satisfying_runs("baseline_identity")
residuals_identity = (
    df_identity.groupby(["ensemble", "t"]).agg({"residual": "mean"}).reset_index()
)
residuals_identity["error"] = residuals_identity["residual"]
errors_identity = residuals_identity

# removing the residual column 
errors_identity = errors_identity.drop(['residual'], axis = 1)

#preparing the moment table
identity_table = errors_identity.groupby("t").agg({"error": ["mean"]})
identity_table.columns = ["identity_mean"]
identity_table["median"] = errors_identity.groupby("t").quantile(median)["error"]
identity_table["up_quant"] = errors_identity.groupby("t").quantile(up_quant)["error"]
identity_table["low_quant"] = errors_identity.groupby("t").quantile(low_quant)["error"]
identity_table["time"] = identity_table.index

#test= residuals_moment.loc[residuals_moment['error']>= 10]
#test.groupby('seed').count()

#Moments

df_moment = satisfying_runs("baseline_deep_moments")
residuals_moment = (
    df_moment.groupby(["ensemble", "t"]).agg({"residual": "mean"}).reset_index()
)
residuals_moment["error"] = residuals_moment["residual"]
errors_moment = residuals_moment

# removing the residual column 
errors_moment = errors_moment.drop(['residual'], axis = 1)

#preparing the moment table
moment_table = errors_moment.groupby("t").agg({"error": ["mean"]})
moment_table.columns = ["error_mean"]
moment_table["median"] = errors_moment.groupby("t").quantile(median)["error"]
moment_table["up_quant"] = errors_moment.groupby("t").quantile(up_quant)["error"]
moment_table["low_quant"] = errors_moment.groupby("t").quantile(low_quant)["error"]
moment_table["time"] = moment_table.index

# combining the Plots

plt.rcParams.update(params)


ax_moments = plt.subplot(132)
plt.plot(moment_table["time"], moment_table["median"])
plt.fill_between(moment_table["time"],moment_table["low_quant"], moment_table["up_quant"], alpha=0.2)
#ax_moments.set_yscale('log')
plt.title(r"Test MSE ($\varepsilon$) with $\phi($Moments$)$")
plt.xlabel(r"Time($t$)")

ax_identity = plt.subplot(131, sharey=ax_moments)
plt.plot(identity_table["time"], identity_table["median"])
plt.fill_between(identity_table["time"],identity_table["low_quant"], identity_table["up_quant"], alpha=0.2)
plt.title(r"Test MSE ($\varepsilon$) with $\phi($Identity$)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

ax_deep = plt.subplot(133, sharey=ax_moments)
plt.plot(deep_table["time"], deep_table["median"])
plt.fill_between(deep_table["time"],deep_table["low_quant"], deep_table["up_quant"], alpha=0.2)
plt.title(r"Test MSE ($\varepsilon$) with $\phi($ReLU$)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

#delete from this to inserts once have the other runs

# Creating insets
median_deep = deep_table["median"]
time_deep = deep_table["time"]

time_window = [20, 30]
ave_value = 0.5 * (median_deep[time_window[0]] + median_deep[time_window[1]])
window_width = 0.25*ave_value
matplotlib.rcParams.update({'ytick.labelsize': 5})

axins = zoomed_inset_axes(ax_deep, 4, loc="lower right")
axins.plot(time_deep, median_deep)
plt.fill_between(time_deep, deep_table["low_quant"], deep_table["up_quant"], alpha=0.2)
x1, x2, y1, y2 = (
    time_window[0],
    time_window[1],
    ave_value - window_width,
    ave_value + window_width,
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=5, visible = False)
plt.yticks(fontsize=5)
mark_inset(ax_deep, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

plt.savefig(output_path)
