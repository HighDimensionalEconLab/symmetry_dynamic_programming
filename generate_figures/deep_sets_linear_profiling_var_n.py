import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utilities import get_results_by_tag, plot_params

params = plot_params((8, 3.5))
quantiles= [0.1,0.25,0.5,0.75,0.9]

output_dir = "./figures"
plot_name = "deep-sets-linear-profiling-var-n"
output_path = output_dir + "/" + plot_name + ".pdf"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"         


df_deep = get_results_by_tag(api, project, "baseline_deep_sets_N", get_config=True)
assert(df_deep.id.nunique() == 1000)

df_deep = df_deep[df_deep["retcode"] == 0]
plot_dict = {'train_time':{'ticks_x':[50, 100, 1000, 10000, 100000], 'ticks_y':[20, 100],
                'title':r"Computation time (seconds)", 'loc':'upper left', 'subplot':121},
            'test_u_rel_error':{'ticks_x':[50, 100, 1000, 10000, 100000], 'ticks_y':[0.0001, 0.001],
                'title':r"Policy errors ($\epsilon_{\mathrm{rel}}$)", 'loc':'lower left', 'subplot':122}
                }

 
for n, vals in plot_dict.items():
    plt.rcParams.update(params)
    df = df_deep.groupby('N')[n].quantile(quantiles).unstack(level=-1)
    df.reset_index(inplace=True)
    df.columns = ['N'] + [f'quantile_{q}' for q in quantiles]
    ax = plt.subplot(vals['subplot'])
    plt.plot(df['N'], df['quantile_0.5'], label=r"Median")
    plt.fill_between(df['N'], df["quantile_0.1"],df["quantile_0.9"], color ='cornflowerblue', alpha=0.2, label = r"$10$th and $90$th percentiles")
    plt.fill_between(df['N'], df["quantile_0.25"],df["quantile_0.75"], color ='cornflowerblue', alpha=0.6, label = r"$25$th and $75$th percentiles")
    ax.set_xscale('log')
    if n == 'test_u_rel_error':
        ax.set_yscale('log')
        ticks_y1 = ax.get_yticks()

    ax.xaxis.set_ticks(vals['ticks_x'])
    ax.yaxis.set_ticks(vals['ticks_y']) 
    plt.title(vals['title'])
    plt.xlabel(r"N")
    plt.legend(prop={"size": params['font.size']}, loc=vals['loc'])
    plt.tight_layout()
plt.show()

#plt.savefig(output_path)
