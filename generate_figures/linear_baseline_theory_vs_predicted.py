import wandb
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from matplotlib import cm
import yaml
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


api = wandb.Api()
#overall_tag = api.runs(sym_runs, filters={"tags": "baseline_deep_sets"})
sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"
'''
def first_satisfying_run(summary_value, threshold, tag):
    overall_tag = api.runs(sym_runs, filters={"tags": tag})
    for i in range(len(overall_tag)):
        try: 
            value = overall_tag[i].summary.get(summary_value)
            print(overall_tag[i].summary.get('seed'))    
        except:
            print("THERE IS NO RUN DATA ASSOCIATED")
        else:
            if float(value) <= threshold:
                run_id = overall_tag[i].id
                reference_path = f'{sym_runs}/run-{run_id}-test_results:v0'
                artifact = api.artifact(str(reference_path))
                get = artifact.get("test_results")
                data = pd.DataFrame(data = get.data, columns = get.columns)
                return(data)
'''


artifact = api.artifact('highdimensionaleconlab/symmetry_dynamic_programming/run-mio1y1es-test_results:v0')
get = artifact.get("test_results")
data = pd.DataFrame(data = get.data, columns = get.columns)
data = data[data["ensemble"] == 0]
'''
plt.plot(
    data["t"], data["u_hat"], label=r"$u(X_t)$, $\phi($Identity$)$"
)

artifact1 = api.artifact('highdimensionaleconlab/symmetry_dynamic_programming/run-sussiyau-test_results:v0')
get1 = artifact1.get("test_results")
data1 = pd.DataFrame(data = get1.data, columns = get1.columns)
data1 = data1[data1["ensemble"] == 0]
plt.plot(
    data1["t"], data1["u_hat"], label=r"$u(X_t)$, $\phi$(ReLu$)$"
)
plt.legend(loc='best')
'''
diff = data["u_hat"]- data["u_reference"]
diff1 = data1["u_hat"]- data1["u_reference"]

plt.plot(
    data1["t"], diff
)
plt.plot(
    data1["t"], diff1
)


'''
df_deep = first_satisfying_run("test_u_rel_error", 0.001, "baseline_deep_sets_one_run")
df_deep = df_deep[df_deep["ensemble"] == 0]
#df_moments = first_satisfying_run("test_u_rel_error", 0.001, "baseline_moments")
#df_moments = df_moments[df_moments["ensemble"] == 0]
df_identity = first_satisfying_run("test_u_rel_error", 0.001, "baseline_identity_one_run")
df_identity = df_identity[df_identity["ensemble"] == 0]


plt.plot(
    df_deep["t"], df_deep["u_hat"]
)
plt.plot(
    df_identity["t"], df_identity["u_hat"]
)
'''
