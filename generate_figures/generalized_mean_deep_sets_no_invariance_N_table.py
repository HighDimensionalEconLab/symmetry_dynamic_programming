import pandas as pd
import wandb
from utilities import df_to_latex, get_results_by_tag

output_dir = "./figures"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

N = [2, 4, 8, 32, 64, 512, 1024]
df = pd.DataFrame()
for n in N:
    df_n = get_results_by_tag(
        api, project, f"generalized_mean_no_invariance_N_{n}", get_config=True
    )
    # assert df.id.nunique() == whenever we know how many
    df = pd.concat([df_n, df])
    df = df[df['retcode'] == 0]
    
df_no_invar = pd.DataFrame(df.groupby("model.N")["test_rel_error"].median())

df_deep = get_results_by_tag(api, project, "generalized_mean_deep_sets_L_N", get_config=True)
df_deep = df_deep[df_deep["retcode"] == 0]
# assert df_deep.id.nunique() == whenver we know how many

df = df_deep.pivot_table(
    index="model.N", columns="model.ml_model.L", values="test_rel_error", aggfunc="median"
)
df_results = pd.merge(df, df_no_invar, on="model.N", how="left")
df_results.rename_axis("N", axis="index", inplace=True)
df_results.rename(columns={'test_rel_error': 'no_invariance_rel_error'}, inplace=True) #just so in utilities its clearly different

with open(output_dir + "/generalized_mean_deep_sets_no_invariance_N_table.tex", "w") as file:
    file.write(df_to_latex(df_results))
