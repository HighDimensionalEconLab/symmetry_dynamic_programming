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

# success table invariance
df_no_invariance_success = 100 * (
    df[df["retcode"] == 0].groupby("model.N")["retcode"].count()
    / df.groupby("model.N")["retcode"].count()
)

# rel_error table invariance
df = df[df["retcode"] == 0]
df_no_invariance_test_rel_error = pd.DataFrame(df.groupby("model.N")["test_rel_error"].median())

df = get_results_by_tag(api, project, "generalized_mean_deep_sets_L_N", get_config=True)
# assert df_deep.id.nunique() == whenver we know how many

# success table deepsets
df_deep_sets_success = 100 * (
    df[df["retcode"] == 0].groupby(["model.N", "model.ml_model.L"])["retcode"].count()
    / df.groupby(["model.N", "model.ml_model.L"])["retcode"].count()
)
df_deep_sets_success = df_deep_sets_success.unstack("model.ml_model.L")
# rel_error table deepsets
df = df[df["retcode"] == 0]
df_deep_sets_rel_error = df.pivot_table(
    index="model.N", columns="model.ml_model.L", values="test_rel_error", aggfunc="median"
)

# success final table
df_success = pd.merge(df_deep_sets_success, df_no_invariance_success, on="model.N", how="outer")
df_success.rename_axis("N", axis="index", inplace=True)
df_success.rename(columns=lambda x: f'{x}_success', inplace=True)
  # so our table is formatted right in latex

# rel error final table
df_test_rel_error = pd.merge(
    df_deep_sets_rel_error, df_no_invariance_test_rel_error, on="model.N", how="outer"
)
df_test_rel_error.rename_axis("N", axis="index", inplace=True)
df_test_rel_error.rename(columns=lambda x: f'{x}_rel_error', inplace=True)
df_test_rel_error.rename(
    columns={"test_rel_error_rel_error": "no_invariance_rel_error"}, inplace=True
)  # just to make utilities mapping clearer

with open(output_dir + "/generalized_mean_L_N_rel_error.tex", "w") as file:
    file.write(df_to_latex(df_test_rel_error))

with open(output_dir + "/generalized_mean_L_N_success.tex", "w") as file:
    file.write(df_to_latex(df_success))
