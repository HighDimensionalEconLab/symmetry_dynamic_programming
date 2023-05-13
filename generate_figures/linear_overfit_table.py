import pandas as pd
import wandb
from utilities import df_to_latex, get_results_by_tag

output_dir = "../figures"
api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

df_results = pd.DataFrame()
df = get_results_by_tag(api, project, "deep_sets_overfit", get_config=True)
assert df.id.nunique() == 400
df = df.rename(columns={"model.train_subsample_trajectories": "Number of data points"})
df_results = pd.DataFrame()
df_results["success"] = df[df["retcode"] == 0].groupby("Number of data points").count()["id"]
df_results[["train_loss", "test_loss"]] = (
    df[df["retcode"] == 0].groupby("Number of data points")[["train_loss", "test_loss"]].median()
)
df_results["test_u_rel_error"] = (
    df[df["retcode"] == 0].groupby("Number of data points")["test_u_rel_error"].median() * 100
)

with open(output_dir + "/linear_overfit_table.tex", "w") as file:
    file.write(df_to_latex(df_results))
