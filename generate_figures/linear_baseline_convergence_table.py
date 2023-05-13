import pandas as pd
import wandb
from utilities import df_to_latex, get_results_by_tag

output_dir = "../figures"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

# what do I call this?
networks = {
    "Identity": {"Baseline": "baseline_identity"},
    "Moments": {"Baseline": "baseline_deep_moments"},
    "Deep Sets": {"Baseline": "baseline_deep_sets"},
}
retcodes = [0, -1, -2, -3]
df_full = pd.DataFrame()
for name, dict in networks.items():
    for key, tag in dict.items():
        df = get_results_by_tag(api, project, tag)
        assert df.id.nunique() == 100
        df_summary = pd.DataFrame()
        df_summary["Description"] = [key]
        df_summary["Group"] = [name]
        for num in retcodes:
            df_summary[f"retcode = {num}"] = [df[df["retcode"] == num].count()["retcode"]]
        df_full = pd.concat([df_full, df_summary])
df_full = df_full.set_index(["Group", "Description"])

with open(output_dir + "/linear_baseline_convergence_table.tex", "w") as file:
    file.write(df_to_latex(df_full))
