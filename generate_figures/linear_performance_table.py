import pandas as pd
import wandb
from utilities import df_to_latex, get_results_by_tag

output_dir = "./figures"

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

networks = {
    "Identity": {"Baseline": "baseline_identity"},
    "Moments": {
        "Baseline: Moments (1,2,3,4)": "baseline_deep_moments",
        "Moments (1,2)": "L_2_deep_moments",
        "Very Shallow (1 layer)": "very_shallow_1_layer_deep_moments",
    },
    "Deep Sets": {
        "Baseline: L= 4": "baseline_deep_sets",
        "L = 2": "L_8_deep_sets",
        "L = 16": "L_16_deep_sets",
        r"$\textup{Deep}~(\phi:\textup{2 layers},  \rho:\textup{4 layers})$": "deep_2_4_deep_sets",
        r"$\textup{Shallow}~(\phi:\textup{1 layer},  \rho:\textup{2 layers})$": "shallow_1_2_deep_sets",
    },
}

cols = ["train_time", "train_loss", "val_loss", "test_loss"]
df_full = pd.DataFrame()
for name, dict in networks.items():
    for key, tag in dict.items():
        df = get_results_by_tag(api, project, tag)
        assert df.id.nunique() == 100
        df_success = df[df["retcode"] == 0]
        df_summary = pd.DataFrame()
        df_summary["Description"] = [key]
        df_summary["Group"] = [name]
        df_summary["success"] = [df_success.count()["retcode"]]
        df_summary["trainable_parameters"] = [df["trainable_parameters"].quantile(0.5) / 1000]
        for col in cols:
            df_summary[col] = [df_success[col].quantile(0.5)]
        df_summary["test_u_rel_error"] = [df_success["test_u_rel_error"].quantile(0.5) * 100]

        df_full = pd.concat([df_full, df_summary])
df_full = df_full.set_index(["Group", "Description"])

with open(output_dir + "/linear_performance_table.tex", "w") as file:
    file.write(df_to_latex(df_full))
