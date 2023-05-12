import pandas as pd
import wandb
from utilities import get_results_by_tag

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"

df_nu_150 = get_results_by_tag(api, project,"deep_sets_nonlinear_nu_150_one_run")
print(df_nu_150["id"].nunique()) # counts the number of unique, I think?
baseline_deep_moments = get_results_by_tag(api, project,"baseline_deep_moments", max_runs = 3)
baseline_deep_moments_no_stuff = get_results_by_tag(api, project,"baseline_deep_moments", max_runs = 2, get_summary = False, get_config = False)
print(baseline_deep_moments["id"].nunique()) # check unique, I think?

#for one runs
df_nu_150_test_results = get_results_by_tag(api, project,"deep_sets_nonlinear_nu_150_one_run", test_results=True)

#for multiple artifact ones I stopped it after grabbing 5
baseline_deep_moments_test_results = get_results_by_tag(api, project,"baseline_deep_moments", test_results=True)
baseline_deep_moments = get_results_by_tag(api, project,"baseline_deep_moments")
