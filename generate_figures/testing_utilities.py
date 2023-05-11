import pandas as pd
import wandb
from utilities import get_results_by_tag

project = "highdimensionaleconlab/symmetry_dynamic_programming"

#for one runs
df_nu_150_test_results = get_results_by_tag(project,"deep_sets_nonlinear_nu_150_one_run", test_results=True)
df_nu_150 = get_results_by_tag(project,"deep_sets_nonlinear_nu_150_one_run")

#for multiple artifact ones I stopped it after grabbing 5
baseline_deep_moments_test_results = get_results_by_tag(project,"baseline_deep_moments", test_results=True)
baseline_deep_moments = get_results_by_tag(project,"baseline_deep_moments")
