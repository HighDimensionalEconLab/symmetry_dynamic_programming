import pandas as pd
import wandb
from utilities import get_results_by_tag

api = wandb.Api()
project = "highdimensionaleconlab/symmetry_dynamic_programming"


df = get_results_by_tag(api, project,"deep_sets_nonlinear_nu_150_one_run")
assert(df.id.nunique() == 1) # i.e, one run
assert(df.retcode[0] == 0) # access the retcode of the first row, which is the only one here

dict_res = df.to_dict(orient='index')[0] # as a dictionary if only one row
assert(dict_res["retcode"] == 0) # dict don't support dict_res.retcode

# only get a subset of stuff.  Summary is by default, config and test_results are optional
df = get_results_by_tag(api, project,"baseline_deep_moments", max_runs = 2, get_summary = False, get_config = False)
assert(df.id.nunique() == 2)
assert(len(df.columns) == 2) # id and name always there

df = get_results_by_tag(api, project,"baseline_deep_moments", max_runs = 40)
assert(df.id.nunique() == 40)
df = df[df.retcode==0] # only keep successful runs
print(df.val_loss.mean()) # conditional on success

df = get_results_by_tag(api, project,"deep_sets_nonlinear_nu_150_one_run", get_test_results=True, get_summary = False)
assert(df.id.nunique() == 1) # only one artifact
assert(df.t.nunique() == 64)
df.drop(columns=["id", "name"], inplace=True) # given single run could drop to clean up clutter, but not really
df.set_index(["ensemble", "t"], inplace=True) # can index as we see fit.

#This is a big one and includes summary statistics.  Some examples below
df = get_results_by_tag(api, project,"baseline_deep_moments", get_test_results=True, max_runs = 10)
assert(df.id.nunique() == 10)
assert(df.t.nunique() == 64)
df.set_index(["id", "ensemble", "t"], inplace=True) # can index as we see fit.
df = df[df.retcode==0] #e.g. filter only successful runs
df.groupby("t").mean(numeric_only=True)  # groupings, etc.
df.groupby("t").quantile([0.95, 0.5, 0.05],numeric_only=True).u_rel_error # for example...
df.groupby("ensemble").mean(numeric_only=True).u_rel_error # across all t and all runs...