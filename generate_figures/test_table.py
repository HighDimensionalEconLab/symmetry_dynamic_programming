
api = wandb.Api()
sym_runs = "highdimensionaleconlab/symmetry_dynamic_programming"


overall_tag = api.runs(sym_runs, filters={"tags": "baseline_deep_moments"})
overall_tag[1].summary 