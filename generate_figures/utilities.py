import pandas as pd
import wandb

def plot_params(figsize, fontsize, ticksize):
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "figure.figsize": figsize,
        "figure.dpi": 80,
        "figure.edgecolor": "k",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
    }
    return(params)



def get_results_by_tag(api, project, tag, get_summary = True, get_config = False, get_test_results = False, max_runs = 1000, drop_summary_cols = ["test_results", "_wandb"], drop_config_cols = []):

    runs = api.runs(project, filters={"tags": tag})    
    df = pd.DataFrame() # will concatenate

    for i in range(min(len(runs), max_runs)):
        run = runs[i]
        id = run.id
        cols = {'id': id, 'name': run.name}
        if get_summary:
            cols.update(dict(run.summary))
            for col_name in drop_summary_cols: # dropping details which don't fit in dataframes well
                if cols.get(col_name) is not None: 
                    del cols[col_name]
        if get_config:
            cols.update(dict(run.config))       
            for col_name in drop_config_cols:
                if cols.get(col_name) is not None: 
                    del cols[col_name]             
 
        # Conditionally get the test results or just directly add the new values for the columns
        if get_test_results:
            reference_path = f"{project}/run-{id}-test_results:v0"
            test_results = api.artifact(str(reference_path)).get("test_results")
            run_data = pd.DataFrame(data = test_results.data, columns = test_results.columns)

            # Add columns across everything for dropped columns.  Repetition but allows for indexing later
            for k, v in cols.items():
                run_data[k] = v
        else:
            # Create a data frame with one row from the columns
            run_data = pd.DataFrame({k: [v] for k, v in cols.items()})

        df = pd.concat([df, run_data], ignore_index=True)
        
    return df