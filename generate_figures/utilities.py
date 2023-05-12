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



def get_results_by_tag(api, project, tag, get_summary = True, get_config = False, get_test_results = False, max_runs = 1000):
    runs = api.runs(project, filters={"tags": tag})
    
    df = pd.DataFrame()
    for i in range(min(len(runs), max_runs)):
        id = runs[i].id
        cols = {'id': id, 'name': runs[i].name}
        if get_summary:
            cols.update(dict(runs[i].summary))
        if get_config:
            cols.update(dict(runs[i].config))        
 
        # Conditionally get the test results or just directly add the new values for the columns
        if get_test_results:
            reference_path = f"{project}/run-{id}-test_results:v0"
            test_results = api.artifact(str(reference_path)).get("test_results")
            run_data = pd.DataFrame(data = test_results.data, columns = test_results.columns)

            # Add columns across everything in the test_results
            for k, v in cols.items():
                run_data[k] = v
        else:
            # Create a dataframe with one row from the columns
            run_data = pd.DataFrame({k: [v] for k, v in cols.items()})

        df = pd.concat([df, run_data], ignore_index=True)
        
    return df



    # for all runs
    # always add in the run name and id, etc.  as coulmns
    # Then if summary = True, merge in the whole summary info
    # if config = True, merge in the whole config 
    # if test_resulst = False, just add that whole set of coluimns as a row
           # else add get the test_results, then add in those columns to all of the test results then concatentate the whole dataframe.
#Lets put this in a few clean branch off of main, add in a test_get_results.py function to the plotting which lets us play with it.
