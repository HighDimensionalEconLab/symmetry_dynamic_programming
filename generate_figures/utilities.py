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



def get_results_by_tag(project, tag, test_results = False):
    api = wandb.Api()
    count=0
    runs = api.runs(project, filters={"tags": tag})
    df_test_results = pd.DataFrame()
    for i in range(len(runs)):
        count+=1
        df = pd.DataFrame()
        cols = dict(runs[i].summary)
        cols.update(dict(runs[i].config))
        cols['id'] = runs[i].id
        df = pd.DataFrame.from_dict(cols, orient='index').T
 
        if test_results:
            reference_path = f"{project}/run-{cols['id']}-test_results:v0"
            test_results = api.artifact(str(reference_path)).get("test_results")
            data = pd.DataFrame(data = test_results.data, columns = test_results.columns)
            df = pd.concat([df, data], axis = 1)
            df.fillna(method='ffill', inplace=True)
        df_test_results = pd.concat([df, df_test_results])
        
        if count == 5:
            return(df_test_results)  

    return(df_test_results) 



    # for all runs
    # always add in the run name and id, etc.  as coulmns
    # Then if summary = True, merge in the whole summary info
    # if config = True, merge in the whole config 
    # if test_resulst = False, just add that whole set of coluimns as a row
           # else add get the test_results, then add in those columns to all of the test results then concatentate the whole dataframe.
#Lets put this in a few clean branch off of main, add in a test_get_results.py function to the plotting which lets us play with it.
