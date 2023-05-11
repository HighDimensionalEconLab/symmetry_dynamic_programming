import pandas as pd
import wandb

api = wandb.Api()

def get_plot_params(figsize, fontsize, ticksize):
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


def get_results_by_tag(project, tag, cols_config = [], test_results = False, cols_summary = []):
    # The `cols` will always automatically ahve the run name in it, which will help with filtering to check if there was more than one run since the new datastructure coudl have multiple rows per run.
    # if test_results = False then just collect all of the cols and cols_config and create a dataframe with them.
    # if test_results = True then create a dataframe by pulling down the test_results artifact, addd the cols and cols_config to that dataframe (which makes repeition but that is OK for our purposes here), and then just concatenates all of those dataframes together
   # return a single dataframe regardless.
    runs = api.runs(project, filters={"tags": tag})
    if test_results == False:
        d=[]
        for i in range(len(runs)):
            get = dict(runs[i].summary)
            for n in cols_config:
                get[n] = float(runs[i].config.get(n))
                d.append(get) 
        return(pd.DataFrame.from_records(d))
    
#ignore the test its just cuz I dont want to wait 
    else: 
        test = 0
        first=True 
        for i in range(len(runs)):
    
            run_id = runs[i].id
            reference_path = f'{project}/run-{run_id}-test_results:v0'
            artifact = api.artifact(str(reference_path))

            if float(runs[i].summary.get('retcode')) == 0: 
                get = artifact.get("test_results")
                data = pd.DataFrame(data = get.data, columns = get.columns)
                for n in cols_config:
                    data[n] = runs[i].config.get(n)
                for n in cols_summary:
                    data[n] = runs[i].summary.get(n)
                if first == True: 
                    all_df = data
                    first = False
                else:
                    test+=1
                    all_df = pd.concat([all_df, data])
                if test >10: 
                    return(all_df)
        return(all_df)
        
