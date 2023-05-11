import pandas as pd
import wandb

api = wandb.Api()

def test():
    print(1)
    return()

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

def first_satisfying_run(project, tag):
    overall_tag = api.runs(project, filters={"tags": tag})
    for i in range(len(overall_tag)): 
        if float(overall_tag[i].summary.get('retcode')) == 0:
            run_id = overall_tag[i].id
            reference_path = f'{project}/run-{run_id}-test_results:v0'
            artifact = api.artifact(str(reference_path))
            get = artifact.get("test_results")
            data = pd.DataFrame(data = get.data, columns = get.columns)
            return(data)

        

def satisfying_runs1(project, tag, config, artifact = False, summary = []):
    d=[]
    overall_tag = api.runs(project, filters={"tags": tag})
    
    if artifact == True: 

        test = 0
        first=True 
        for i in range(len(overall_tag)):
    
            run_id = overall_tag[i].id
            reference_path = f'{project}/run-{run_id}-test_results:v0'
            artifact = api.artifact(str(reference_path))

            if float(overall_tag[i].summary.get('retcode')) == 0: 
                get = artifact.get("test_results")
                data = pd.DataFrame(data = get.data, columns = get.columns)
                for n in config:
                    data[n] = overall_tag[i].config.get(n)
                for n in summary:
                    data[n] = overall_tag[i].summary.get(n)
                if first == True: 
                    all_df = data
                    first = False
                else:
                    test+=1
                    all_df = pd.concat([all_df, data])
                if test >5: 
                    return(all_df)
    
    else: 

        for i in range(len(overall_tag)):
            get = dict(overall_tag[i].summary)
            for n in config:
                get[n] = float(overall_tag[i].config.get(n))
                d.append(get) 
    
        return(pd.DataFrame.from_records(d))
