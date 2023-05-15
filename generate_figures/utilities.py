import pandas as pd

def plot_params(figsize, fontsize=10, ticksize=14):
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
    return params


def get_results_by_tag(
    api,
    project,
    tag,
    get_summary=True,
    get_config=False,
    get_test_results=False,
    max_runs=1000,
    drop_summary_cols=[
        "test_results",
        "_wandb",
    ],  # causes trouble when merging the test_results dataframe
    drop_config_cols=[
        "trainer.logger.tags",
    ],
):

    runs = api.runs(project, filters={"tags": tag})
    df = pd.DataFrame()  # will concatenate

    for i in range(min(len(runs), max_runs)):
        run = runs[i]
        id = run.id
        cols = {"id": id, "name": run.name}
        if get_summary:
            # dropping details which don't fit in dataframes well
            cols.update(dict(run.summary))
            for col_name in drop_summary_cols:
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
            run_data = pd.DataFrame(data=test_results.data, columns=test_results.columns)

            # Add columns across everything for dropped columns.  Repetition but allows for indexing later
            for k, v in cols.items():
                run_data[k] = v
        else:
            # Create a data frame with one row from the columns
            run_data = pd.DataFrame({k: [v] for k, v in cols.items()})

        df = pd.concat([df, run_data], ignore_index=True)

    return df

def df_to_latex(df):
    formatters=[]

    potential_cols = {
    "success": {'name': r"\shortstack{Success \\(\%)}", 'format': "{:0.0f}\%".format},
    "train_time": {'name': r"\shortstack{Time \\ (s)}", 'format': "{:0.0f}".format},
    "trainable_parameters": {'name': r"\shortstack{Parameters \\ (Thousands, K)}", 'format': "{:.1f}".format},
    "train_loss": {'name': r"\shortstack{Train MSE \\ ($\varepsilon$)}", 'format': "{:.1e}".format},
    "val_loss": {'name': r"\shortstack{Val MSE \\ ($\varepsilon$)}", 'format': "{:.1e}".format},
    "test_loss": {'name': r"\shortstack{Test MSE \\ ($\varepsilon$)}", 'format': "{:.1e}".format},
    "test_u_rel_error": {'name': r"\shortstack{Policy Error\\ ($\epsilon_{\mathrm{rel}}$)}", 'format': "{:.2f}\%".format},
    "retcode = 0": {'name': r"\shortstack{Success \\(\%)}", 'format': "{:0.0f}\%".format},
    "retcode = -2": {'name': r"\shortstack{Violation of transversality\\ (\%)}", 'format': "{:0.0f}\%".format},
    "retcode = -1": {'name': r"\shortstack{Early stopping failure \\ (\%)}", 'format': "{:0.0f}\%".format},
    "retcode = -3": {'name': r"\shortstack{Overfitting \\ (\%)}", 'format': "{:0.0f}\%".format},
    }
    if isinstance(df.index, pd.MultiIndex):
        column_format = 'll'
    else: column_format= 'c' #I like number columb centered could also make l 
    

    for col in df.columns:
        if col in potential_cols.keys():
            column_format += 'c'  # center the columns contents depending on number of columns
            df = df.rename(columns={col: potential_cols[col]['name']})
            formatters.append(potential_cols[col]['format'])
        else:
            df = df.drop(columns=[col])
    
    latex_str = df.to_latex(
        multicolumn=True,
        multirow=True,
        formatters=formatters,
        longtable=False,
        sparsify=True,
        escape=False,
        column_format= column_format,
        na_rep = '-'
    )
    if isinstance(df.index, pd.MultiIndex):
        latex_str = latex_str.replace('\\bottomrule\n', '') #removes double line when multi-index 

    return latex_str
        


