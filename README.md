# Symmetry and Dynamic Programming
Source for "Exploiting Symmetry in High-Dimensional Dynamic Programming"

## Installing and Testing

1. Ensure you have installed Python.  For example, using [Anaconda](https://www.anaconda.com/products/individual)
2. Recommended but not required: Install [VS Code](https://code.visualstudio.com/) along with its [Python Extension](https://code.visualstudio.com/docs/languages/python)
3. Clone this repository.
  - Recommended: With VS Code, go `<Shift-Control-P>` to open up the commandbar, then choose `Git Clone`, and use the URL `https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git`.  That will give you a full environment to work with.
  - ALternatively, you can clone it with git installed `git clone https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git`
4. Install dependencies.  Consider a conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).  With a terminal in that cloned folder,
```bash
pip install -r requirements.txt
```
If you are in the VS Code, open its [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) within the project window will start in the correct location.

If pytorch is not working, consider [installing manually](https://pytorch.org/get-started/locally/#start-locally) with `conda install pytorch cudatoolkit=10.2 -c pytorch ` or something similar, and then retrying the dependencies installation.

## Jupyter Notebook for Exploration

You can load the Jupyter notebook [baseline_example.ipynb](baseline_example.ipynb) directly in VS Code or on the command-line with `jupyter lab` run in the local directory.  This notebook loads the `baseline_example.py` and provides a utility to explore it in the notebook instead of on the command-line.



## CLI Usage
There is a command-line interface to solve for the equilibrium given various model and neural network parameters.  This is especially convenient for deploying on the cloud (e.g. using https://grid.ai) when running in parallel.

To use this, in a console at the root of this project, you can do things such as the following.
```bash
python baseline_example.py --trainer.max_epochs 5
```
Or to change a neural network architecture, you could try things such as 
```bash
python baseline_example.py --trainer.max_epochs 5 --model.L 8
```
```bash
python baseline_example.py --trainer.max_epochs 5 --model.phi_layers 1
```

To change the economic variables such nonlinearity in prices, you could try things such as

```bash
python baseline_example.py --trainer.max_epochs 5 --model.nu 1.05
```

To see the list of all possible options, see `python baseline_example.py --help`

The output of these prints to the console, but is also saved in a folder named `lightning_logs` for the particular experiment.  This includes
- `config.yaml` which lets you see the full set of parameters used in the experiment
- `metrics.yaml` for a summary of the results
- `test_results.csv` which includes the full on the "test" trajectories.  That CSV file can be loaded for plotting
- `checkpoints/best.ckpt` which is the results of the training process.  See [here](https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#checkpoint-loading) for more details on loading checkpoints.  For example, `model = InvestmentEulerBaseline.load_from_checkpoint(PATH)`

Finally, tensorboard is an important tool to examine the convergence of machine learning models when trying to find the appropriate parameters.  After executing an experiment or two, go into your console and type
```
tensorboard --logdir .
```
It will give you a local URL (e.g., http://localhost:6006/ ) to analyze your results.
