# Symmetry and Dynamic Programming
Source for "Exploiting Symmetry in High-Dimensional Dynamic Programming"

## Installing

1. Ensure you have installed Python.  For example, using [Anaconda](https://www.anaconda.com/products/individual)
2. Recommended but not required: Install [VS Code](https://code.visualstudio.com/) along with its [Python Extension](https://code.visualstudio.com/docs/languages/python)
3. Clone this repository
  - Recommended: With VS Code, go `<Shift-Control-P>` to open up the commandbar, then choose `Git Clone`, and use the URL `https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git`.  That will give you a full environment to work with.
  - Alternatively, you can clone it with git installed `git clone https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git`
4. Install dependencies.  Consider a conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).  With a terminal in that cloned folder,
```bash
pip install -r requirements.txt
```
If you are in VS Code, opening its [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) within the project window will start it in the correct location.

If pytorch is not working, consider [installing manually](https://pytorch.org/get-started/locally/#start-locally) with `conda install pytorch cpuonly -c pytorch ` or something similar, and then retrying the dependencies installation.  GPUs are not required for these experiments.

## Jupyter Notebook for Exploration

You can load the Jupyter notebook [baseline_example.ipynb](baseline_example.ipynb) directly in VS Code or on the command-line with `jupyter lab` run in the local directory.  This notebook loads the `investment_euler.py` and provides utilities to examine the output without using it on the commandline.



## CLI Usage
There is a command-line interface to solve for the equilibrium given various model and neural network parameters.  This is especially convenient for deploying on the cloud or when running in parallel.

The default values of all parameters is given by [investment_euler_default.yaml](investment_euler_default.yaml). You can override these by passing in a different YAML file, or by passing in the parameters on the commandline.

To use this, in a console at the root of this project, you can do things such as the following.
```bash
python investment_euler.py --trainer.max_epochs=5
```
Or to change the neural network architecture, you could try things such as increasing the `L` of the model
```bash
python investment_euler.py --trainer.max_epochs=2 --model.rho.n_in=8 --model.phi.n_out=8 
```
Or changing the number of layers
```bash
python investment_euler.py --trainer.max_epochs=5 --model.phi.layers=1
```

To change the economic variables such nonlinearity in prices, you could try things such as

```bash
python investment_euler.py --trainer.max_epochs=5 --model.nu=1.05
```

## Logs and Hyperparameter Tuning

Central to deep learning is the need to tuning hyperparameters.  See [W&B](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) for more details, including distributions.

To use, first create an account with [Weights and Biases](https://wandb.ai/) then, assuming you have installed the packages above, ensure you have logged in,
```bash
wandb login
```

For our example sweep, in a terminal run
```bash
wandb sweep train_time_sweep.yaml
```
This will create a new sweep on the server.  It will give you a URL to the sweep, which you can open in a browser.  You can also see the sweep in your [W&B dashboard](https://wandb.ai/home).  You will need the returned ID as well.

This doesn't create any "agents".  To do that, take the `<sweep_id>` that was returned and run
```bash
wandb agent <sweep_id>
```
Or to only execute a fixed number of experiments on that agent, give it a count (e.g. `wandb agent --count 10 <sweep_id>`).

You can then login to the server and run that same line, with the provided sweep_id, to execute the same experiments on a different machine.

See [TODO](TODO) for an example of one of these sweeps.