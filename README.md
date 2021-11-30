# Symmetry and Dynamic Programming
Source for "Exploiting Symmetry in High-Dimensional Dynamic Programming"

## Installing and Testing

1. Clone the repo and `cd` to it; i.e. run
        
        git clone https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git

    or use the `Code` button to the top right of this webpage (if you are on GitHub Desktop.)


3. Install dependencies.  Consider a conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```bash
pip install -r requirements.txt

```

If pytorch is not working, consider [installing manually](https://pytorch.org/get-started/locally/#start-locally) with `conda install pytorch cudatoolkit=10.2 -c pytorch ` or something similar, and then retrying the dependencies installation.
