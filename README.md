# Symmetry and Dynamic Programming
Source for "Exploiting Symmetry in High-Dimensional Dynamic Programming"


## Installing and Testing

1. Clone the repo and `cd` to it; i.e. run
        
        git clone https://github.com/HighDimensionalEconLab/symmetry_dynamic_programming.git

    or use the `Code` button to the top right of this webpage (if you are on GitHub Desktop.)

2. Install pytorch through conda.  For example, with a GPU:
```bashe
conda install pytorch cudatoolkit=11.3 -c pytorch -c conda-forge
```
Or without,
```bash
conda install pytorch cpuonly -c pytorch
```

3. Install packages
```bash
pip install -r requirements.txt
pip install -e .
```
