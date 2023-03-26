#!/bin/bash

# Function which takes a sweep name, creates the sweep, then creates a single agent before continuing
# This function could be replaced with something like https://github.com/wandb/wandb/issues/5207
#!/bin/bash

# Define the project name
PROJECT_NAME="symmetry_dynamic_programming" # swap out globally

# Define the run_sweep_and_agent function
run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  
  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "replication_scripts/$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  wandb agent $SWEEP_ID
}

# Call experiments sequentially.  VERY SLOW
# baseline_deep_sets_N is left out for now due to the large number of experiments

# Primary examples with multiple seeds 
run_sweep_and_agent "baseline_deep_sets"
run_sweep_and_agent "baseline_deep_moments"
run_sweep_and_agent "baseline_identity"

# One run examples
# run_sweep_and_agent "baseline_deep_sets_one_run"
# run_sweep_and_agent "baseline_deep_moments_one_run"
# run_sweep_and_agent "baseline_identity_one_run"
# run_sweep_and_agent "deep_sets_nonlinear_nu_130_one_run"
# run_sweep_and_agent "deep_sets_nonlinear_nu_150_one_run"

# Additional robustness tables
