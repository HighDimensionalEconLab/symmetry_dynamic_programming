#!/bin/bash

# Function which takes a sweep name, creates the sweep, then creates a single agent before continuing
# This function could be replaced with something like https://github.com/wandb/wandb/issues/5207
run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  
  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --name "$SWEEP_NAME" "replication_scripts/$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  wandb agent $SWEEP_ID
}

# Call all experiments sequentially
run_sweep_and_agent "baseline_deep_moments"
