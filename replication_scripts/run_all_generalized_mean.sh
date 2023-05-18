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

# List of "N" we will use for the no invariance experiments 
list_of_N="8 16 32"

# Path to the template
template="replication_scripts/generalized_mean_no_invariance_template.yaml"

# Loop over each N
for N in $list_of_N; do
    # Generate the output filename
    output_file="replication_scripts/generalized_mean_no_invariance_N_${N}.yaml"

    # Use sed to replace all occurrences of NNN with the current N
    sed "s/NNN/${N}/g" $template > $output_file
done

# Run all sweeps

run_sweep_and_agent "generalized_mean_deep_sets_L_N"

# Loop over each N for each invariance sweep
for N in $list_of_N; do
    run_sweep_and_agent "generalized_mean_no_invariance_N_${N}"
done