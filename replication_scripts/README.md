# Replication Instructions

1. Login with W&B to your entity, and choose a `<project_name>`.  Here we will use `symmetry_test`,

2. For every `.yaml` in this folder run this, e.g. `wandb sweep --name <yaml_file_subset> --project <project_name> <yaml_file>`   where the `name` argument is optional, but helps organize the sweeps in the W&B UI.  

For example:
```bash
wandb sweep --name baseline_deep_sets --project symmetry_test replication_scripts/baseline_deep_sets.yaml
wandb sweep --name baseline_deep_moments --project symmetry_test replication_scripts/baseline_deep_moments.yaml
```

Optionally, you can include your W&B entity with `--entity <entity_name>` as well.

3. For each of those sweeps, create an agent by copying from the output of the `wandb sweep` command line.  These have the form `wandb agent <agent_name>/<sweep_id>`

