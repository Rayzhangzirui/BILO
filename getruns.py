#!/usr/bin/env python
from mlflow.tracking import MlflowClient
import pandas as pd

# Initialize the client
client = MlflowClient()

# Fetch all experiments
experiments = client.search_experiments()


# get all experiment id and name
all_experiment_id_name = {}

for experiment in experiments:
    name = experiment.name
    id = experiment.experiment_id
    all_experiment_id_name[id] = name


allruns = client.search_runs(all_experiment_id_name.keys())

# Prepare data for the table
all_runs_data = []

# get the experiment_id, run_id, experiment_name, run_name, artifacts_uri from all run
for run in allruns:
    # get exp name from exp id
    expname = all_experiment_id_name[run.info.experiment_id]
    
    # for artifact_uri, replace file:///home/ziruz16/ by /Users/Ray/project/
    artifact_uri = run.info.artifact_uri
    artifact_uri = artifact_uri.replace("file:///home/ziruz16/", "/Users/Ray/project/")

    all_runs_data.append([run.info.experiment_id, run.info.run_id, expname, run.info.run_name, artifact_uri])


# Create a DataFrame
df = pd.DataFrame(all_runs_data)
# Set the column names
df.columns = ['experiment_id', 'run_id', 'experiment_name', 'run_name', 'artifacts_uri']

# Save the DataFrame to a CSV file
df.to_csv('allruns.csv', index=False)
