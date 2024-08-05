#!/usr/bin/env python
import sys
import os
from util import *
import mlflow

class MlflowHelper:
    # helper class for mlflow

    def __init__(self) -> None:
        self.client = mlflow.tracking.MlflowClient()
        
    
    def get_run(self, **kwargs):
        # get run id from experiment name and run name
        # kwargs: experiment_name, run_name, run_id
        if 'run_id' in kwargs:
            run_id = kwargs['run_id']
        else:
            experiment_name = kwargs['experiment_name']
            run_name = kwargs['run_name']
            run_id = self.get_id_by_name(experiment_name, run_name)
        return run_id
    
    def get_id_by_name(self, experiment_names, run_name):
        # get unique run id from experiment name and run name
        # experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # get run id from run name and experiment id

        # if experiment_names is a string, convert to list
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]

        runs = mlflow.search_runs(experiment_names=experiment_names, filter_string=f"run_name LIKE '{run_name}'")['run_id']
        if len(runs) == 0:
            raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_names}'")
        elif len(runs) > 1:
            # raise warning
            run_id = runs[0]
            print(f"Warning: Multiple runs found with name '{run_name}' in experiment '{experiment_names}'. Using the first one ({run_id}).")

        else:
            run_id = runs[0]

        return run_id

    def get_artifact_dict_by_id(self, run_id):
        # get all artifact paths from run_id
        
        # get artifact dir
        run = mlflow.get_run(run_id)
        artifact_dir = run.info.artifact_uri[7:] 

        # get all artifact paths
        artifacts = self.client.list_artifacts(run_id)
        paths = [artifact.path for artifact in artifacts]
        # get dictioinary of name - full path 
        paths = {path.split('/')[-1]: self.client.download_artifacts(run_id, path) for path in paths}
        paths['artifacts_dir'] = artifact_dir
        
        return paths

    def gen_artifact_path(self, run_id, fname):

        run = mlflow.get_run(run_id)
        artifact_dir = run.info.artifact_uri[7:] 
        return os.path.join(artifact_dir, fname)

    def get_metric_history(self, run_id):
        # get all metric from run
        metrics = self.client.get_run(run_id).data.metrics
        metrics_history = {}
        for key, value in metrics.items():
            print(key, value)
            
            histo = self.client.get_metric_history(run_id,key)
            values = [metric.value for metric in histo]
            steps = [metric.step for metric in histo]
            
            metrics_history[key] = values
        metrics_history['steps'] = steps
        return metrics_history, metrics


def get_active_artifact_dir():
    # Get the current artifact URI
    artifact_uri = mlflow.get_artifact_uri()

    # Remove 'file://' prefix
    if artifact_uri.startswith("file://"):
        artifact_dir = artifact_uri.replace("file://", "", 1)

    return artifact_dir

def get_active_artifact_path(filename):
    # generate path for current run
    artifact_dir = get_active_artifact_dir()
    # Append the filename
    file_path = os.path.join(artifact_dir, filename)

    return file_path


def load_artifact(exp_name=None, run_name=None, run_id=None, name_str=None):
    """ 
    Load options and artifact paths from mlflow run id or name
    """
    if name_str is not None:
        try:
            exp_name, run_name = name_str.split(':')
        except ValueError:
            raise ValueError("name_str must be in the format 'exp_name:run_name'")

    helper = MlflowHelper()        
    if run_id is None:
        run_id = helper.get_id_by_name(exp_name, run_name)

    artifact_paths = helper.get_artifact_dict_by_id(run_id)
    
    return artifact_paths
    
if __name__ == "__main__":
    # test if can get run_id corretly
    # python MlflowHelper.py experiment_name=experiment_1 run_name=run_1
    # python MlflowHelper.py run_id=1a2b3c4d5e6f7g8h9i0j
    
    helper = MlflowHelper()
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    id = helper.get_run(**kwargs)
    print(id)
    print_dict(helper.get_artifact_dict_by_id(id))
