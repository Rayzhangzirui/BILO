#!/usr/bin/env python
# unified logger for mlflow, stdout, csv
import sys
import os
from util import *
from config import *
import mlflow
from MlflowHelper import *
from torchtnt.utils.loggers import StdoutLogger, CSVLogger
import socket


class Logger:
    def __init__(self, opts):
        self.opts = opts
        
        #
        if self.opts['use_mlflow']:
            runname = self.opts['run_name']
            expname = self.opts['experiment_name']
            
            # check if experiment exist. If not, create experiment
            if mlflow.get_experiment_by_name(expname) is None:
                mlflow.create_experiment(expname)
                print(f"Create experiment {expname}")

            # check if run_name already exist. If exist, raise warning
            if mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(expname).experiment_id, filter_string=f"tags.mlflow.runName = '{runname}'").shape[0] > 0:
                Warning(f"Run name {runname} already exist!")
                # Warning ValueError(f"Run name {runname} already exist!")

            mlflow.set_experiment(expname)
            self.mlflow_run = mlflow.start_run(run_name=self.opts['run_name'])
            mlflow.set_tag("host", socket.gethostname())
            self.mlrun = mlflow.active_run()

        self.save_dir = self.get_dir()

        if self.opts['use_stdout']:
            self.stdout_logger = StdoutLogger(precision=6)
        

        if self.opts['use_csv']:
            # use_mlflow = False
            assert not self.opts['use_mlflow'], "Cannot use both mlflow and csv"

            path = self.gen_path('metrics.csv')
            # error if file exist
            if os.path.exists(path):
                raise ValueError(f"File {path} already exist!")
            self.csv_logger = CSVLogger(path)
        
        
    def set_tags(self, key:str, value:str):
        if self.opts['use_mlflow']:
            mlflow.set_tags({key:value})
        else:
            print(key, value)

    def log_metrics(self, metric_dict:dict, step=None, prefix=''):
        # remove key with None value
        metric_dict = {prefix+k:v for k,v in metric_dict.items()}

        if self.opts['use_mlflow']:
            mlflow.log_metrics(to_double(metric_dict), step=step)

        if self.opts['use_stdout']:
            self.stdout_logger.log_dict(payload = metric_dict, step=step)

        if self.opts['use_csv']:
            self.csv_logger.log_dict(payload = metric_dict, step=step)

    def close(self):
        if self.opts['use_mlflow']:
            mlflow.end_run()

    def log_options(self, options: dict):
        # save optioins as json
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(options))
        savedict(options, self.gen_path('options.json'))
    
    def log_params(self, params: dict):
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(params))
        else:
            print(params)

    def get_dir(self):
        # get artifact dir
        if self.opts['use_mlflow']:
            return get_active_artifact_dir()
        else:
            dpath = os.path.join(RUNS, self.opts['experiment_name'], self.opts['run_name'])
            os.makedirs(dpath, exist_ok=True)
            return dpath

    def gen_path(self, filename: str):
        return os.path.join(self.save_dir, filename)
    
    def load_artifact(self, exp_name=None, run_name=None, name_str=None):
        # return all {filename: path} in artifact directory
        # load from mlflow or local directory
        if name_str is not None:
            try:
                parts = name_str.split(':')
                if len(parts) == 2:
                    # exp_name:run_name
                    source = 'mlflow' if self.opts['use_mlflow'] else 'local'
                    exp_name = parts[0]
                    run_name = parts[1]
                elif len(parts) == 3:
                    # source:exp_name:run_name, source = local or mlflow
                    source = parts[0]
                    assert source in ['local', 'mlflow'], "source must be 'local' or 'mlflow'"
                    exp_name = parts[1]
                    run_name = parts[2]
            except ValueError:
                raise ValueError("name_str must be in the format '[source:]exp_name:run_name'")

        if source == 'mlflow':
            # get artifact from mlflow
            helper = MlflowHelper()
            run_id = helper.get_id_by_name(exp_name, run_name)
            artifact_dict = helper.get_artifact_dict_by_id(run_id)
            
        else:
            # get files in directory
            dpath = os.path.join(RUNS, exp_name, run_name)
            print(f"Load artifact from {dpath}")
            artifact_dict = {fname: os.path.join(dpath, fname) for fname in os.listdir(dpath)}
            artifact_dict['artifacts_dir'] = dpath
        
        return artifact_dict



            




if __name__ == "__main__":
    # simple test of logger

    opts  = {'use_mlflow':False, 'use_stdout':True, 'use_csv':False, 'experiment_name':'tmp', 'run_name':'testlogger', 'save_dir':'./test'}

    # read options from command line key value pairs
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        key = args[i]
        val = args[i+1]
        opts[key] = val

    logger = Logger(opts)
    for i in range(10):
        logger.log_metrics({'loss':i}, step=i)
        logger.log_metrics({'param':i}, step=i)
    
    logger.log_options(opts)
    print('')
    print(logger.get_dir())
    print(logger.gen_path('test.txt'))

    logger.close()
    

