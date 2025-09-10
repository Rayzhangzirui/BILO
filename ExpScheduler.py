#!/usr/bin/env python
import os
import sys
import argparse
import logging
import time
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import threading

import yaml

import numpy as np

import mlflow

from parseyaml import process_yaml
from utilgpu import pick_gpu_lowest_memory



def create_experiment_if_not_exists(experiment_name):
    """
    Create an MLflow experiment if it does not already exist.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        print(f"Experiment '{experiment_name}' already exists.")
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with Experiment ID: {experiment_id}")

def delete_run(exp_name, run_name):
    """
    Delete a run with the given name in the specified experiment.
    """
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is not None:
        runs = mlflow.search_runs([experiment.experiment_id])
        run = runs[runs['tags.mlflow.runName'] == run_name]
        if not run.empty:
            run_id = run.iloc[0].run_id
            mlflow.delete_run(run_id)
            print(f"Run '{run_name}' deleted from experiment '{exp_name}'")
        else:
            print(f"Run '{run_name}' not found in experiment '{exp_name}'")
    else:
        print(f"Experiment '{exp_name}' not found")


def check_run_success(exp_name, run_name):
    """
    Check if the specified run name in the given experiment name has completed successfully.
    Uses a cache to avoid repeated costly mlflow.search_runs calls.
    """
    if not hasattr(check_run_success, "cache"):
        check_run_success.cache = {}

    cache = check_run_success.cache

    # If we've already cached successful runs for this experiment, use the cache
    if exp_name in cache:
        return run_name in cache[exp_name]

    # Otherwise, query mlflow and cache the successful runs for this experiment
    experiment = mlflow.get_experiment_by_name(exp_name)
    if not experiment:
        cache[exp_name] = set()
        return False

    runs = mlflow.search_runs([experiment.experiment_id])
    if runs.empty:
        cache[exp_name] = set()
        return False

    # Cache all successful or running run names for this experiment
    successful_runs = set(runs.loc[runs['status'].isin(['FINISHED', 'RUNNING']), 'tags.mlflow.runName'])
    cache[exp_name] = successful_runs

    return run_name in successful_runs

MEMORY_THRESHOLD = 0  # MiB
TIME_OUT = 3600  # 1 hour
WAIT_TIME = 30  # 30 seconds before submitting the next process
GPUS = None
class ProcessRun:
    def __init__(self, experiment_name, run_name, command, parent=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.command = command
        self.parent = parent
        self.children = []
        self.process_id = None
        self.status = "pending"
    
    def __repr__(self):
        return f"ProcessRun({self.experiment_name}, {self.run_name}, {self.command}, {self.parent}, {self.children}, {self.process_id}, {self.status})\n"
        

    def execute_command(self):
        """Executes the command, waits until GPU is available, and logs the process."""
        logging.info(f"Starting {self.experiment_name}:{self.run_name} with command: {self.command}")

        # Check for GPU with sufficient memory, max wait time is 60 min
        start_time = time.time()
        best_gpu, available_memory = pick_gpu_lowest_memory(GPUS)
        while available_memory < MEMORY_THRESHOLD:  # Assuming MEMORY_THRESHOLD is defined
            time.sleep(60)  # Wait a while before retrying
            best_gpu, available_memory = pick_gpu_lowest_memory(GPUS)
            if time.time() - start_time > TIME_OUT:
                logging.error(f"No GPU available with memory > {MEMORY_THRESHOLD} MiB")
                self.status = "failed"
                return

        self.command += f" device cuda:{best_gpu}"
        # Run the command and wait for the process to start
        process = subprocess.Popen(self.command, shell=True)
        self.process_id = process.pid
        self.status = "running"
        logging.info(f"Running {self.run_name} on GPU {best_gpu} (PID: {self.process_id})")

        # Wait for the process to complete
        process.wait()
        if process.returncode == 0:
            self.status = "success"
        else:
            self.status = "failed"
        
        logging.info(f"{self.experiment_name}:{self.run_name} finished with status: {self.status}")

    def is_pid_in_nvidia_smi(self):
        """Check if the process PID is visible in nvidia-smi."""
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv"], capture_output=True, text=True).stdout
        return str(self.process_id) in output

class ExpScheduler:
    def __init__(self, dryrun=True, redo=False, filter_str=None, skip=False, program = None):

        self.command_dict = {}

        self.process_to_run = []

        # redo run even if it exists
        self.redo = redo

        # skip confirmation
        self.skip = skip

        # dryrun
        self.dryrun = dryrun

        # string to filter the run
        self.filter_str = filter_str
        

        self.program = program
    
    
    def read_config(self, yaml_file):
        try:
            # Read the YAML file
            with open(yaml_file, 'r') as file:
                yaml_string = file.read()

            # Load the YAML string into a Python dictionary
            data = yaml.safe_load(yaml_string)
            # data = yaml.load(yaml_string, Loader=yaml.FullLoader)

            # print full yaml if debug
            logging.debug(data)
            
            # Process the YAML data to flatten it into a dictionary
            self.command_dict = process_yaml(data)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        # set key as run_name
        for run_name, command in self.command_dict.items():
            command = command + f" run_name {run_name}"
            # append "./runexp.py" to the command
            self.command_dict[run_name] = f'{self.program} {command}'
        
        

    
    def filter_command(self):
        # filter the command based on the filter string
        if self.filter_str:
            # match regular expression    
            self.command_dict = {k: v for k, v in self.command_dict.items() if re.match(self.filter_str, k)}
        
        # print the filtered command
        logging.debug("Filtered command:")
        for key, command in self.command_dict.items():
            logging.debug(f"{key}")

    def process_dependencies(self):
        # for each argument in command_dict, extract experiment_name and run_name

        # filter the command based on the filter string
        self.filter_command()

        created_experiments = set()

        #  create experiment and collect the processes to run
        for key, command in self.command_dict.items():

            args = command.split()

            # check if experiment_name and run_name are present in the command
            if 'experiment_name' not in args or 'run_name' not in args:
                # skip the command if experiment_name and run_name are not present
                logging.info(f"Skipping command: {command}, experiment_name or run_name not found.")
                continue
                

            exp_name = args[args.index('experiment_name') + 1]
            run_name = args[args.index('run_name') + 1]
            
            parent_name = args[args.index('restore') + 1] if 'restore' in args else None
            
            # Modify the command for dry run if necessary, this need to be done after split
            if self.dryrun:
                command = f"echo '{command}'"  # Wrap the command with echo
            
            process = ProcessRun(exp_name, run_name, command, parent=parent_name)

            # check if run already exists
            if (not self.redo) and check_run_success(exp_name, run_name):
                logging.info(f"Run {exp_name}:{run_name} already exists. Skipping...")
                continue
            
            if exp_name not in created_experiments:
                create_experiment_if_not_exists(exp_name)
                created_experiments.add(exp_name)
                
            self.process_to_run.append(process)
            logging.info(f"Added process {exp_name}:{run_name} to the schedule.")
        
        successful_runs = set()
        # create parent and children relationship 
        for process in self.process_to_run:
            if process.parent:
                # find the parent process
                tmp = process.parent.split(':')
                parent_exp_name = tmp[0]
                parent_run_name = tmp[1]
                parent = [p for p in self.process_to_run if process.parent == p.experiment_name + ":" + p.run_name]
                
                # has to be only one parent or no parent
                if len(parent) == 0:
                    # if parent process does not exist in yaml, check if it exists in mlflow
                    if process.parent in successful_runs or check_run_success(parent_exp_name, parent_run_name):    
                        # add to successful runs
                        successful_runs.add(process.parent)
                        logging.info(f"Parent process {process.parent} already exists for {process.run_name}")
                        process.parent = None
                    else:
                        logging.error(f"Parent process {process.parent} not found for {process.run_name}, skip")
                        # raise ValueError(f"Parent process {process.parent} not found for {process.run_name}")
                        
                elif len(parent) ==1:
                    parent[0].children.append(process)
                    logging.info(f"Added {process.run_name} as child to {parent[0].run_name}")
                else:
                    logging.error(f"Multiple parent processes found for {process.run_name}")
                    raise ValueError(f"Multiple parent processes found for {process.run_name}")

        self.print_schedule()
        
        # ask for confirmation, proceed if 'y' or no input for 10 seconds
        def input_with_timeout(prompt, timeout):
            def get_input():
                nonlocal response
                response = input(prompt)

            response = None
            thread = threading.Thread(target=get_input)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                return 'y'
            return response

        if not self.dryrun and not self.skip:
            response = input_with_timeout("Do you want to run the above schedule? (y/n): ", 10)
            if response.lower() != 'y':
                logging.info("Exiting without running the schedule.")
                sys.exit(0)

    
    def print_schedule(self):
        # print the schedule for confirmation
        parent_processes = [proc for proc in self.process_to_run if proc.parent is None]

        # print parent processes and their children
        for proc in parent_processes:
            print(f"{proc.experiment_name}:{proc.run_name}\n{proc.command}")
            for child in proc.children:
                print(f"\t{child.experiment_name}:{child.run_name}\n\t{child.command}")
            print('\n')

    def run_parents_and_children(self):
        logging.info(f"Running in parallel mode...")
        parent_processes = [proc for proc in self.process_to_run if proc.parent is None]

        logging.info("Starting parent processes...")

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {}
            for proc in parent_processes:
                futures[executor.submit(self.run_with_children, proc)] = proc
                # sleep for 30 seconds to avoid running all the processes at once
                time.sleep(WAIT_TIME)

            for future in as_completed(futures):
                proc = futures[future]
                try:
                    future.result()  # This will raise any exception encountered
                    logging.info(f"Process {proc.run_name} finished with status: {proc.status}")
                except Exception as e:
                    logging.error(f"Process {proc.run_name} failed with error: {e}")
    
    def run_parents_and_children_sequential(self):
        logging.info(f"Running in sequential mode...")
        parent_processes = [proc for proc in self.process_to_run if proc.parent is None]

        logging.info("Starting parent processes...")

        for proc in parent_processes:
            self.run_with_children(proc)
            time.sleep(WAIT_TIME)

    def run_with_children(self, parent_process):
        # Run the parent process and then run the children in parallel
        logging.info(f"Running parent process {parent_process.run_name}...")
        parent_process.execute_command()

        if parent_process.status == "success":
            logging.info(f"Parent process {parent_process.run_name} completed successfully.")
            self.run_children_in_parallel(parent_process.children)
        else:
            logging.error(f"Parent process {parent_process.run_name} failed. Skipping children.")

    def run_children_in_parallel(self, children):
        if not children:
            return

        logging.info("Starting child processes...")

        with ThreadPoolExecutor() as executor:
            futures = {}

            for child in children:
                # sleep for 30 seconds to avoid running all the processes at once
                futures[executor.submit(child.execute_command)] = child
                time.sleep(30)

            for future in as_completed(futures):
                child = futures[future]
                try:
                    future.result()  # This will raise any exception encountered
                    logging.info(f"Child process {child.run_name} finished with status: {child.status}")
                except Exception as e:
                    logging.error(f"Child process {child.run_name} failed with error: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Positional argument for yaml
    parser.add_argument('yaml', type=str, help='path to the yaml file')
    
    # Optional arguments with shorthand
    parser.add_argument('--dryrun', '-n', action='store_true', help='Run the script in dry run mode')
    parser.add_argument('--redo', '-r', action='store_true', help='Redo the process')
    parser.add_argument('--seq', action='store_true', help='sequential run')
    parser.add_argument('--filter', '-f', type=str, default=None, help='filter the run')
    parser.add_argument('--mem', '-m', type=int, default=500, help='memory threshold to run the process')
    parser.add_argument('--time', '-t', type=int, default=3600, help='timeout')
    parser.add_argument('--log', '-l', type=str, default='ERROR', help='log level')
    parser.add_argument('--gpus', '-g', type=str, default='0,1,2,3,4,5,6,7', help='comma-separated list of available GPU IDs')
    parser.add_argument('--program', '-p', type=str, default='./runexp.py', help='program to run')

    # skip confirmation
    parser.add_argument('--skip', '-s', action='store_true', help='skip confirmation')
    
    parser.add_argument('--wait', '-w', type=int, default=30, help='wait time before submitting the next process')
    # set global
    WAIT_TIME = parser.parse_args().wait
    GPUS = parser.parse_args().gpus.split(',')
    MEMORY_THRESHOLD = parser.parse_args().mem
    # convert to int
    GPUS = [int(gpu) for gpu in GPUS]

    args = parser.parse_args()
    
    logging.basicConfig(level=args.log, format='%(asctime)s - %(levelname)s - PID %(process)d - %(message)s')
    exp_scheduler = ExpScheduler(dryrun=args.dryrun, redo=args.redo, filter_str=args.filter, skip=args.skip, program=args.program)
    exp_scheduler.read_config(args.yaml)
    exp_scheduler.process_dependencies()

    # save command to file
    pid = os.getpid()
    host_name = os.popen('hostname').read().strip()

    f = open("commands.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    f.write(f'  pid: {pid}')
    f.write(f'  host: {host_name}')
    f.write('\n')
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.close()

    if args.seq:
        exp_scheduler.run_parents_and_children_sequential()
    else:
        exp_scheduler.run_parents_and_children()
    

    # convert command line arguments to string
    string_command = ' '.join(sys.argv[1:])
    
    email_content = f"finished on {host_name} with command: {string_command}"
    os.system(f'echo {email_content} | mail -s "Experiment finished" ziruz16@uci.edu')