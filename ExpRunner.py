#!/usr/bin/env python
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow
from config import MLFLOW_TRACKING_URI

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
    """
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(exp_name)
    if not experiment:
        return False

    # List all runs in the experiment
    runs = mlflow.search_runs([experiment.experiment_id])
    if runs.empty:
        return False

    # Filter runs by name and check if any have completed successfully
    successful_runs = runs[(runs['tags.mlflow.runName'] == run_name) & (runs['status'] == 'FINISHED')]
    return not successful_runs.empty

class TaskList:
    def __init__(self, parent):
        # first command
        # None is a placeholder for no command
        self.parent = parent
        self.child = []

    def print(self):
        print(f"Parent:")
        print(f"  {self.parent}")
        print("Children:")
        for cmd in self.child:
            print(f"  {cmd}")
        print()
        

    def add_child(self, child):
        # list of commands to run after first command
        self.child.append(child)
    
    def run_parent(self):
        """Executes the parent command and waits for it to finish."""
        if self.parent is None:
            return True

        try:
            print(f"Executing parent command: {self.parent}")
            subprocess.run(self.parent, shell=True, check=True)
            print("Parent command completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing parent command: {e}")
            return False
        return True
    
    def run_child_parallel(self):
        """Executes all child commands in parallel."""

        valid_children = [cmd for cmd in self.child if cmd is not None]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(subprocess.run, cmd, shell=True, check=True) for cmd in valid_children]
            for future in futures:
                try:
                    future.result()  # Wait for each child command to complete
                    print("Child command completed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error in executing child command: {e}")

    def execute(self):
        """Executes the parent command followed by child commands in parallel if parent succeeds."""
        if self.run_parent():
            print("Starting child commands in parallel...")
            self.run_child_parallel()
        else:
            print("Parent command failed. Child commands will not be executed.")

def run_task_lists_in_parallel(task_lists):
    """Runs multiple TaskList objects in parallel."""
    with ThreadPoolExecutor() as executor:
        # Submit all task lists to the executor
        futures = [executor.submit(task.execute) for task in task_lists]
        
        # Optionally wait for all to complete and handle exceptions
        for future in futures:
            try:
                future.result()  # Wait for completion and raise exceptions if any
            except Exception as e:
                print(f"An error occurred: {e}")


class ExperimentRunner:
    def __init__(self, dryrun=True):
        os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
        self.dryrun = dryrun

        self.common = "problem poivar trainable_param D flags fixiter N_res_train 102 N_dat_train 101 datafile dataset/varpoi.mat output_activation id fwidth 64 width 128"
        self.init_exp = "poivar2_init"
        self.init_opt = f"max_iter 20000 testcase 0  experiment_name {self.init_exp}"

        self.inv_exp = "poivar2_inv"
        self.inv_opt = f"l1grad None l2grad 1e-3 max_iter 10000 experiment_name {self.inv_exp}"
        self.inv_testcase = 9
        
        self.nzopt = 'N_dat_train 51 use_noise True variance 0.0001'
        
        self.seeds = [0,1,2,3,4,5]

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e2', '1e1', '1e0']


    def create_ml_experiment(self):
        # Create MLflow experiments
        create_experiment_if_not_exists(self.init_exp)
        create_experiment_if_not_exists(self.inv_exp)

    def get_init_command(self, ttype, seed):
        prefix = "bsc" if ttype == "vanilla-init" else "simu"
        run_name = f"{prefix}_init_sd{seed}"
        if check_run_success(self.init_exp, run_name):
            print(f"Skipping {run_name}, success.")
            return None
        command = f"./runexp.py {self.common} {self.init_opt} traintype {ttype} run_name {run_name} seed {seed}"
        return command

    def get_inv_command(self, ttype, dweight, seed):

        if ttype == "adj-simu":
            restore_from = f"simu_init_sd{seed}"
            runname = f"simu_t{self.inv_testcase}_sd{seed}"
            weight = ""

        elif ttype == "vanilla-inv":
            restore_from = f"bsc_init_sd{seed}"
            runname = f"bsc_t{self.inv_testcase}_w{dweight}_sd{seed}"
            weight = f"data {dweight}"
        else:
            raise ValueError(f"Unknown traintype {ttype}")
        
        if check_run_success(self.inv_exp, runname):
            print(f"Skipping {runname} as it has already succeeded.")
            return None

        command = f"./runexp.py {self.common} {self.inv_opt} traintype {ttype} run_name {runname} testcase {self.inv_testcase} restore {self.init_exp}:{restore_from} {self.nzopt} {weight} seed {seed}"
        return command
    
    def create_tasks(self):
        # create a list of tasksList
        all_tasks = []
        for seed in self.seeds:
            adj_parent = TaskList(self.get_init_command('adj-init', seed))
            
            adj_parent.add_child(self.get_inv_command('adj-simu', None, seed))

            vanilla_parent = TaskList(self.get_init_command('vanilla-init', seed))
            for weight in self.vanilla_weights:
                vanilla_parent.add_child(self.get_inv_command('vanilla-inv', weight, seed))

            all_tasks.append(adj_parent)
            all_tasks.append(vanilla_parent)

        return all_tasks

    def run_experiments(self):
        self.create_ml_experiment()
        tasks = self.create_tasks()
        if self.dryrun:
            for task in tasks:
                task.print()
        else:
            run_task_lists_in_parallel(tasks)
    
class fkExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem fk trainable_param rD,rRHO flags fixiter Nt 51 Nx 51 datafile dataset/fk.mat width 128"
        
        # init setup
        self.init_exp = "fk_init"
        self.init_opt = f"max_iter 10000 testcase 1 experiment_name {self.init_exp}"

        # inverse problem setup
        self.inv_exp = "fk_inv"
        self.inv_opt = f"max_iter 50000 experiment_name {self.inv_exp} N_dat_train 11"
        self.inv_testcase = 2

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e-1', '1e0', '1e1']
        
        self.nzopt = 'use_noise True variance 0.0001'
        
        self.seeds = [0,1,2,3,4,5]

class VarPoiExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem poivar trainable_param D flags fixiter N_res_train 102 N_dat_train 101 datafile dataset/varpoi.mat output_activation id fwidth 64 width 128"
        
        # init setup
        self.init_exp = "poivar2_init"
        self.init_opt = f"max_iter 20000 testcase 0  experiment_name {self.init_exp}"

        # inverse problem setup
        self.inv_exp = "poivar2_inv_reg-4"
        self.inv_opt = f"l1grad None l2grad 1e-4 max_iter 10000 experiment_name {self.inv_exp}"
        self.inv_testcase = 9
        # 9 = hat function

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e2', '1e1', '1e0']
        
        self.nzopt = 'N_dat_train 51 use_noise True variance 0.0001'
        
        self.seeds = [0,1,2,3,4,5]

class HeatExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem heat flags fixiter Nt 51 Nx 51 datafile dataset/heat-2.mat fwidth 64 width 128"
        
        # init setup
        self.init_exp = "heat_init"
        self.init_opt = f"max_iter 10000 testcase 0  experiment_name {self.init_exp}"

        # inverse problem setup
        self.inv_exp = "heat_inv"
        self.inv_opt = f"l1grad None l2grad 1e-2 max_iter 10000 experiment_name {self.inv_exp} N_ic_train 102"
        self.inv_testcase = 2 
        # 2 = hat, 1=sin

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e1', '1e2', '1e3']
        
        self.nzopt = 'use_noise True N_dat_train 21 variance 0.001'
        
        self.seeds = [0,1,2,3,4]


class BurgerExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem burger flags fixiter  Nt 51 Nx 51 datafile dataset/burger-02.mat fwidth 64 width 128 output_activation id "
        
        # init setup
        self.init_exp = "burger_init"
        self.init_opt = f"max_iter 10000 testcase 3 experiment_name {self.init_exp}"

        # inverse problem setup
        self.inv_exp = "burger_inv_ndat21"
        self.inv_opt = f"l1grad None l2grad 1e-3 max_iter 10000 experiment_name {self.inv_exp} N_ic_train 102"

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e1', '1e2', '1e3']
        
        # self.nzopt = 'use_noise True N_dat_train 21 variance 0.001'
        self.nzopt = 'use_noise None N_dat_train 19'
        
        self.seeds = [0]
    
    def create_tasks(self):
        # create a list of tasksList
        all_tasks = []
        testcases = [1,2,3,4]
        
        for i in testcases:
            init_test = i
            
            init_runname = f"simu_init_t{init_test}"
            command = f"./runexp.py {self.common} {self.init_opt} traintype adj-init run_name {init_runname} testcase {init_test}"

            if check_run_success(self.init_exp, init_runname):
                command = None
            
            parent = TaskList(command)

            for j in testcases:
                # inverse problem
                # skip the same testcase
                if i == j:
                    continue

                inv_test = j
            
                inv_runname = f"simu_i{init_test}_t{inv_test}_l2reg-3_n19"
                command = f"./runexp.py {self.common} {self.inv_opt} traintype adj-simu {self.nzopt} run_name {inv_runname} testcase {inv_test} restore {self.init_exp}:{init_runname}"

                if check_run_success(self.inv_exp, inv_runname):
                    command = None

                parent.add_child(command)

            
            all_tasks.append(parent)


        return all_tasks



class odeExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem simpleode flags fixiter width 128 trainable_param a21 datafile ./dataset/odep3.mat N_res_train 101 N_dat_train 101"
        
        # init setup
        self.init_exp = "ode1p_init"
        self.init_opt = f"max_iter 10000 testcase 2 experiment_name {self.init_exp}"

        # inverse problem setup
        self.inv_exp = "ode1p_inv"
        self.inv_opt = f"max_iter 20000 experiment_name {self.inv_exp} N_res_train 11 N_dat_train 11"
        self.inv_testcase = 1

        # weights for vanilla experiments
        self.vanilla_weights = [ '1e-2', '1e-1', '1e0']
        
        self.nzopt = 'use_noise True variance 0.05'
        
        self.seeds = [0,1,2,3,4]


class DarcyExpRunner(ExperimentRunner):
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem darcy trainable_param f flags fixiter  fwidth 64 width 128 output_activation id"
        
        # init setup
        self.init_exp = "darcy_sig_init"
        self.init_opt = f"max_iter 10000 experiment_name {self.init_exp} datafile dataset/darcy_sigmoid.mat"

        # inverse problem setup
        self.inv_exp = "darcy_inv"
        self.inv_opt = f"l1grad 1e-9 l2grad None max_iter 5000 experiment_name {self.inv_exp} datafile dataset/darcy.mat"
    
    def create_tasks(self):
        # create a list of tasksList
        all_tasks = []

        # testcases = [3, 4, 5]
        # inf = [4, 5, 3]

        testcases = [ 2, 2]
        inf = [ 5, 3]

        all_tasks = []

        for i in range(len(testcases)):
            init_test = testcases[i]
            inv_test = inf[i]
            init_runname = f"simu_init_t{init_test}"
            command = f"./runexp.py {self.common} {self.init_opt} traintype adj-init run_name {init_runname} testcase {init_test}"

            if check_run_success(self.init_exp, init_runname):
                command = None
            
            parent = TaskList(command)

            inv_runname = f"simu_i{init_test}_t{inv_test}"
            command = f"./runexp.py {self.common} {self.inv_opt} traintype adj-simu run_name {inv_runname} testcase {inv_test} restore {self.init_exp}:{init_runname}"

            if check_run_success(self.inv_exp, inv_runname):
                command = None

            parent.add_child(command)

            all_tasks.append(parent)


        return all_tasks

        

class fkopExpRunner(ExperimentRunner):
    # FK with deeponet
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "width 128 trunk_depth 2 branch_depth 2"
        
        # init setup
        self.init_exp = "fkop_pretrain"
        self.init_opt = f"experiment_name {self.init_exp} max_iter 20000"

        # inverse problem setup
        self.inv_exp = "fkop_inv"
        self.inv_opt = f"experiment_name {self.inv_exp} testcase 2 datafile dataset/fk.mat"
        
        self.nzopt = 'use_noise True N_dat_train 11 variance 0.0001'
        self.seeds = [0,1,2,3,4,5]
    
    def create_tasks(self):
        # override the parent class method
        # create a list of tasksList


        dataset = {
            "coarse": "dataset/fk_op_data_coarse_n51.mat",
            "dense": "dataset/fk_op_data_dense_n51.mat",
            "ood": "dataset/fk_op_data_ood_n51.mat",
        }

        tasks = {}

        for key, value in dataset.items():
            command = f"./opengine.py experiment_name {self.init_exp} {self.common} run_name {key} datafile {value}  max_iter 20000"

            # check if the run already exists
            if check_run_success(self.init_exp, key):
                command = None

            tasks[key] = TaskList(command)
        
        for key, value in dataset.items():
            for seed in self.seeds:
                name=f"inv_{key}_sd{seed}"
                command = f"./opengine.py {self.common} {self.inv_opt} traintype inverse run_name {name} restore {self.init_exp}:{key} {self.nzopt} seed {seed}"

                if check_run_success(self.inv_exp, name):
                    command = None

                tasks[key].add_child(command)

        return list(tasks.values())


class VarPoiOpExpRunner(ExperimentRunner):
    # variable poisson with deeponet
    def __init__(self, dryrun=True):
        # call the parent class constructor
        super().__init__(dryrun)

        # common setup
        self.common = "problem varpoi param_dim 101 width 128 trunk_depth 2 branch_depth 2"
        
        # init setup
        self.init_exp = "varpoiop_pretrain"
        self.init_opt = f"experiment_name {self.init_exp} max_iter 20000"

        # inverse problem setup
        self.inv_exp = "varpoiop_inv"
        self.inv_opt = f"experiment_name {self.inv_exp} testcase 9 datafile dataset/varpoi.mat max_iter 20000"
        
        self.nzopt = 'N_dat_train 51 use_noise True variance 0.0001'
        self.seeds = [0,1,2,3,4,5]
        # weights for regularization
        self.l2grad = [1e-3, 1e-4, 1e-5]
        
    
    def create_tasks(self):
        # override the parent class method

        # https://stackoverflow.com/questions/66932956/python-mkl-threading-layer-intel-is-incompatible-with-libgomp-so-1-library        
        # export MKL_SERVICE_FORCE_INTEL=1
        os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

        # dataset with different length scale l = 0.1, 0.2, 0.3, 0.4 etc
        dataset = {"l01":"dataset/dat_op_varpoi_l01.mat",
        "l02":"dataset/dat_op_varpoi_l02.mat",
        "l03":"dataset/dat_op_varpoi_l03.mat",
        "l04":"dataset/dat_op_varpoi_l04.mat",
        }
        tasks = {}

        # pretrain model
        for key, value in dataset.items():
            command = f"./opengine.py {self.common} {self.init_opt} run_name {key} datafile {value}"

            # check if the run already exists
            if check_run_success(self.init_exp, key):
                command = None

            tasks[key] = TaskList(command)
        
        for key, value in dataset.items():
            for seed in self.seeds:
                for w in self.l2grad:
                    name=f"inv_{key}_l2grad{w}_sd{seed}"
                    command = f"./opengine.py {self.common} {self.inv_opt} traintype inverse run_name {name} restore {self.init_exp}:{key} {self.nzopt} seed {seed} l2grad {w}"

                    if check_run_success(self.inv_exp, name):
                        command = None

                    tasks[key].add_child(command)

        return list(tasks.values())

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='run tests.')
    parser.add_argument('test', type=str, help='Test to run')
    parser.add_argument('-n', '--dryrun', action='store_true', help='Print commands without running them.')

    args = parser.parse_args()


    match args.test:
        case "fk":
            experiment = fkExpRunner(dryrun=args.dryrun)
        case "varpoi":
            experiment = VarPoiExpRunner(dryrun=args.dryrun)
        case "heat":
            experiment = HeatExpRunner(dryrun=args.dryrun)
        case "burger":
            experiment = BurgerExpRunner(dryrun=args.dryrun)
        case "ode":
            experiment = odeExpRunner(dryrun=args.dryrun)
        case "fkop":
            experiment = fkopExpRunner(dryrun=args.dryrun)
        case "darcy":
            experiment = DarcyExpRunner(dryrun=args.dryrun)
        case "varpoiop":
            experiment = VarPoiOpExpRunner(dryrun=args.dryrun)
        case _:
            raise ValueError(f"Unknown test {args.test}")
    
    experiment.run_experiments()
