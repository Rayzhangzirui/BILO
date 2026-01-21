#!/usr/bin/env python

import argparse
import mlflow

def create_experiment_if_not_exists(experiment_name):
    """
    Create an MLflow experiment if it does not already exist.

    :param experiment_name: The name of the experiment to create.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        print(f"Experiment '{experiment_name}' already exists.")
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with Experiment ID: {experiment_id}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Create MLflow experiments if they do not exist.')
    parser.add_argument('experiment_names', nargs='+', help='A list of experiment names to create.')

    # Parse the arguments
    args = parser.parse_args()

    # Process each experiment name
    for experiment_name in args.experiment_names:
        create_experiment_if_not_exists(experiment_name)

if __name__ == '__main__':
    main()
