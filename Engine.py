#!/usr/bin/env python
import os
from torchinfo import summary

from Options import *
from util import *
from Problems import create_pde_problem
from lossCollection import lossCollection

from Logger import Logger
from Trainer import *
from Sampler import *

def get_trainer(traintype):
    network, method = traintype.split('-')

    if network == 'pinn':
        return PinnTrainer
    elif network in {'fno', 'deeponet'}:
        if method == 'inv':
            return OperatorInverseTrainer
        elif method == 'init':
            return OperatorPretrainTrainer
    elif network == 'bilo':
        if method == 'simu':
            return BiLevelTrainer
        elif method == 'init':
            return BiloInitTrainer
    elif network == 'exact':
        if method == 'inv':
            return UpperTrainer
    elif network == 'bpinn':
        if method == 'psgld' or method == 'sgld' or method == 'map':
            return PinnTrainer
    else:
        raise ValueError('trainer not found')
    
    if method == 'mala':
        return MetropolisAdjustedLangevinDynamics
    elif method == 'hmc':
        return HamiltonianMonteCarlo

class Engine:
    def __init__(self, opts=None) -> None:

        self.device = set_device(opts['device'])
        self.opts = opts
        self.restore_artifacts = {}
        self.logger = None

        self.logger = None
        self.trainer = None

        self.setup_logger()
        self.restore_run()

    def setup_logger(self):
        self.logger = Logger(self.opts['logger_opts'])

    def setup_problem(self):
        # setup PDE problem, mathematical model, initial parameter, gt parameters etc
        self.pde = create_pde_problem(self.opts['pde_opts'])

        # init/sample/optimization give different loss function
        self.pde.config_traintype(self.opts['traintype'])
        
        # init/sample/optimization use different dataset
        self.pde.setup_dataset(self.opts['dataset_opts'], self.opts['noise_opts'], self.device)
        
        # setup network
        self.net = self.pde.setup_network(**self.opts['nn_opts'])

    def restore_opts(self, restore_opts):
        ''' only restore neural network options, other options are from command line or default
        '''
        # self.opts['nn_opts'].update(restore_opts['nn_opts'])
        do_not_restore = ['rff_trainable','train_embed','rank']
        for key in restore_opts['nn_opts']:
            if key in do_not_restore:
                continue
            self.opts['nn_opts'][key] = restore_opts['nn_opts'][key]
        
    def restore_run(self):
        # if restore is empty, do nothing
        if self.opts['restore'] != '':
            # if is director
            if os.path.isdir(self.opts['restore']):
                opts_path = os.path.join(self.opts['restore'], 'options.json')
                restore_opts = read_json(opts_path)
                self.restore_artifacts = {fname: os.path.join(self.opts['restore'], fname) for fname in os.listdir(self.opts['restore']) if fname.endswith('.pth')}
                self.restore_artifacts['artifacts_dir'] = path
                path = self.opts['restore']
                print(f'restore from directory {path}')

            else:
                #  restore from exp_name:run_name
                self.restore_artifacts = self.logger.load_artifact(name_str=self.opts['restore'])
                restore_opts = read_json(self.restore_artifacts['options.json'])
                print(f'restore from {self.opts["restore"]}')
        
            self.restore_opts(restore_opts)
        else:
            print('no restore')
    
    def setup_trainer(self):
        self.lossCollection = lossCollection(self.net, self.pde, self.opts['weights'])

        trainerClass = get_trainer(self.opts['traintype'])
        print(f'trainer: {trainerClass}')
        self.trainer = trainerClass(self.opts['train_opts'], self.net, self.pde, self.device, self.lossCollection, self.logger)
        
        if self.restore_artifacts:
            self.trainer.restore(self.restore_artifacts['artifacts_dir'])
        else:
            print('no restore trainer')
    
    def run(self):
        # training
        print_dict(self.opts)
        self.logger.log_options(self.opts)

        self.trainer.train()
        self.trainer.save()
    


if __name__ == "__main__":
    # test all component
    optobj = Options()
    optobj.parse_args(*sys.argv[1:])

    # set seed
    set_seed(optobj.opts['seed'])

    eng = Engine(optobj.opts)

    eng.setup_problem()
    eng.setup_network()
    eng.setup_logger()
    eng.setup_trainer()

    summary(eng.net)
    eng.run()