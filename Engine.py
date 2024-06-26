#!/usr/bin/env python
from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
import os

import mlflow


from Options import *
from util import *


from MlflowHelper import load_artifact
from Problems import create_pde_problem
from lossCollection import lossCollection

from Logger import Logger
from Trainer import Trainer

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
        # setup pde problem
        self.pde = create_pde_problem(self.opts['pde_opts'])
        
        self.pde.setup_dataset(self.opts['dataset_opts'], self.opts['noise_opts'])
        
        self.net = self.pde.setup_network(**self.opts['nn_opts'])

    def restore_opts(self, restore_opts):
        ''' restore options from a previous run, and update with new options
        '''
        self.opts['nn_opts'].update(restore_opts['nn_opts'])
        
        

    def restore_run(self):
        # actual restore is called in setup_lossCollection, need to known collection of trainable parameters
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
        
            self.restore_opts(restore_opts)
    
        

    def setup_trainer(self):
        self.lossCollection = lossCollection(self.net, self.pde, self.opts['weights'], self.opts['loss_opts'])
        self.trainer = Trainer(self.opts['train_opts'], self.net, self.pde, self.device, self.lossCollection, self.logger)
        self.trainer.config_train(self.opts['traintype'], self.opts['scheduler_opts'])

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