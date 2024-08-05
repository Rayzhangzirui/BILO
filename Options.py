#!/usr/bin/env python
import sys
import json
from BaseOption import BaseOptions

from MlflowHelper import MlflowHelper


default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test'},
    'restore': '',
    'traintype': 'vanilla-inv',
    'flags': '', 
    'device': 'cuda',
    'seed': 0,
    'pde_opts': {
        'problem': 'simpleode',
        'exact_param': None, # used for poisson problem to define exact parameter of pde, for generating training data.
        'trainable_param': '', # list of trainable parameters, e.g. 'D,rho'
        'init_param': '', # nn initial parameter as string, e.g. 'D,1.0'
        'datafile': '',
        'dat_use_res': False, # used in fkproblem, use res as training data
        'testcase': 0,
        # for heat problem 0.1 and poisson problem
        'D': 0.1,
        'use_exact_u0':False,
        'D0': 1.0,
        # for scalar poisson
        'p': 1,
        # For GBM problem
        'whichdata':'ugt_dat', # which data to use for data loss
    },
    'nn_opts': {
        'depth': 4,
        'width': 64,
        # 'input_dim': 1,
        # 'output_dim': 1,
        'use_resnet': False,
        'with_param': True,
        'fourier':False,
        'siren': False,
        'with_func': False,
    },
    'func_opts': {
        'fdepth': 4,
        'fwidth': 8,
        'activation': 'tanh',
        'output_activation': 'softplus',
        'fsiren': False,
    },
    'scheduler_opts': {
        'scheduler': 'constant',
    },
    'dataset_opts': {
        'N_res_train': 101,
        'N_res_test': 101,
        'N_dat_train': 101,
        'N_dat_test': 101,

        # for heat problem
        'N_ic_train':101, # point for evaluating l2grad
        
        # for gbm problem
        'N_bc_train': 101,

        'Nx':51,
        'Nt':51,
    },
    'train_opts': {
        'print_every': 20,
        'max_iter': 100000,
        'burnin':10000,
        'tolerance': 1e-6, # stop if loss < tolerance
        'patience': 1000,
        'delta_loss':1e-5, # stop if no improvement in delta_loss in patience steps
        'monitor_loss':True,
        'lr': 1e-3,
        # for simu training
        'lr_net': 1e-3,
        'lr_pde': 1e-3,
        # for bi-level training
        'tol_lower': 1e-3, # lower level tol
        'max_iter_lower':1000,
        'loss_net':'res,fullresgrad,bc', # loss for network weights
        'loss_pde':'data,l2grad,l1grad', # loss for pde parameter
        'reset_optim':True, # reset optimizer state
        'whichoptim':'adam'
    },
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
    'weights': {
        'res': 1.0,
        'fullresgrad': 0.001,
        'resgradfunc': None,
        'data': 1.0,
        'bc':None,
        'funcloss':None, #mse of unknonw function
        'l2grad':None,
        'l1grad':None,
    },
    'loss_opts': {
        'msample':100, #number of samples for resgrad
    }
}



class Options(BaseOptions):
    def __init__(self):
        self.opts = default_opts
    
    
    def process_flags(self):

        if self.opts['flags'] != '':
            self.opts['flags'] = self.opts['flags'].split(',')
            assert all([flag in ['small','local','post','fixiter','lintest'] for flag in self.opts['flags']]), 'invalid flag'
        else:
            self.opts['flags'] = []

        if 'small' in self.opts['flags']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
        
        if 'lintest' in self.opts['flags']:
            # use small network (linear function) for testing
            self.opts['nn_opts']['depth'] = 0
            self.opts['nn_opts']['width'] = 1
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
            
        if 'local' in self.opts['flags']:
            # use local logger
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            self.opts['logger_opts']['use_csv'] = False
        
        if 'fixiter' in self.opts['flags']:
            # fix number of iterations, do not use early stopping
            self.opts['train_opts']['burnin'] = self.opts['train_opts']['max_iter']
        
        if 'post' in self.opts['flags']:
            # post process only
            self.opts['train_opts']['max_iter'] = 0
            self.opts['train_opts']['burnin'] = 0
            # do not use mlflow, use stdout
            # mlrun is created when logger is initialized, so set use_mlflow to False
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            # parse resotre expname:runname
            restore = self.opts['restore'].split(':')
            expname = restore[0]
            runname = restore[1]
            # get artifact path from mlflow, set save_dir
            helper = MlflowHelper()
            run_id = helper.get_id_by_name(expname, runname)
            paths = helper.get_artifact_dict_by_id(run_id)
            self.opts['logger_opts']['save_dir'] = paths['artifacts_dir']
            
    
    def process_problem(self):
        ''' handle problem specific options '''
        
        if self.opts['pde_opts']['problem'] in {'poisson','poisson2'}:
            self.opts['pde_opts']['trainable_param'] = 'D'
        else:
            # remove D0 key
            self.opts['pde_opts'].pop('D0', None)
        
        if self.opts['pde_opts']['problem'] in {'poivar','heat','burger','varfk','darcy'}:
            # merge func_opts to nn_opts, use function embedding
            self.opts['nn_opts'].update(self.opts['func_opts'])
            self.opts['nn_opts']['with_func'] = True


        else:
            # for scalar problem, can not use l2reg
            self.opts['weights']['l2grad'] =  None
            self.opts['nn_opts']['with_func'] = False
        

        # Need to specify trainable_param, which is used in fullresgrad loss
        if self.opts['pde_opts']['problem'] in {'heat','burger'}:
            self.opts['pde_opts']['trainable_param'] = 'u0'
        
        if self.opts['pde_opts']['problem'] in {'poivar'}:
            self.opts['pde_opts']['trainable_param'] = 'D'
        
        if self.opts['pde_opts']['problem'] in {'darcy'}:
            self.opts['pde_opts']['trainable_param'] = 'f'
        
        del self.opts['func_opts']

    def processing(self):
        ''' handle dependent options '''
        
        
        self.process_flags()

        self.process_problem()

        # training type 
        # vanilla-fwd, vanilla-inv
        # adj-fwd, adj-inv
        # for vanilla PINN, nn does not include parameter
        assert self.opts['traintype'] in {'vanilla-inv','vanilla-init','adj-init', 'adj-simu', 'adj-bi1opt'}, 'invalid traintype'
        
        if self.opts['traintype'].startswith('vanilla'):
            self.opts['weights']['fullresgrad'] = None
            self.opts['nn_opts']['with_param'] = False

            if self.opts['traintype'].endswith('init'):
                # for vanilla training, all parameters are states in optimizer
                # for init, require_grad is false,
                # for inv, some require_grad is true
                self.opts['pde_opts']['trainable_param'] = ''
            

        if self.opts['traintype'].startswith('adj'):
            # if not vanilla PINN, nn include parameter
            self.opts['nn_opts']['with_param'] = True

        # for init of both vanilla and adj
        if self.opts['traintype'].endswith('init'):
            if self.opts['nn_opts']['with_func']:
                # use function embedding, use mse of function as loss to train param_func
                # these set available loss, actuall loss is determined by weights
                
                # first of all, both adj and van need funcloss
                self.opts['weights']['funcloss'] = 1.0
                
                # For adj, need the following 
                self.opts['train_opts']['loss_net'] = 'res,fullresgrad,data,bc'
                self.opts['train_opts']['loss_pde']= 'funcloss'
                
                # for init, no matter adj or van, need lr of pde to fit funcloss
                self.opts['train_opts']['lr_pde'] = 1e-3
            else:
                # for scalar problem
                # these set available loss, actuall loss is determined by weights
                # for init, loss_pde lr is 0.0
                self.opts['train_opts']['loss_net'] = 'res,fullresgrad,data,bc'
                self.opts['train_opts']['loss_pde'] = 'data'
                self.opts['weights']['funcloss'] = None

                # for scaler problem, set lr of pde param to 0.0
                self.opts['train_opts']['lr_pde'] = 0.0

        # convert to list of losses
        self.opts['train_opts']['loss_net'] = self.opts['train_opts']['loss_net'].split(',')
        self.opts['train_opts']['loss_pde'] = self.opts['train_opts']['loss_pde'].split(',')
        # remove inative losses (weight is None)
        self.opts['train_opts']['loss_net'] = [loss for loss in self.opts['train_opts']['loss_net'] if self.opts['weights'][loss] is not None]
        self.opts['train_opts']['loss_pde'] = [loss for loss in self.opts['train_opts']['loss_pde'] if self.opts['weights'][loss] is not None]
        

        
        # After traintype is processed 
        # convert trainable param to list of string, split by ','
        if self.opts['pde_opts']['trainable_param'] != '':
            self.opts['pde_opts']['trainable_param'] = self.opts['pde_opts']['trainable_param'].split(',')
        else:
            self.opts['pde_opts']['trainable_param'] = []
        
        if self.opts['pde_opts']['init_param'] != '':
            self.opts['pde_opts']['init_param'] = self.convert_to_dict(self.opts['pde_opts']['init_param'])




if __name__ == "__main__":
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))