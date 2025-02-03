#!/usr/bin/env python
import sys
import json
from BaseOption import BaseOptions
from MlflowHelper import MlflowHelper


gbm_options ={
    # For GBM problem
        'whichdata':'ugt_dat', # which data to use for data loss
        'usewdat':False, # use weight for data loss
        'combinedat':False, # combine X_dat for residual
        'pos_trans':False, # positive transformation
        'force_bc':True, # force boundary condition
        'th1_range':[0.2,0.5], # range of theta1
        'th2_range':[0.5,0.8], # range of theta2
}

# for BilO and PINN
default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test'},
    'restore': '',
    'traintype': 'pinn-inv',
    'flags': '', 
    'device': 'cuda',
    'seed': 0,
    'pde_opts': {
        'problem': 'simpleode',
        'exact_param': None, # used for poisson problem to define exact parameter of pde, for generating training data.
        'trainable_param': '', # list of trainable parameters, e.g. 'D,rho'
        'init_param': '', # nn initial parameter as string, e.g. 'D,1.0'
        'datafile': '',
        'dat_use_res': False, # used in fkproblem and heatproblem, use res as training data
        'testcase': 0,
        # for heat problem 0.1 and poisson problem
        'use_exact_u0':False,
        'D0': 1.0,
        # for scalar poisson
        'p': 1,
        # For poissonhyper
        'theta': [0.0,1.0],

        'gt_param':'', # ground truth parameter to generate data as string, e.g. 'D,0.1'

        # for point process problem
        'gamma': [1.0,0.004], #alpha and beta for gamma prior
        'uniform':[0.0,30.0], # range of uniform prior
        'n_snapshot': 100, 
        'use_exact_sol': False, # use exact solution as nerual network
        'use_simpson': True, # use simpson rule to compute integral
        'lower_bound': '',
        'upper_bound': '',
    },
    'nn_opts': {
        'depth': 4,
        'width': 64,
        # 'input_dim': 1,
        # 'output_dim': 1,
        'use_resnet': False,
        'with_param': True,
        'fourier':False,
        'rbf':False,
        'skip_param': True,
        'siren': False,
        'with_func': False,
        'omega0': 30.0,
        'sigma': 1.0,
        'modifiedmlp': False,
        # if reload the following options, reset_optim must be true,
        # otherwise,  ValueError("loaded state dict contains a parameter group")
        'rff_trainable':False,
        'train_embed': False,   
        # LoRA
        'rank': 0,
        'lora_alpha':1.0,
    },
    'func_opts': {
        'fdepth': 4,
        'fwidth': 8,
        'activation': 'tanh',
        'output_activation': 'id',
        'fsiren': False,
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
        # for fk problem
        'Nx_train':51,
        'Nt_train':51,
        # batch size for lower/net
        'net_batch_size': 100000,
        # batch size for upper/pde
        'pde_batch_size': 100000,
        'exclude_bc': False,
    },
    'train_opts': {
        'print_every': 10,
        'max_iter': 10000,
        'burnin':0,
        # for early stopping
        'tolerance': -1e9, # stop if loss < tolerance, choose large negative number because neg_likelihood might be negative
        'patience': 1000,
        'delta_loss': 0.0, # stop if no improvement in delta_loss in patience steps
        'monitor_loss':False,
        # for simu training
        'lr_net': 1e-3,
        'lr_pde': 1e-3,
        # for bi-level training
        'tol_lower': 1e9, # lower level tol
        
        'max_iter_lower':1000,
        'loss_net': '', # loss for network weights
        'loss_pde': '', # loss for pde parameter
        'loss_test': '', # loss for testing
        'loss_monitor': '', # loss for monitoring
        'reset_optim':True, # reset optimizer state
        'simu_update': True, #simultaneous update of network and pde
        # string for optimizer name
        'optim_net':'Adam',
        'optim_pde':'Adam',
        # string for optimizer options
        'opts_net':'amsgrad,True',
        'opts_pde':'amsgrad,True',
        # scheduler options
        'sch_net':'ExponentialLR',
        'sch_pde':'ExponentialLR',
        'schopt_net':'gamma,1.0',
        'schopt_pde':'gamma,1.0',
        'acc_iter': 1, # accumulate n-iteration of gradient
        # sampling options
        'backtrack': False, # backtrack LoRA weights for rejected proposal
        'merge': False, # merge LoRA weights for accepted proposal
        # HMC options
        'lf_steps': 10, # number of steps for LeapFrog
        'random_L': False, # make lf_steps random
        'example_every': 0, # for collecting pde solution samples during sampling
        'adapt_M': False, # adapt mass matrix
        'refresh': False, # refresh the optimizer
    },
    'noise_opts':{
        'use_noise': False,
        'std': 0.1,
        'length_scale': 0.0,
    },
    'weight_as_sigma': 'post_res,post_data,prior_weight,post_fun,prior_fun', # list of weights treated as sigma in the negative log likelihood
    'weights': ''
}


# For operator learning
op_default_opts = default_opts.copy()
op_default_opts['traintype'] = 'deeponet-init'

op_default_opts['nn_opts'] = {
        'arch': 'deeponet',
        # for DeepONet
        'branch_depth': 4,
        'trunk_depth': 4,
        'width': 64,
        # for FNO
        'n_modes': 32,
        'n_layers': 4,
        'hidden_channels': 32,
    }
op_default_opts['pde_opts']= {
        'datafile': '',
        'problem': 'fkop',
        'testcase': 1,
        # dimension of parameter, for scalar, same as number of scalar parameters
        # for function, same as discretization of function
        'param_dim': 1,
        "dat_use_res": False,
    }

op_default_opts['train_opts']= {
        'print_every': 20,
        'max_iter': 10000,
        'burnin':1000,
        'tolerance': 1e-6, # stop if loss < tolerance
        'patience': 1000,
        'reset_optim':True,
        'lr': 1e-3,
        'loss_net': ['data'], # loss for network weights
        'loss_test': [], # loss for testing
        'loss_pde':[],
        'loss_monitor': [],
        'optim':'Adam',
        'opts':{},
        'sch':'ExponentialLR',
        'schopt':{'gamma':1.0},
    }

op_default_opts['dataset_opts']= {
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
        # for fk problem
        'Nx_train':11,
        'Nt_train':11,

        'batch_size': 1000,
        'split': 0.9,
    }

op_default_opts['weights']= ''



class Options(BaseOptions):
    def __init__(self):
        self.opts = default_opts
        # is it operator learning
        self.oplearn = False
    

    def parse_args(self, *args):
        # first parse args and update dictionary
        # then process dependent options
        self.preprocess_traintype(*args)
        self.parse_nest_args(*args)
        
        self.process_problem()
        if self.oplearn:
            self.process_oplearn()
        else:
            self.process_traintype()
        
        self.process_flags()
    
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
        
        # if self.opts['pde_opts']['problem'] in {'poisson','poisson2','poissonbayesian'}:
            # self.opts['pde_opts']['trainable_param'] = 'D'
    
        
        if self.opts['pde_opts']['problem'] in {'poivar','heat','burger','varfk','darcy'}:
            # merge func_opts to nn_opts, use function embedding
            self.opts['nn_opts'].update(self.opts['func_opts'])
            self.opts['nn_opts']['with_func'] = True

        else:
            self.opts['nn_opts']['with_func'] = False
        
        # include gbm options
        if self.opts['pde_opts']['problem'] == 'gbm':
            self.opts['pde_opts'].update(gbm_options)

        # Need to specify trainable_param, which is used in fullresgrad loss
        if self.opts['pde_opts']['problem'] in {'heat','burger'}:
            self.opts['pde_opts']['trainable_param'] = 'u0'
        
        if self.opts['pde_opts']['problem'] in {'poivar'}:
            self.opts['pde_opts']['trainable_param'] = 'D'
        
        if self.opts['pde_opts']['problem'] in {'darcy'}:
            self.opts['pde_opts']['trainable_param'] = 'f'

        # convert string to dict
        if self.opts['pde_opts']['problem'] in {'pointprocess'}:
            self.opts['pde_opts']['lower_bound'] = self.convert_to_dict(self.opts['pde_opts']['lower_bound'])
            self.opts['pde_opts']['upper_bound'] = self.convert_to_dict(self.opts['pde_opts']['upper_bound'])
        
        del self.opts['func_opts']
    
    def preprocess_traintype(self, *args):
        # find index of traintype
        i = 0
        while i < len(args):
            if args[i] == 'traintype':
                break
            i += 1
        traintype = args[i+1]

        # set default
        network, method = traintype.split('-')
        if network in {'fno','deeponet'}:
            self.opts = op_default_opts
            self.oplearn = True
        else:
            self.opts = default_opts
            self.oplearn = False
        
    def process_oplearn(self):
        # process options related to operator learning
        arch, stage = self.opts['traintype'].split('-')
        assert stage in {'init','inv'}, 'invalid traintype'
        assert arch in {'deeponet','fno'}, 'invalid traintype'

        self.opts['nn_opts']['arch'] = arch

        self.opts['weights'] = self.convert_to_dict(self.opts['weights'])
        
        # set default wegiths to be 1.0 if not specified
        for loss in self.opts['train_opts']['loss_net'] + self.opts['train_opts']['loss_pde'] + self.opts['train_opts']['loss_test'] + self.opts['train_opts']['loss_monitor']:
            if loss not in self.opts['weights']:
                self.opts['weights'][loss] = 1.0
                

    def process_traintype(self):
        # process options related to training
        # split traintype
        network, method = self.opts['traintype'].split('-')
        traintype = self.opts['traintype']

        assert network in {'pinn','bpinn','bilo','exact'}, 'invalid network'

        match network:
            case 'pinn':
                self.opts['nn_opts']['with_param'] = False
                assert method in {'fwd','inv','init'}, 'invalid method'
                
                # for initialization, scalar pde params are not trainable
                # but function params are trainable
                # if method == 'init':
                #     self.opts['pde_opts']['trainable_param'] = ''
                
            case 'bpinn':
                self.opts['nn_opts']['with_param'] = False
                # weight_as_sigma must be specified

                assert method in {'mala','hmc','sgld','psgld','map'}, 'invalid method'
            
            case 'bilo':
                self.opts['nn_opts']['with_param'] = True
                assert method in {'mala','hmc','sgld','psgld','init','inv','simu'}, 'invalid method'
            
            case 'exact':
                self.opts['pde_opts']['use_exact_sol'] = True
                assert method in {'mala','hmc','sgld','psgld','inv'}, 'invalid method'

        if method == 'sgld':
            # use noise for sgld
            self.opts['train_opts']['optim_net'] = 'SGLD'
        
        if method == 'psgld':
            # use noise for psgld
            self.opts['train_opts']['optim_net'] = 'pSGLD'

        # convert weight to dict
        self.opts['weights'] = self.convert_to_dict(self.opts['weights'])
        # convert weight_as_sigma to list
        self.opts['weight_as_sigma'] = self.opts['weight_as_sigma'].split(',')
        
        # for init of both pinn and adj
        # if method == 'init':
        #     if self.opts['nn_opts']['with_func']:
        #         # use function embedding, use mse of function as loss to train param_func
        #         assert 'funcloss' in self.opts['train_opts']['loss_pde'], 'funcloss must be in loss_pde'
                

        # convert loss_net, loss_pde, loss_test, loss_monitor to list of string
        self.opts['train_opts']['loss_net'] = self.opts['train_opts']['loss_net'].split(',') if self.opts['train_opts']['loss_net'] != '' else []
        # fullresgrad must follow res
        if 'fullresgrad' in self.opts['train_opts']['loss_net']:
            i = self.opts['train_opts']['loss_net'].index('fullresgrad')
            j = self.opts['train_opts']['loss_net'].index('res')
            assert i == j+1, 'fullresgrad must follow res'


        self.opts['train_opts']['loss_pde'] = self.opts['train_opts']['loss_pde'].split(',') if self.opts['train_opts']['loss_pde'] != '' else []
        self.opts['train_opts']['loss_test'] = self.opts['train_opts']['loss_test'].split(',') if self.opts['train_opts']['loss_test'] != '' else []
        self.opts['train_opts']['loss_monitor'] = self.opts['train_opts']['loss_monitor'].split(',') if self.opts['train_opts']['loss_monitor'] != '' else []

        # check: test loss should not be in loss_net or loss_pde
        for loss in self.opts['train_opts']['loss_test']:
            assert loss not in self.opts['train_opts']['loss_net'], f'{loss} should not be in loss_net'
            assert loss not in self.opts['train_opts']['loss_pde'], f'{loss} should not be in loss_pde'
        
        # set default wegiths to be 1.0 if not specified
        for loss in self.opts['train_opts']['loss_net'] + self.opts['train_opts']['loss_pde'] + self.opts['train_opts']['loss_test'] + self.opts['train_opts']['loss_monitor']:
            if loss not in self.opts['weights']:
                self.opts['weights'][loss] = 1.0

        # After traintype is processed 
        # convert trainable param to list of string, split by ','
        if self.opts['pde_opts']['trainable_param'] != '':
            self.opts['pde_opts']['trainable_param'] = self.opts['pde_opts']['trainable_param'].split(',')
        else:
            self.opts['pde_opts']['trainable_param'] = []
        
        self.opts['pde_opts']['gt_param'] = self.convert_to_dict(self.opts['pde_opts']['gt_param'])
        self.opts['pde_opts']['init_param'] = self.convert_to_dict(self.opts['pde_opts']['init_param'])
        
        # convert optimizer option to dict
        self.opts['train_opts']['opts_net'] = self.convert_to_dict(self.opts['train_opts']['opts_net'])
        self.opts['train_opts']['opts_pde'] = self.convert_to_dict(self.opts['train_opts']['opts_pde'])
        # if optim is not Adam, remove amsgrad
        if self.opts['train_opts']['optim_net'] != 'Adam':
            self.opts['train_opts']['opts_net'].pop('amsgrad',None)
        if self.opts['train_opts']['optim_pde'] != 'Adam':
            self.opts['train_opts']['opts_pde'].pop('amsgrad',None)

        # convert scheduler option to dict
        self.opts['train_opts']['schopt_net'] = self.convert_to_dict(self.opts['train_opts']['schopt_net'])
        self.opts['train_opts']['schopt_pde'] = self.convert_to_dict(self.opts['train_opts']['schopt_pde'])


    def print(self):
        print(json.dumps(self.opts, indent=2, sort_keys=True))



if __name__ == "__main__":

    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))