#!/usr/bin/env python
# define problems for PDE
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise

from BaseProblem import BaseProblem
from MatDataset import MatDataset

class PoissonProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs
        # default 1
        self.p = kwargs.get('p', 1)

        # D for generating data
        self.D = kwargs['D']

        # initial guess
        self.all_params_dict = {'D': kwargs['D0']}
        

        self.lambda_transform = lambda x, u, param: u * x * (1 - x)


    def residual(self, nn, x):
        def f(x):
            return -(torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
        x.requires_grad_(True)
        
        u_pred = nn(x, nn.pde_params_dict)
        u_x = torch.autograd.grad(u_pred, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = nn.params_expand['D'] * u_xx - f(x)

        return res, u_pred

    def u_exact(self, x, param:dict):
        return torch.sin(torch.pi * self.p * x) / param['D']

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.all_params_dict.items():
            print(f'{k} = {v}')
        print(f'p = {self.p}')  

    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = MatDataset()

        # residual col-pt (collocation point), no need for u
        dataset['x_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
        dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

        # data col-pt, for testing, use exact param
        dataset['x_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
        dataset['u_dat_test'] = self.u_exact(dataset['x_dat_test'], {'D': self.D})

        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['x_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)

        dataset['u_dat_train'] = self.u_exact(dataset['x_dat_train'], {'D': self.D})

        self.dataset = dataset

    def validate(self, nn):
        '''compute err '''
        with torch.no_grad():
            err = torch.abs(nn.pde_params_dict['D'] - self.D)
        return {'abserr': err}

    def setup_dataset(self, dsopt, noise_opt, device='cuda'):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)
        self.dataset.to_device(device)


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'poisson'

    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = PoissonProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])

    # save dataset
    fpath = os.path.join(optobj.opts['logger_opts']['save_dir'], 'dataset.mat')
    prob.dataset.save(fpath)


