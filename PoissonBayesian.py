#!/usr/bin/env python
# define problems for PDE
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise, error_logging_decorator

from BayesianProblem import BayesianProblem
from MatDataset import MatDataset
from DenseNet import DenseNet


class ExactSolution(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, pde_params_dict:dict):
        u = torch.sin(torch.pi * x) / pde_params_dict['D']
        return u

class PoissonBayesian(BayesianProblem):
    # Poisson equation with Bayesian inference
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs
        # default 1
        self.p = kwargs.get('p', 1)

        self.use_exact_sol = kwargs.get('use_exact_sol', False)
        


        # D for generating data
        self.default_param = {'D': 1}
        self.pde_params = ['D']
        
        self.lambda_transform = lambda x, u, param: u * x * (1 - x)
        

        self.setup_parameters(**kwargs)


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
        pass
    
    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = MatDataset()

        # residual col-pt (collocation point), no need for u
        dataset['X_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
        dataset['X_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

        # data col-pt, for testing, use exact param
        dataset['X_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
        dataset['u_dat_test'] = self.u_exact(dataset['X_dat_test'], {'D': self.gt_param['D']})

        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['X_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)
        dataset['u_dat_train'] = self.u_exact(dataset['X_dat_train'], {'D': self.gt_param['D']})

        self.dataset = dataset

    def validate(self, nn):
        '''compute err '''
        # with torch.no_grad():
        #     err = torch.abs(nn.pde_params_dict['D'] - self.gt_param['D']).item()
        mean = self.estimator.get_mean('D')
        std = torch.sqrt(self.estimator.get_population_variance('D'))
        return {'Dmean': mean.item(), 'Dstd': std.item()}

    def setup_dataset(self, dsopt, noise_opt, device='cuda'):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)
        
        self.sigma = np.sqrt(noise_opt['std'])
        self.dataset.to_device(device)
    
    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim


        if self.use_exact_sol:
            net = ExactSolution(**kwargs,
                                lambda_transform = self.lambda_transform,
                                all_params_dict = self.init_param,
                                trainable_param = self.opts['trainable_param'])
            net.param_net = []
        else:
            net = DenseNet(**kwargs,
                            lambda_transform = self.lambda_transform,
                            all_params_dict = self.init_param,
                            trainable_param = self.opts['trainable_param'])

        return net

    def nll_prior_pde(self, nn):
        '''P(\Theta), prior of PDE parameter'''
        # use log normal
        # D = nn.pde_params_dict['D'].squeeze()
        # mu = 0
        # sigma = 0.25
        # P = torch.exp(-0.5*((torch.log(D) - mu)/sigma)**2)/(D*sigma*np.sqrt(2*np.pi))

        # uniform distribution in [a, b]
        # adhoc, the hard boundary need to be handled carefully in sampling algorithm
        D = nn.pde_params_dict['D'].squeeze()
        a = 0
        b = 4
        P = torch.where((D >= a) & (D <= b), torch.tensor(1.0/(b-a)), torch.tensor(0.0))
        nll = -torch.log(P)
        return nll

    

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_prediction(savedir)
        self.plot_variation(savedir)

        # if self.hist and self.hist['D']:
            # self.hist['D'] might be empty if not sampling
        self.plot_mean_std(savedir)
        self.visualize_distribution(savedir)
        self.plot_examples(savedir)

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

