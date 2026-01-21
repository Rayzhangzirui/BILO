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


class NonlinearPoisson(BayesianProblem):
    # bayesian PINN paper example 3.2.3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs
        
        self.use_exact_sol = False
        self.dataset = MatDataset(kwargs['datafile'])
        
        # D for generating data
        self.default_param = {'k': 0.5}
        self.pde_params = ['k']
        
        
        # boundary condition at [-0.7. 0.7] is sin(6x)^3
        x1 = -0.7
        y1 = np.sin(6*(-0.7))**3
        x2 = 0.7
        y2 = np.sin(6*0.7)**3
        slope = (y2 - y1) / (x2 - x1)
        self.lambda_transform = lambda x, u, param: u * (x-x1) * (x-x2) + y1 + slope * (x - x1)
        
        self.setup_parameters(**kwargs)

        self.u_true = lambda x: torch.sin(6*x)**3
        self.u_xx_true = lambda x: 216 * torch.sin(6*x) * torch.cos(6*x)**2 - 108 * torch.sin(6*x)**3
        self.f_true = lambda x: 0.01 * self.u_xx_true(x) + self.gt_param['k'] * torch.tanh(self.u_true(x))

    def residual(self, net, x):
        f = self.f_true(x)
        x.requires_grad_(True)
        
        u = net(x, net.pde_params_dict)
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = 0.01 * u_xx + net.params_expand['k'] * torch.tanh(u) - f

        return res, u

    def print_info(self):
        # print info of pde
        # print all parameters
        pass

    @torch.no_grad()
    def validate(self, net):
        tmp = {'k': net.pde_params_dict['k']}
        if self.is_sampling:
            mean = self.estimator.get_mean('k')
            std = torch.sqrt(self.estimator.get_population_variance('k'))
            tmp['k_mean'] = mean
            tmp['k_std'] = std
        
        return tmp

    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        ''' downsample for training'''
        
        self.create_dataset_from_file(ds_opts)
        self.dataset.to_torch()

        if noise_opts['use_noise']:
            add_noise(self.dataset, noise_opts, x_name='X_dat_train', u_name='u_dat_train')

        # synthetic noise for residual
        # iid gaussian 
        self.dataset['res_nz'] = torch.randn_like(self.dataset['u_res_train']) * noise_opts['res_std']
        
        self.dataset.to_device(device)
    
    
    
    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        net = DenseNet(**kwargs,
                        lambda_transform = self.lambda_transform,
                        all_params_dict = self.init_param,
                        trainable_param = self.opts['trainable_param'])

        return net

    def create_dataset_from_file(self, dsopt):
        # porcssed in numpy
        dataset = self.dataset

        dataset.to_np()
        k_gt = self.gt_param['k'].item()

        # find k_gt in dataset['kgrd']
        # dataset['ugrid'] is a (n_k, n_x) matrix
        idx = np.argmin(np.abs(dataset['kgrid'] - k_gt))
        u = dataset['ugrid'][idx, :]
        u = u.reshape(-1,1)
        x = self.dataset['x']
        x = x.reshape(-1,1)

        self.dataset['X_dat_test'] = x
        self.dataset['u_dat_test'] = u
        self.dataset['X_res_test'] = x
        self.dataset['u_res_test'] = u
        
        self.dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['X_res_test', 'u_res_test'], replace='_test')
        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['X_dat_test', 'u_dat_test'], replace='_test', exclude_bd=dsopt['exclude_bd'])

    def nll_prior_pde(self, nn):
        '''P(\Theta), prior of PDE parameter, log(k) ~ N(0, 1)'''
        
        k = nn.pde_params_dict['k'].squeeze()
        nll = -0.5 * (torch.log(k) ** 2)
        return nll

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        if self.is_sampling:
            self.plot_mean_std(savedir)
            self.visualize_distribution(savedir)
            self.plot_examples(savedir)
        else:   
            self.plot_prediction(savedir)
            self.plot_variation(savedir)

