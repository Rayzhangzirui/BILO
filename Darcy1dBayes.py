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
from DenseNet import DenseNet, ParamFunction


class GausianRandomField:
    # lmbd_i =  (i pi)^2 + tau^2 )^(-alpha)
    # f = sum_{i=1}^K a_i  sqrt(lmbd_i) sin(i pi x)

    def __init__(self, K = 10.0, alpha = 1.0, tau=3.0):
        self.K = K
        self.alpha = alpha
        self.tau = tau
    
    
    def f(self, a, x):
        # x is (N,1)
        # a is (K,)
        tau = self.tau
        K = self.K

        indices = torch.arange(1, K + 1, dtype=torch.float32, device=x.device)
        lmbd = (indices**2 * torch.pi**2 + tau**2)**(-self.alpha)
        k_pi_x = torch.pi * x @ indices.view(1,-1) # (N,K)
        coeff =  (torch.sqrt(lmbd) * a).view(1,-1) # (1,K)
        terms = torch.cos(k_pi_x) * coeff
        return terms.sum(dim=1, keepdim=True) # (N,1)

class FunctionExpansion(torch.nn.Module):
    # Function expansion for f(x) = sum_{i=1}^K a_i lmbd_i (i pi)^2 sin(i pi x)
    # 1og10 D(x) = f(x)
    def __init__(self, a, grf):

        super(FunctionExpansion, self).__init__()
        # Initialize a_i as nn.Parameter for gradient-based optimization
        self.a = torch.nn.Parameter(a)
        # self.register_parameter('a', param=torch.nn.Parameter(a))
        self.grf = grf

    def forward(self, x):
        f = self.grf.f(self.a, x)
        return torch.exp(f)


class Darcy1dBayes(BayesianProblem):
    # Poisson equation with Bayesian inference
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs

        self.use_exact_sol = False
        self.usegrf = self.opts['usegrf']

        self.testcase = self.opts['testcase']
        self.dataset = MatDataset(self.opts['datafile'])

        self.loss_dict['post_fun'] = self.nll_fun
        
        self.pde_params = ['D']

        self.lambda_transform = lambda x, u, param: u * x * (1 - x)
        self.setup_parameters(**self.opts)

        # self.f = lambda x: 10.0 * ( 1.0 + torch.where(x > 0.5, 1.0, 0.0) )
        self.f = lambda x: 10.0
    
    
    def residual(self, nn, x):
        x.requires_grad_(True)
        
        u = nn(x, nn.pde_params_dict)
        D = nn.params_expand['D']
        
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(D*u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = - u_xx - self.f(x)

        return res, u
    
    def nll_fun(self, net):
        # posterior of f(x)
        f_pred = net.pde_params_dict['D'](self.dataset['X_dat_train'])
        err = f_pred - self.dataset['D_dat_train']
        # negative log likelihood
        nll = 0.5 * torch.sum(err**2)
        return nll
    

    def print_info(self):
        # print info of pde
        # print all parameters
        pass

    @torch.no_grad()
    def validate(self, nn):
        '''l2 error with D'''
        D = nn.pde_params_dict['D'](self.dataset['X_dat_test'])
        D_exact = self.dataset['D_dat_test']
        D_err = torch.mean(torch.square(D - D_exact))
        D_max = torch.max(torch.abs(D - D_exact))
        return {'l2err': D_err.item(), 'maxerr': D_max.item()}

    def create_dataset_from_file(self, dsopt):
        # porcssed in numpy
        dataset = self.dataset

        dataset.to_np()

        self.grf = GausianRandomField(K=dataset['K'], alpha=dataset['alpha'], tau=dataset['tau'])

        i = self.testcase

        u = dataset['u'][i-1, :]
        u = u.reshape(-1,1)
        
        x = self.dataset['x']
        x = x.reshape(-1,1)
        
        D = dataset['D'][i-1, :]
        D = D.reshape(-1,1)

        self.dataset['X_dat_test'] = x
        self.dataset['u_dat_test'] = u
        self.dataset['D_dat_test'] = D
        self.dataset['X_res_test'] = x
        self.dataset['u_res_test'] = u
        self.dataset['D_res_test'] = D
        
        self.dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['X_res_test', 'D_res_test','u_res_test'], replace='_test')
        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['X_dat_test', 'D_dat_test','u_dat_test'], replace='_test', exclude_bd=dsopt['exclude_bd'])
    
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
        
        # initial guess of a
        if self.usegrf:
            # Infer device from grf which has already been moved to the correct device in setup_dataset
            grf_device = self.grf.lmbd.device
            a = torch.tensor(self.dataset['a'][self.testcase-1, :], dtype=torch.float32, device=grf_device)
            self.param_fun = FunctionExpansion(a, self.grf)
        else:
        
            self.param_fun = ParamFunction(fdepth=kwargs['fdepth'], fwidth=kwargs['fwidth'],
                                fsiren=kwargs['fsiren'],
                                activation=kwargs['activation'], output_activation=kwargs['output_activation'],
                                output_transform=lambda x, u: torch.exp(u))

        init_param = {'D': self.param_fun}
        net = DenseNet(**kwargs,
                        lambda_transform = self.lambda_transform,
                        all_params_dict = init_param,
                        trainable_param = self.opts['trainable_param'])
        return net

    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_dat_train']
        D = net.pde_params_dict['D'](x)
        
        return torch.mean(torch.square(D - self.dataset['D_dat_train']))

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        
        if self.is_sampling:
            self.plot_prediction(savedir, vname='u')
            self.plot_prediction(savedir, vname='D')
            self.plot_mean_std(savedir,'u')
            self.plot_mean_std(savedir,'D')
            self.plot_examples(savedir)
        else:
            self.plot_prediction(savedir, vname='u')
            self.plot_prediction(savedir, vname='D')
            self.plot_variation(savedir)
        

    

