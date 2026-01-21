#!/usr/bin/env python
# PoissonProblem with variable parameter
import torch
import torch.nn as nn
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise, error_logging_decorator

from BaseProblem import BaseProblem
from BayesianProblem import BayesianProblem
from MatDataset import MatDataset

from DenseNet import DenseNet, ParamFunction

from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class VarPoiBayesProblem(BayesianProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs        
                                
        self.testcase = kwargs['testcase']
        # no exact solution
        self.use_exact_sol = False
        self.loss_dict['post_data'] = self.nll_data
        self.loss_dict['prior_fun'] = self.nll_prior_fun
        self.loss_dict['prior_fun_laplace'] = self.nll_prior_fun_laplace
        self.loss_dict['prior_fun_tv'] = self.nll_prior_fun_tv

        self.lambda_transform = lambda x, u, param: u * x * (1.0 - x)

        self.dataset = None
        self.dataset = MatDataset(kwargs['datafile'])

    def residual(self, nn, x):
        
        x.requires_grad_(True)
        
        u = nn(x, nn.pde_params_dict)
        D = nn.params_expand['D']

        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dxDux = torch.autograd.grad(D*u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = dxDux - self.dataset['f_res_train']
        return res, u

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        self.param_fun = ParamFunction(fdepth=kwargs['fdepth'], fwidth=kwargs['fwidth'],
                                    fsiren=kwargs['fsiren'],
                                    ffourier=kwargs['ffourier'],
                                    activation=kwargs['activation'], output_activation=kwargs['output_activation'],
                                    output_transform=lambda x, u: u * x * (1.0 - x) + 1.0 )
                
        self.all_params_dict = {'D': self.param_fun}

        net = DenseNet(**kwargs,
                        lambda_transform=self.lambda_transform,
                        all_params_dict= self.all_params_dict,
                        trainable_param = self.opts['trainable_param'])
        # net.setup_embedding_layers()
        return net
    
    def nll_data(self, nn):
        '''compute negative log likelihood of data
        P(Data|Theta) = product of 1/sqrt(2*pi*sigma^2) * exp(-0.5*(u_pred - u_data)^2/sigma^2)
        -log(P(Data|Theta)) = 0.5 * sum((u_pred - u_data)^2/sigma^2) + 0.5 * N * log(2*pi*sigma^2)
        Ignore the last term as it is independent of theta

        sigma is not included here. Obtained from "weight" when computing the total loss
        '''
        u_pred = nn(self.dataset['X_dat_train'], nn.pde_params_dict)
        err = u_pred - self.dataset['u_dat_train']
        # negative log likelihood
        nll = 0.5 * torch.sum(err**2)
        return nll
    
    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['X_res_train'])
        return res, pred
        
    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'], net.pde_params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))        
        return loss
    
    def create_dataset_from_file(self, dsopt):
        '''create dataset from file'''
        assert self.dataset is not None, 'datafile provide, dataset should not be None'
        uname = f'u{self.testcase}'
        dname = f'd{self.testcase}'

    
        self.dataset['x_dat'] = self.dataset['x']
        self.dataset['X_dat_test'] = self.dataset['x']
        self.dataset['u_dat'] = self.dataset[uname]
        self.dataset['u_dat_test'] = self.dataset[uname]
        self.dataset['D_dat'] = self.dataset[dname]
        self.dataset['D_dat_test'] = self.dataset[dname]

        self.dataset['x_res'] = self.dataset['x']
        self.dataset['X_res_test'] = self.dataset['x']
        self.dataset['f_res'] = self.dataset['f']
        self.dataset['D_res'] = self.dataset[dname]
        
        self.dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['x_res', 'D_res', 'f_res'])
        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['x_dat', 'u_dat', 'D_dat'])

    
    def setup_dataset(self, dsopt, noise_opts=None, device='cuda'):
        '''add noise to dataset'''
        self.create_dataset_from_file(dsopt)
        if noise_opts['use_noise']:
            add_noise(self.dataset, noise_opts)
        
        self.dataset.to_device(device)
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_res_train']
        y = net.func_param(x)
        return torch.mean(torch.square(y - self.dataset['D_res_train']))

    def nll_prior_fun(self, net):
        
        all_weight = parameters_to_vector(net.pde_params_dict['D'].parameters())
        nll = torch.sum(0.5 * (all_weight**2))
        return nll
    
    def nll_prior_fun_tv(self, net):
        # total variation prior
        # D(x) ~ exp(- w int_0^1 |D'(x)|^2 dx)
        x = self.dataset['X_res_train'].requires_grad_(True)
        D = net.pde_params_dict['D'](x)
        D_x = torch.autograd.grad(D, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(D))[0]
        tv = torch.mean(D_x**2)
        return tv
    
    def nll_prior_fun_laplace(self, nn):
        '''laplace prior'''
        all_weight = parameters_to_vector(nn.pde_params_dict['D'].parameters())
        nll = torch.sum(torch.abs(all_weight))
        return nll
    


    @torch.no_grad()
    def validate(self, nn):
        '''compute l2 error and linf error of inferred D(x)'''
        x  = self.dataset['x_dat']
        D = self.dataset['D_dat']
        
        Dpred = nn.pde_params_dict['D'](x)
        l2norm = torch.mean(torch.square(D - Dpred))
        linfnorm = torch.max(torch.abs(D - Dpred)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def plot_upred(self, savedir=None):
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['u_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['upred_dat_test'], label='NN')
        ax.scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_Dpred(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['D_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['Dpred_dat_test'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_D_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_MAP(self, savedir=None):
        '''plot MAP solution'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['u_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['MAP_u'], label='MAP')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_MAP.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_dat'], self.dataset['D_dat'], label='Exact')
        ax.plot(self.dataset['x_dat'], self.dataset['MAP_D'], label='MAP')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_MAP_D.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.dataset.to_np()
        if self.is_sampling:
            self.plot_mean_std(savedir,'u')
            self.plot_mean_std(savedir,'D')
            self.plot_examples(savedir)
            self.plot_MAP(savedir)
        else:
            self.plot_upred(savedir)
            self.plot_Dpred(savedir)

        