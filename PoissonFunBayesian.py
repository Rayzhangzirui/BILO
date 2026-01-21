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


class ExactSolution:
    # - u'' = f
    # f = sum_{i=1}^K a_i lmbd_i sin(i pi x)
    # u = sum_{i=1}^K a_i lmbd_i (i pi)^-2 sin(i pi x)
    # lmbd_i = 1/sqrt( (i pi)^2 + tau^2 )
    tau = 3

    @staticmethod
    def u(a, x):
        #  a is 1-d coefficient, x is n-by-1, u is n-by-1
        tau = ExactSolution.tau
        K = len(a)
        indices = torch.arange(1, K + 1, dtype=torch.float32, device=x.device) # shape (K,)
        lmbd = 1 / torch.sqrt(indices**2 * torch.pi**2 + tau**2) # shape (K,)
        sin_args = torch.pi * x @ indices.view(1,-1) # shape (n, K)
        coeff = (lmbd * a * (indices*torch.pi)**(-2)).view(1,-1) # shape (n, K)
        terms = torch.sin(sin_args) * coeff
        return terms.sum(dim=1, keepdim=True)
    
    @staticmethod
    def f(a, x):
        tau = ExactSolution.tau
        K = len(a)
        indices = torch.arange(1, K + 1, dtype=torch.float32, device=x.device)
        lmbd = 1 / torch.sqrt(indices**2 * torch.pi**2 + tau**2)
        sin_args = torch.pi * x @ indices.view(1,-1)
        coeff =  (lmbd * a).view(1,-1)
        terms = torch.sin(sin_args) * coeff
        return terms.sum(dim=1, keepdim=True)
    
class ExactSolutionNet(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, pde_params_dict:dict):
        f = pde_params_dict['f']
        return ExactSolution.u(f.a, x)
        


class FunctionExpansion(torch.nn.Module):
    # Function expansion for f(x) = sum_{i=1}^K a_i lmbd_i (i pi)^2 sin(i pi x)
    def __init__(self, a):

        super(FunctionExpansion, self).__init__()
        # Initialize a_i as nn.Parameter for gradient-based optimization
        self.register_parameter('a', param=torch.nn.Parameter(a))

    def forward(self, x):
        return ExactSolution.f(self.a, x)


class PoissonFunBayesian(BayesianProblem):
    # Poisson equation with Bayesian inference
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.opts=kwargs
        # default 1
        self.use_exact_sol = kwargs.get('use_exact_sol', False)
        
        self.loss_dict['post_fun'] = self.nll_fun
        self.loss_dict['prior_fun'] = self.nll_prior_fun
        
        self.pde_params = ['f']
        n = self.opts['field_dim']
        self.gt_a = torch.zeros(n, dtype=torch.float32)
        self.gt_a[0] = 1

        self.lambda_transform = lambda x, u, param: u * x * (1 - x)
        self.setup_parameters(**kwargs)

    def u_exact(self, x, a):
        # exact solution
        return ExactSolution.u(a, x)
    
    def f_exact(self, x, a):
        # exact solution
        return ExactSolution.f(a, x)

    def residual(self, nn, x):
        x.requires_grad_(True)
        
        u = nn(x, nn.pde_params_dict)
        f = nn.params_expand['f']
        
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = - u_xx - f

        return res, u
    
    def nll_fun(self, net):
        # posterior of f(x)
        f_pred = net.pde_params_dict['f'](self.dataset['X_dat_train'])
        err = f_pred - self.dataset['f_dat_train']
        # negative log likelihood
        nll = 0.5 * torch.sum(err**2)
        return nll
    
    def nll_prior_fun(self, nn):
        
        all_weight = nn.pde_params_dict['f'].a
        nll = torch.sum(0.5 * (all_weight**2))
        return nll


    def print_info(self):
        # print info of pde
        # print all parameters
        pass
    
    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = MatDataset()

        if dsopt['exclude_bd'] == False:
            # residual col-pt (collocation point), no need for u
            dataset['X_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
            # data col-pt, for testing, use exact param
            dataset['X_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)
        
        else:
            # exlcude pts near boundary
            dataset['X_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] + 2)[1:-1].view(-1, 1)
            dataset['X_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train'] + 2)[1:-1].view(-1, 1)

        dataset['X_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)
        dataset['X_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)

        dataset['u_dat_test'] = self.u_exact(dataset['X_dat_test'], self.gt_a)        
        dataset['f_res_train'] = self.f_exact(dataset['X_res_train'], self.gt_a)
        dataset['f_dat_test'] = self.f_exact(dataset['X_dat_test'], self.gt_a)

        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['u_dat_train'] = self.u_exact(dataset['X_dat_train'], self.gt_a)
        dataset['f_dat_train'] = self.f_exact(dataset['X_dat_train'], self.gt_a)

        self.dataset = dataset

    @torch.no_grad()
    def validate(self, nn):
        '''output a1 and a2'''
        n = nn.pde_params_dict['f'].a.shape[0]
        if n == 1:
            return {'a1': nn.pde_params_dict['f'].a[0].item()}
        else:
            return {'a1': nn.pde_params_dict['f'].a[0].item(), 'a2': nn.pde_params_dict['f'].a[1].item()}

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
        self.param_fun = FunctionExpansion(self.gt_a)
        init_param = {'f': self.param_fun}


        if self.use_exact_sol:
            net = ExactSolutionNet(**kwargs,
                                lambda_transform = self.lambda_transform,
                                all_params_dict = init_param,
                                trainable_param = self.opts['trainable_param'])
            net.param_net = []
        else:
            net = DenseNet(**kwargs,
                            lambda_transform = self.lambda_transform,
                            all_params_dict = init_param,
                            trainable_param = self.opts['trainable_param'])

        return net

    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_dat_train']
        f = net.pde_params_dict['f'](x)
        return torch.mean(torch.square(f - self.dataset['f_dat_train']))
    
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        
        # get mean and variance of function
        self.dataset['u_pred_mean'] = self.estimator.get_mean('u_pred')
        self.dataset['u_pred_var'] = self.estimator.get_population_variance('u_pred')


        # get mean and variance of function
        self.dataset['f_pred_mean'] = self.estimator.get_mean('f_pred')
        self.dataset['f_pred_var'] = self.estimator.get_population_variance('f_pred')

        # get final prediction
        self.dataset['upred_res_test'] = net(self.dataset['X_res_test'], net.pde_params_dict)
        self.dataset['upred_dat_test'] = net(self.dataset['X_dat_test'], net.pde_params_dict)

        self.dataset['fpred_dat_test'] = net.pde_params_dict['f'](self.dataset['X_dat_test'])
        
        # save solution example
        if 'u' in self.u_samples:
            self.dataset['eg_u'] = self.u_samples['u']
            self.dataset['eg_f'] = self.u_samples['f']

        if hasattr(self, 'u_exact'):
            self.dataset['uinf_dat_test'] = self.u_exact(self.dataset['X_dat_test'], net.pde_params_dict['f'].a)

    def plot_pred_f(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['X_dat_test'], self.dataset['f_dat_test'], label='Exact')
        ax.plot(self.dataset['X_dat_test'], self.dataset['fpred_dat_test'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_f_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
            plt.close(fig)

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_prediction(savedir)
        self.plot_pred_f(savedir)
        self.plot_variation(savedir)

        # if self.hist and self.hist['D']:
            # self.hist['D'] might be empty if not sampling
        self.plot_mean_std(savedir,'u')
        self.plot_mean_std(savedir,'f')

        # self.visualize_distribution(savedir)
        self.plot_examples(savedir)

    

