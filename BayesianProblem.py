#!/usr/bin/env python
# define problems for PDE
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from util import generate_grf, add_noise, error_logging_decorator
from BaseProblem import BaseProblem
from typing import Dict, Any

from arviz import ess

class WelfordEstimator:
    def __init__(self, burnin: int = 0):
        """
        burnin: Number of initial updates to ignore (for each field).
        """
        self.burnin = burnin
        # Dictionary mapping a string key -> dict with Welford stats
        # for that quantity/field.
        # Each entry has:
        #   {
        #     'n': int,
        #     'mean': Tensor or None,
        #     'M2': Tensor or None,
        #     'total_count': int
        #   }
        self.estimators: Dict[str, Dict[str, Any]] = {}
    
    def update(self, x: torch.Tensor, key: str) -> None:
        """
        Update the Welford statistics for the given key, using data x.

        Args:
            x  : A scalar or tensor of any shape.
            key: A string identifying which quantity/field to update.
        """
        # If we've never seen this key, create its entry
        if key not in self.estimators:
            self.estimators[key] = {
                'n': 0,
                'mean': None,
                'M2': None,
                'total_count': 0
            }
        
        est = self.estimators[key]
        est['total_count'] += 1
        
        # Check burn-in
        if est['total_count'] <= self.burnin:
            return
        
        # Initialize mean, M2 with zeros of the same shape as x
        if est['mean'] is None:
            est['mean'] = torch.zeros_like(x)
            est['M2']   = torch.zeros_like(x)
        
        with torch.no_grad():
            est['n'] += 1
            delta = x - est['mean']
            est['mean'] += delta / est['n']
            delta2 = x - est['mean']
            est['M2'] += delta * delta2
    
    def get_mean(self, key: str) -> torch.Tensor:
        """
        Return the current mean of the field `key`.
        """
        est = self.estimators.get(key)
        if est is None or est['n'] == 0:
            return torch.tensor(float('nan'))
        return est['mean']

    def get_variance(self, key: str) -> torch.Tensor:
        """
        Return the sample variance (unbiased) of the field `key`.
        """
        est = self.estimators.get(key)
        if est is None or est['n'] < 2:
            return torch.tensor(float('nan'))
        # Sample variance uses (n - 1)
        return est['M2'] / (est['n'] - 1)

    def get_population_variance(self, key: str) -> torch.Tensor:
        """
        Return the population variance of the field `key`.
        """
        est = self.estimators.get(key)
        if est is None or est['n'] < 1:
            return torch.tensor(float('nan'))
        # Population variance uses n
        return est['M2'] / est['n']


class BayesianProblem(BaseProblem):
    # Poisson equation with Bayesian inference
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hist = {}
        # sample at regular interval
        self.u_samples = {}
        # MAP estimate
        self.MAP_sample = {}

        self.default_param = {}
        self.gt_param = {}
        self.init_param = {}
        self.estimator = WelfordEstimator()
        self.use_exact_sol = kwargs.get('use_exact_sol', False)
        
        self.loss_dict['post_data'] = self.nll_data
        self.loss_dict['post_res'] = self.nll_res
        self.loss_dict['post_res_nz'] = self.nll_res_nz
        
        self.loss_dict['prior_weight'] = self.nll_prior_nn
        self.loss_dict['prior_fun'] = self.nll_prior_fun
        self.loss_dict['prior_param'] = self.nll_prior_pde

        # store history of parameters
        self.hist = {k: [] for k in kwargs['trainable_param']}
    
    def config_traintype(self, traintype):
        if 'hmc' in traintype:
            self.is_sampling = True
        else:
            self.is_sampling = False
    
    def setup_parameters(self, **kwargs):
        # setup parameters
        for k in self.default_param:
            self.gt_param[k] =   torch.tensor(kwargs['gt_param'].get(k, self.default_param[k]), dtype=torch.float32)
            self.init_param[k] = torch.tensor(kwargs['init_param'].get(k, self.default_param[k]), dtype=torch.float32)
        
        # These are for neural network
        # initial guess
        self.all_params_dict = {k: v for k, v in self.init_param.items()}
        

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

    def nll_res(self, net):
        '''compute negative log likelihood of residual in BPINN
        As nll_data, sigma is not included here
        '''
        self.res, self.upred_res = self.get_res_pred(net)
        nll_res = 0.5 * torch.sum(self.res**2)
        return nll_res
    
    def nll_res_nz(self, net):
        ''' for bayesian PINN
        '''
        self.res, self.upred_res = self.get_res_pred(net)
        res_nz = self.dataset['res_nz']
        nll_res = 0.5 * torch.sum((self.res - res_nz)**2)
        return nll_res
    
    def nll_prior_nn(self, nn):
        '''P(W), prior of weights
        sigma not included here.
        '''
        # use normal distribution
        all_weight = parameters_to_vector(nn.param_net)
        nll = torch.sum(0.5 * (all_weight**2))
        return nll
    
    def nll_prior_fun(self, nn):
        '''P(W), prior of weights
        sigma not included here.
        '''
        # use normal distribution
        all_weight = parameters_to_vector(nn.param_pde)
        nll = torch.sum(0.5 * (all_weight**2))
        return nll

    def nll_prior_pde(self, nn):
        '''P(\Theta), prior of PDE parameter'''
        pass

    @torch.no_grad()
    def update_estimator(self, nn):
        # for computing mean and variance
        # collect scalar unknown or function unknown
        for pname in nn.trainable_param:
            # if scalar parameter
            if isinstance(nn.all_params_dict[pname], torch.nn.Parameter):
                self.hist[pname].append(nn.all_params_dict[pname].item())
                self.estimator.update(nn.all_params_dict[pname], pname)
            # if function parameter
            elif isinstance(nn.all_params_dict[pname], torch.nn.Module):
                fpred = nn.all_params_dict[pname](self.dataset['X_dat_test'])
                self.estimator.update(fpred, f'{pname}_pred')
            else:
                raise ValueError(f'unknown type of parameter {pname}')
        
        # collect solution
        if 'X_dat_test' in self.dataset:
            u_pred = nn(self.dataset['X_dat_test'], nn.pde_params_dict)
            self.estimator.update(u_pred, 'u_pred')
        elif 'x_dat' in self.dataset:
            u_pred = nn(self.dataset['x_dat'], nn.pde_params_dict)
            self.estimator.update(u_pred, 'u_pred')
        else:
            pass
    
    @torch.no_grad()
    def collect_solution(self, net):
        # collect pair of solution and parameters
        if 'u' not in self.u_samples:
            self.u_samples['u'] = []
            for pname in net.trainable_param:
                self.u_samples[pname] = []

        if 'X_dat_test' in self.dataset:
            upred = net(self.dataset['X_dat_test'], net.pde_params_dict)
            upred = upred.squeeze().cpu().numpy()
            self.u_samples['u'].append(upred)

        for pname in net.trainable_param:
            if isinstance(net.pde_params_dict[pname], torch.nn.Parameter):
                # for scalar parameter
                self.u_samples[pname].append(net.pde_params_dict[pname].item())
            elif isinstance(net.pde_params_dict[pname], torch.nn.Module):
                # for function
                fpred = net.pde_params_dict[pname](self.dataset['X_dat_test']).squeeze().cpu().numpy()
                self.u_samples[pname].append(fpred)
            else:
                raise ValueError(f'unknown type of parameter {pname}')
    
    @torch.no_grad()
    def collect_solution_MAP(self, net):
        
        if 'X_dat_test' in self.dataset:
            upred = net(self.dataset['X_dat_test'], net.pde_params_dict)
            upred = upred.squeeze().cpu().numpy()
            self.MAP_sample['u'] = upred

        for pname in net.trainable_param:
            if isinstance(net.all_params_dict[pname], torch.nn.Parameter):
                # for scalar parameter
                self.MAP_sample[pname] = (net.all_params_dict[pname].item())
            elif isinstance(net.all_params_dict[pname], torch.nn.Module):
                # for function
                fpred = net.all_params_dict[pname](self.dataset['X_dat_test']).squeeze().cpu().numpy()
                self.MAP_sample[pname] = (fpred)
            else:
                raise ValueError(f'unknown type of parameter {pname}')

    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        if self.hist:
            # save history of parameters
            # if self.hist is emtpy, that means optimization/pre-training, no sample collected
            for pname in net.trainable_param:
                self.dataset[f'sample_{pname}'] = self.hist[pname]
        
        # get mean and variance of function
        self.dataset['u_pred_mean'] = self.estimator.get_mean('u_pred')
        self.dataset['u_pred_var'] = self.estimator.get_population_variance('u_pred')

        for pname in net.trainable_param:
            if isinstance(net.all_params_dict[pname], torch.nn.Module):
                self.dataset[f'{pname}_pred_mean'] = self.estimator.get_mean(f'{pname}_pred')
                self.dataset[f'{pname}_pred_var'] = self.estimator.get_population_variance(f'{pname}_pred')
                self.dataset[f'{pname}pred_dat_test'] = net.pde_params_dict[pname](self.dataset['X_dat_test'])

        # save char_param
        if hasattr(self, 'char_param'):    
            for pname in self.char_param:
                self.dataset[f'char_{pname}'] = self.char_param[pname]

        if 'X_res_test' in self.dataset:
            self.dataset['upred_res_test'] = net(self.dataset['X_res_test'], net.pde_params_dict)
            self.dataset['upred_dat_test'] = net(self.dataset['X_dat_test'], net.pde_params_dict)
        

        if hasattr(self, 'u_exact'):
            self.dataset['uinf_dat_test'] = self.u_exact(self.dataset['X_dat_test'], net.pde_params_dict)
        
        self.save_collected_solution(net)
        self.prediction_variation(net)
    
    def save_collected_solution(self, net):
        # save solution example
        if 'u' in self.u_samples:
            self.dataset['eg_u'] = self.u_samples['u']
            for pname in net.trainable_param:
                self.dataset[f'eg_{pname}'] = self.u_samples[pname]
        
        # save MAP solution
        if 'u' in self.MAP_sample:
            self.dataset['MAP_u'] = self.MAP_sample['u']
            for pname in net.trainable_param:
                self.dataset[f'MAP_{pname}'] = self.MAP_sample[pname]

               
    @error_logging_decorator
    def visualize_distribution(self, savedir=None):

        for k, v in self.hist.items():
            mean = np.mean(v)
            std = np.std(v)
            vess = ess(np.array(v))

            fig, ax = plt.subplots()
            ax.hist(v, bins=20, density=True, alpha=0.6, color='g')
            ax.set_xlabel(k)
            ax.set_ylabel(f'density of {k}')
            # add title
            ax.set_title(f'{k}: mean={mean:.2f}, std={std:.2f}, ess={vess:.2f}')
            
            if savedir is not None:
                fpath = os.path.join(savedir, f'fig_hist_{k}.png')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')
    
    @error_logging_decorator
    def plot_mean_std(self, savedir=None, vname = 'u'):
        # plot mean and variance of u_pred

        x = self.dataset['X_dat_test']
        u = self.dataset['u_dat_test']
        
        if f'{vname}_pred_mean' not in self.dataset:
            print(f'{vname}_pred_mean not found in dataset')
            return
            
        u_pred_mean = self.dataset[f'{vname}_pred_mean']
        u_pred_var = self.dataset[f'{vname}_pred_var']
        u_pred_std = np.sqrt(u_pred_var)

        fig, ax = plt.subplots()
        
        # plot mean and variance
        ax.plot(x, u_pred_mean, label=f'{vname}_mean')
        ax.fill_between(x.squeeze(), u_pred_mean.squeeze()-2*u_pred_std.squeeze(), u_pred_mean.squeeze()+2*u_pred_std.squeeze(), alpha=0.3)

        # plot ground truth
        ax.plot(self.dataset['X_dat_test'], self.dataset[f'{vname}_dat_test'], label='GT', linestyle='--')

        train_name = f'{vname}_dat_train'
        if train_name in self.dataset:
            ax.scatter(self.dataset['X_dat_train'], self.dataset[train_name], label='data')

        ax.legend(loc='best')

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_{vname}_uq.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')
    
    @error_logging_decorator
    def plot_examples(self, savedir=None):
        # plot mean and variance of u_pred

        x = self.dataset['X_dat_test']
        if 'eg_u' not in self.dataset:
            print('eg_u not found in dataset, no sample collected')
            return
        num_example = len(self.dataset['eg_u'])

        # get all pde parameters
        var_names = self.dataset.filter('eg_')
        param_names = list(set([v.split('_')[1] for v in var_names]))
        # remove u
        param_names.remove('u')

        for i in range(num_example):

            # plo example solution
            fig, ax = plt.subplots()
            ax.scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], label='data')
            ax.plot(x, self.dataset['eg_u'][i], label='u_pred')
            param_dict = {k: self.dataset[f'eg_{k}'][i] for k in param_names}
            
            title = ', '.join([f'{k}={v:.4f}' for k, v in param_dict.items()])
            
            ax.set_title(title)
            ax.legend(loc='best')

            if savedir is not None:
                fpath = os.path.join(savedir, f'fig_example_u{i}.png')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')
                # close the figure
            plt.close(fig)
            

            # plot example function (PDE unknown)
            for k in param_names:
                input = self.dataset[f'eg_{k}'][i]
                # skip for scalar valued problem
                if isinstance(input, float):
                    continue
                fig, ax = plt.subplots()
                
                ax.plot(x, input, label=f'{k}')
                
                ax.legend(loc='best')
                if savedir is not None:
                    fpath = os.path.join(savedir, f'fig_example_{k}{i}.png')
                    fig.savefig(fpath, dpi=300, bbox_inches='tight')
                    print(f'fig saved to {fpath}')
                plt.close(fig)