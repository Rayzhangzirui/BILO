#!/usr/bin/env python
# define problems for PDE
import os
import torch

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import griddata

from BayesianProblem import BayesianProblem
from Options import *
from util import *
from DenseNet import DenseNet
from MatDataset import MatDataset
from GBMproblem import *



def double_logistic_sigmoid(u, uc, sigma_a, ksigmoid):
    # uc is the cutoff
    # u is predicted u
    smooth_sign = torch.tanh(ksigmoid * (u - uc))
    a = 0.5 + 0.5* smooth_sign * (1.0 - torch.exp(- (u - uc)**2 / (sigma_a**2)))
    return a
    

class GBMBayesProblem(GBMproblem, BayesianProblem):
    def __init__(self, **kwargs):
        # Initialize both parent classes properly
        # GBMproblem.__init__(self, **kwargs)
        # BayesianProblem.__init__(self, **kwargs)
        super().__init__(**kwargs)  # This starts the MRO chain

        self.setup_parameters(**kwargs)

        
        self.loss_dict['post_uchar_dat'] = self.nll_uchar_dat
        self.loss_dict['post_uchar_res'] = self.nll_uchar_res
        self.loss_dict['post_ugt_dat'] = self.nll_ugt_dat
        self.loss_dict['post_ugt_res'] = self.nll_ugt_res
        self.loss_dict['post_th1'] = self.nll_th1
        self.loss_dict['post_th2'] = self.nll_th2
        self.loss_dict['post_th1_mse'] = self.nll_th1_mse
        self.loss_dict['post_th2_mse'] = self.nll_th2_mse
        self.loss_dict['prior_param'] = self.nll_prior_pde
        self.loss_dict['prior_th'] = self.nll_th_prior

        # convert to tensor
        self.th1_normal_mu = torch.tensor(self.opts['th1_normal_prior'][0])
        self.th1_normal_sigma = torch.tensor(self.opts['th1_normal_prior'][1])
        self.th2_normal_mu = torch.tensor(self.opts['th2_normal_prior'][0])
        self.th2_normal_sigma = torch.tensor(self.opts['th2_normal_prior'][1])


        self.lower_bound = {'rD': self.rD_range[0], 'rRHO': self.rRHO_range[0], 'th1': self.th1_range[0], 'th2': self.th2_range[0]}
        self.upper_bound = {'rD': self.rD_range[1], 'rRHO': self.rRHO_range[1], 'th1': self.th1_range[1], 'th2': self.th2_range[1]}
    
        
    def nll_prior_pde(self, nn):
        '''P(Theta), prior of PDE parameters using range penalty'''
        nll = 0.0

        def nll_log_normal_penalty(x, mu, sigma):
            return torch.log(x * sigma * torch.sqrt(torch.tensor(2) * torch.pi)) + (torch.log(x) - mu)**2 / (2 * sigma**2)

        
        mu = self.opts['log_normal_mu']
        sigma = self.opts['log_normal_sigma']
        # rD parameter
        if 'rD' in nn.trainable_param:
            # nll += penalty(nn.all_params_dict['rD'].squeeze(), self.rD_range[0], self.rD_range[1])
            nll += nll_log_normal_penalty(nn.all_params_dict['rD'].squeeze(), mu, sigma)
        
        # rRHO parameter  
        if 'rRHO' in nn.trainable_param:
            nll += nll_log_normal_penalty(nn.all_params_dict['rRHO'].squeeze(), mu, sigma)
        
        return nll

    
    def nll_th_prior(self, nn):
        '''P(Theta), prior of threshold parameters using normal prior'''
        nll = 0.0

        def nll_normal_penalty(x, mu, sigma):
            return 0.5 * torch.log(2 * torch.pi * sigma**2) + (x - mu)**2 / (2 * sigma**2)

        # th1 parameter
        if 'th1' in nn.trainable_param:
            nll += nll_normal_penalty(nn.all_params_dict['th1'].squeeze(), self.th1_normal_mu, self.th1_normal_sigma)
        
        # th2 parameter  
        if 'th2' in nn.trainable_param:
            nll += nll_normal_penalty(nn.all_params_dict['th2'].squeeze(), self.th2_normal_mu, self.th2_normal_sigma)
        
        return nll

    def nll_uchar_res(self, net):
        # mse of uchar_res
        data = self.dataset.batch['res']
        X = data['X_res_train']
        u = data['uchar_res_train']
        phi = data['phi_res_train']
        upred = net(X, net.pde_params_dict)
        err = (upred - u) * phi
        return 0.5*torch.sum(err**2)
    
    def nll_uchar_dat(self, net):
        # mse of uchar_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['uchar_dat_train']
        phi = data['phi_dat_train']
        upred = net(X, net.pde_params_dict)
        err = (upred - u) * phi
        return 0.5*torch.sum(err**2)
    
    def nll_ugt_res(self, net):
        # mse of ugt_res
        data = self.dataset.batch['res']
        X = data['X_res_train']
        u = data['ugt_res_train']
        phi = data['phi_res_train']
        upred = net(X, net.pde_params_dict)
        err = (upred - u) * phi
        return 0.5*torch.sum(err**2)
    
    def nll_ugt_dat(self, net):
        # mse of ugt_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['ugt_dat_train']
        phi = data['phi_dat_train']
        upred = net(X, net.pde_params_dict)
        err = (upred - u) * phi
        return 0.5*torch.sum(err**2)

    def nll_th1(self, net):
        # negative log likelihood of th1
        # P(u1|u, th1) = product a^u1 * (1-a)^(1-u1)
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        useg = data['u1_dat_train']
        self.upred_dat = net(X, net.pde_params_dict)
        a = double_logistic_sigmoid(self.upred_dat*phi, net.all_params_dict['th1'], self.opts['sigma_a'], self.opts['ksigmoid'])
        p = useg * torch.log(a + 1e-6) + (1 - useg) * torch.log(1 - a + 1e-6)
        nll = - torch.sum(p)
        return nll

    def nll_th2(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        useg = data['u2_dat_train']
        # save some memory, do not recomputed upred
        a = double_logistic_sigmoid(self.upred_dat*phi, net.all_params_dict['th2'], self.opts['sigma_a'], self.opts['ksigmoid'])
        p = useg * torch.log(a + 1e-6) + (1 - useg) * torch.log(1 - a + 1e-6)
        nll = - torch.sum(p)
        return nll

    def nll_th1_mse(self, net):
        # use mse approximation
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        useg = data['u1_dat_train']
        self.upred_dat = net(X, net.pde_params_dict)
        
        uth = torch.nn.functional.sigmoid( self.opts['ksigmoid'] * (self.upred_dat - net.all_params_dict['th1']))
        err = (uth - useg) * phi
        nll = 0.5*torch.sum(err**2)
        
        return nll

    def nll_th2_mse(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        useg = data['u2_dat_train']
        # do not recomputed upred
        uth = torch.nn.functional.sigmoid( self.opts['ksigmoid'] * (self.upred_dat - net.all_params_dict['th2']))
        err = (uth - useg) * phi
        nll = 0.5*torch.sum(err**2)
        return nll


    def visualize(self, savedir=None):
        super().visualize(savedir)
        self.visualize_distribution(savedir)

        # additional plots for a1, a2
        X = self.dataset['X_dat_train']
        a1 = self.dataset['a1_dat_train']
        a2 = self.dataset['a2_dat_train']
        u1 = self.dataset['u1_dat_train']
        u2 = self.dataset['u2_dat_train']
        plot_scatter(X, a1, ref = u1,fname='a1_dat_train.png', savedir=savedir)
        plot_scatter(X, a2, ref = u2,fname='a2_dat_train.png', savedir=savedir)



    @torch.no_grad()
    def validate(self, nn):
        d = {}
        for pname in nn.trainable_param:
            d[pname] = nn.all_params_dict[pname].item()
            mean = self.estimator.get_mean(pname)
            std = torch.sqrt(self.estimator.get_population_variance(pname))
            d[pname + '_mean'] = mean.item()
            d[pname + '_std'] = std.item()
        return d
        
    @torch.no_grad()
    def collect_solution(self, net):
        # override parent method
        # collect u_grid
        if 'u_dat' not in self.u_samples:
            self.u_samples['u_dat'] = []
            self.u_samples['u_grid'] = []
            for pname in net.trainable_param:
                self.u_samples[pname] = []

        if 'X_dat' in self.dataset:
            upred = net(self.dataset['X_dat'], net.pde_params_dict)
            upred = upred.squeeze().cpu().numpy()
            self.u_samples['u_dat'].append(upred)
        
        if hasattr(self, 'u_grid') and self.u_grid is not None:
            # Ensure u_grid is on CPU and convert to numpy
            if torch.is_tensor(self.u_grid):
                ugrid = self.u_grid.detach().cpu().numpy()
            else:
                ugrid = self.u_grid
            self.u_samples['u_grid'].append(ugrid)

        for pname, v in net.pde_params_dict.items():
            if isinstance(v, torch.nn.Parameter):
                # for scalar parameter
                self.u_samples[pname].append(v.item())
            else:
                raise ValueError(f'unknown type of parameter {pname}')
    
    @torch.no_grad()
    def collect_solution_MAP(self, net):
        for pname in net.trainable_param:
            self.MAP_sample[pname] = (net.all_params_dict[pname].item())
        
        if hasattr(self, 'u_grid') and self.u_grid is not None:
            # Ensure u_grid is on CPU and convert to numpy
            if torch.is_tensor(self.u_grid):
                ugrid = self.u_grid.detach().cpu().numpy()
            else:
                ugrid = self.u_grid
            self.MAP_sample['u_grid'] = ugrid

        

            
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset.to_device(self.dataset.device)
        net.to(self.dataset.device)

        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']

        # write prediction by batch
        self.dataset['upred_dat'] = net(x_dat, net.pde_params_dict)
        self.dataset['upred_dat_train'] = net(x_dat_train, net.pde_params_dict)

        # Prediction by batching for large tensors x_res and x_res_train
        def batch_predict(x, batch_size):
            predictions = []
            for i in range(0, x.shape[0], batch_size):
                batch = x[i:i + batch_size]
                pred = net(batch, net.pde_params_dict)
                predictions.append(pred)
            return torch.cat(predictions, dim=0)

        # Perform batched predictions to avoid OOM errors
        self.dataset['upred_res'] = batch_predict(x_res, 20000)
        self.dataset['upred_res_train'] = batch_predict(x_res_train, 20000)

        # predict the thresholded a1, a2
        X = self.dataset['X_dat_train']
        phi = self.dataset['phi_dat_train']
        upred = net(X, net.pde_params_dict)
        a1 = double_logistic_sigmoid(upred*phi, net.all_params_dict['th1'], self.opts['sigma_a'], self.opts['ksigmoid']) 
        a2 = double_logistic_sigmoid(upred*phi, net.all_params_dict['th2'], self.opts['sigma_a'], self.opts['ksigmoid'])
        self.dataset['upredphi'] = upred * phi
        self.dataset['a1_dat_train'] = a1
        self.dataset['a2_dat_train'] = a2


        self.prediction_variation(net, list_params=['rD', 'rRHO'])
        self.save_collected_solution(net)

    def save_collected_solution(self, net):
        # override parent method

        if 'u_dat' in self.u_samples and len(self.u_samples['u_dat']) > 0:
            self.dataset['eg_u_dat'] = self.u_samples['u_dat']
        if 'u_grid' in self.u_samples and len(self.u_samples['u_grid']) > 0:
            self.dataset['eg_u_grid'] = self.u_samples['u_grid']
            for pname in net.trainable_param:
                self.dataset['eg_' + pname] = torch.tensor(self.u_samples[pname])
        # save MAP sample
        if 'u_grid' in self.MAP_sample:
            self.dataset['MAP_u_grid'] = self.MAP_sample['u_grid']
            for pname in net.trainable_param:
                self.dataset['MAP_' + pname] = self.MAP_sample[pname]
        
