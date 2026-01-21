#!/usr/bin/env python
# # define problems for PDE
import os
import math
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from util import generate_grf, griddata_subsample, error_logging_decorator

from BaseProblem import BaseProblem
from BayesianProblem import BayesianProblem
from MatDataset import MatDataset
from DenseNet import DenseNet, ParamFunction

class DarcyDenseNet(DenseNet):
    ''' override the embedding function of DenseNet
    - div(f grad u) = 1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def output_transform(self, X, u, param):
        ''' impose 0 boundary condition'''
        # u(x,y) =u_NN(x,t) * x * (1-x) * y * (1-y)
        return u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1] * (1 - X[:,0:1])

class GausianRandomField:
    # lmbd_ij =  ( (i^2+j^2) pi^2 + tau^2 )^(-alpha)
    # f = sum_{ij}^K a_ij  sqrt(lmbd_ij) 2 cos(i pi x) cos(j pi y)

    def __init__(self, K = 64, alpha = 1.0, tau=3.0):
        self.K = int(K)
        self.alpha = alpha
        self.tau = tau

        n = int(math.sqrt(K))
        # assert n**2 == K, 'K must be a square number'
        assert n**2 == K, f'K must be a square number, K={K}, n={n}'

        # self.basis
        i_vals = torch.arange(1, n + 1, dtype=torch.int32)
        j_vals = torch.arange(1, n + 1, dtype=torch.int32)

        I, J = torch.meshgrid(i_vals, j_vals, indexing='ij')
        # Flatten to obtain vectors of length K (order: (i-1)*n + j)
        self.I = I.reshape(-1)  # shape: (K,) 111222333
        self.J = J.reshape(-1)  # shape: (K,) 123123123
        
        # Compute eigenvalues λ_k = (π²*(i²+j²) + tau²)^(-alpha)
        lmbd = (torch.pi**2 * (I**2 + J**2) + tau**2) ** (-alpha)
        self.lmbd = lmbd.view(1, self.K)  # shape: (1, K)

    def to_device(self, device):
        self.I = self.I.to(device)
        self.J = self.J.to(device)
        self.lmbd = self.lmbd.to(device)

    def f(self, a, X):
        # x is (N,1)
        # a is (K,)
        x = X[:, 0:1]
        y = X[:, 1:2]

        # shape of X: (N, K)
        # order is cos(1x)cos(1y), cos(1x)cos(2y), cos(1x)cos(3y), cos(2x)cos(1y), cos(2x)cos(2y), ...
        # for consistency with matlab data
        basis = 2 * torch.cos(torch.pi  * x * self.I.view(1, self.K)) * \
               torch.cos(torch.pi  * y * self.J.view(1, self.K))
        
        # (N,1)
        f = torch.sum(a * torch.sqrt(self.lmbd) * basis, dim=1, keepdim=True)

        return f
        

class FunctionExpansion(torch.nn.Module):
    # Function expansion for f(x) = sum_{i=1}^K a_i lmbd_i (i pi)^2 sin(i pi x)
    # 1og10 D(x) = f(x)
    def __init__(self, a, grf, transgrf="exp"):

        super(FunctionExpansion, self).__init__()
        # Initialize a_i as nn.Parameter for gradient-based optimization
        self.a = torch.nn.Parameter(a)
        # self.register_parameter('a', param=torch.nn.Parameter(a))
        self.grf = grf
        self.transgrf = transgrf

    def forward(self, x):
        f = self.grf.f(self.a, x)
        if self.transgrf == "exp":
            D = torch.exp(f)
        elif self.transgrf == "sigmoid":
            # heaviside(sigmoid - 0.5) * 9 + 3
            D = torch.sigmoid(20*f) * 9 + 3
        else:
            # throw error
            raise ValueError(f"unknown transgrf {self.transgrf}")

        return D



class Darcy2dBayes(BayesianProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = 2 # x, y
        self.output_dim = 1
        self.opts=kwargs

        self.dataset = MatDataset(kwargs['datafile'])
        self.testcase = kwargs['testcase']
        self.usegrf = self.opts['usegrf']
        self.transgrf = self.opts['transgrf']
        self.f = lambda x,y: 10.0

        # no exact solution, D is transformed
        self.use_exact_sol = False

    def residual(self, nn, X):
        ''' - div(f grad u) = 1'''

        X.requires_grad_(True)

        x = X[:, 0:1]
        y = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((x,y), dim=1)

        # Forward pass through the network
        u_pred = nn(nn_input, nn.pde_params_dict)

        # Get the predicted f
        f = nn.params_expand['f']

        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u_pred)
        
        # Compute gradients with respect to the sliced tensors
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=v, create_graph=True,retain_graph=True)[0]
        u_y = torch.autograd.grad(u_pred, y, grad_outputs=v, create_graph=True,retain_graph=True)[0]

        # Compute the divergence of the predicted f
        f_x = torch.autograd.grad(u_x * f, x, grad_outputs=v, create_graph=True,retain_graph=True)[0]
        f_y = torch.autograd.grad(u_y * f, y, grad_outputs=v, create_graph=True,retain_graph=True)[0]

        # Compute the right-hand side of the PDE
        res = - (f_x + f_y) - self.f(x,y)
        
        
        return res, u_pred
    
    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['X_res_train'])
        return res, pred

    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'], net.pde_params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))
        return loss
    
    def nll_data(self, nn):
        u_pred = nn(self.dataset['X_dat_train'], nn.pde_params_dict)
        err = u_pred - self.dataset['u_dat_train']
        # negative log likelihood
        nll = 0.5 * torch.sum(err**2)
        return nll
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_dat_train']
        y = net.pde_params_dict['f'](x)
        return torch.mean(torch.square(y - self.dataset['f_dat_train']))
    
    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        # initial guess of a
        if self.usegrf:
            # Infer device from grf which has already been moved to the correct device in setup_dataset
            grf_device = self.grf.lmbd.device
            a = torch.tensor(self.dataset['a'][self.testcase-1, :], dtype=torch.float32, device=grf_device)
            self.param_fun = FunctionExpansion(a, self.grf, self.transgrf)
        else:
            self.param_fun = ParamFunction(input_dim=2, output_dim=1,
                fdepth=kwargs['fdepth'], fwidth=kwargs['fwidth'],fsiren=kwargs['fsiren'],
                                        activation=kwargs['activation'],
                                        output_transform=lambda x, u: torch.exp(u))
                    
        self.all_params_dict = {'f': self.param_fun}

        net = DarcyDenseNet(**kwargs,
                            all_params_dict= self.all_params_dict,
                            trainable_param = ['f'])
        
        return net
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_meshgrid('u_dat_test','upred_dat_test',savedir=savedir)
        self.plot_meshgrid('f_dat_test','fpred_dat_test',savedir=savedir)
        if self.is_sampling:
            self.plot_mean_std(savedir,'u')
            self.plot_mean_std(savedir,'f')
            # self.plot_examples(savedir)
    
    def create_dataset_from_file(self, dsopt):
        # use all data for training
        dataset = self.dataset
        dataset.to_np()
        self.grf = GausianRandomField(K=dataset['K'], alpha=dataset['alpha'], tau=dataset['tau'])
    
        u = dataset['u'][:,:,self.testcase-1].squeeze()
        f = dataset['A'][:,:,self.testcase-1].squeeze()
        
        dataset.pop(f'u',None)
        dataset.pop(f'A',None)
        
        dataset['u'] = u
        dataset['A'] = f

        gx = dataset['gx']
        gy = dataset['gy']

        dataset['X_dat_test'] = np.column_stack((gx.reshape(-1, 1,order='F'), gy.reshape(-1, 1,order='F')))
        dataset['u_dat_test'] = u.reshape(-1, 1,order='F')
        dataset['f_dat_test'] = f.reshape(-1, 1,order='F')
        
        dataset['X_res_test'] = np.column_stack((gx.reshape(-1, 1,order='F'), gy.reshape(-1, 1,order='F')))
        dataset['u_res_test'] = u.reshape(-1, 1,order='F')
        dataset['f_res_test'] = f.reshape(-1, 1,order='F')


        # Nx is number of grid point for residual loss
        Nx = dsopt['Nx']
        sgx, sgy, su = griddata_subsample(gx, gy, u, Nx, Nx)
        _, _, sf = griddata_subsample(gx, gy, f, Nx, Nx)

        dataset['X_res_train'] = np.column_stack((sgx.reshape(-1, 1,order='F'), sgy.reshape(-1, 1,order='F')))
        dataset['u_res_train'] = su.reshape(-1, 1,order='F')
        dataset['f_res_train'] = sf.reshape(-1, 1,order='F')

        # Nx_train is number of grid point for data loss
        Nx = dsopt['Nx_train']
        sgx, sgy, su = griddata_subsample(gx, gy, u, Nx, Nx)
        _, _, sf = griddata_subsample(gx, gy, f, Nx, Nx)

        dataset['X_dat_train'] = np.column_stack((sgx.reshape(-1, 1,order='F'), sgy.reshape(-1, 1,order='F')))
        dataset['u_dat_train'] = su.reshape(-1, 1,order='F')
        dataset['f_dat_train'] = sf.reshape(-1, 1,order='F')
        
        dataset.printsummary()
    
    @torch.no_grad()
    def validate(self, nn):
        '''compute l2 error and linf error of inferred f(t,x)'''
        
        x = self.dataset['X_res_test']
        f = self.dataset['f_res_test']
        
        f_pred = nn.pde_params_dict['f'](x)
        err = f - f_pred
        l2norm = torch.mean(torch.square(err))
        linfnorm = torch.max(torch.abs(err)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}
    
    @error_logging_decorator
    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        ''' downsample for training'''
        
        self.create_dataset_from_file(ds_opts)
        self.grf.to_device(device)
        self.dataset.to_torch()

        if noise_opts['use_noise']:
            print('add noise to training data')
            noise = noise_opts['std'] * torch.randn_like(self.dataset['u_dat_train'])
    
            self.dataset['noise'] = noise
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']
        
        self.dataset.to_device(device)
    
    def plot_meshgrid(self, name_true, name_pred, savedir=None):
        # plot u at X_res, 
        
        u = self.dataset[name_true]
        u_pred = self.dataset[name_pred]
        
        # reshape to 2D, in numpy
        Nx = int(self.dataset['gx'].shape[0])
        u = u.reshape(Nx, Nx)
        u_pred = u_pred.reshape(Nx, Nx)
        err = u - u_pred

        # get min max of u
        min_u = np.min(u)
        max_u = np.max(u)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # 2D plot
        cax = ax[0].imshow(u_pred , cmap='viridis', extent=[0, 1, 0, 1], origin='lower', vmin=min_u, vmax=max_u)
        ax[0].set_title('NN')
        fig.colorbar(cax, ax=ax[0])

        cax = ax[1].imshow(u , cmap='viridis', extent=[0, 1, 0, 1], origin='lower', vmin=min_u, vmax=max_u)
        ax[1].set_title('Exact')
        fig.colorbar(cax, ax=ax[1])

        cax = ax[2].imshow(err, cmap='plasma', extent=[0, 1, 0, 1], origin='lower')
        ax[2].set_title('Error')
        fig.colorbar(cax, ax=ax[2])

        fig.tight_layout()

        if savedir is not None:
            path = os.path.join(savedir, f'fig_grid_{name_pred}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    @error_logging_decorator
    def plot_mean_std(self, savedir=None, vname = 'u'):
        # plot mean and variance of u_pred

        if f'{vname}_pred_mean' not in self.dataset:
            print(f'{vname}_pred_mean not found in dataset')
            return
        
        Nx = int(self.dataset['gx'].shape[0])

        u_gt = self.dataset[f'{vname}_dat_test'].reshape(Nx, Nx)

        u_pred_mean = self.dataset[f'{vname}_pred_mean'].reshape(Nx, Nx)
        u_pred_var = self.dataset[f'{vname}_pred_var'].reshape(Nx, Nx)
        u_pred_std = np.sqrt(u_pred_var)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        cax = ax[0].imshow(u_gt, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
        ax[0].set_title(f'{vname} gt')
        fig.colorbar(cax, ax=ax[0])

        cax = ax[1].imshow(u_pred_mean, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
        ax[1].set_title(f'{vname} mean')
        fig.colorbar(cax, ax=ax[1])

        cax = ax[2].imshow(u_pred_std, cmap='plasma', extent=[0, 1, 0, 1], origin='lower')
        ax[2].set_title(f'{vname} std')
        fig.colorbar(cax, ax=ax[2])

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_{vname}_uq.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')