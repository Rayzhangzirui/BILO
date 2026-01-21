#!/usr/bin/env python
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from MatDataset import MatDataset 
from DeepONet import DeepONet, OpData, scalarFNO
# from BaseOperator import BaseOperator
from BayesianProblem import BayesianProblem
from PointProcess import PointProcess, PDE
from util import griddata_subsample, generate_grf, error_logging_decorator
from itertools import cycle
from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO

def linear_interp(xi, yi, xq):
    """
    Performs differentiable linear interpolation.
    
    xi: 1D tensor of sorted grid points (strictly increasing), shape (N,)
    yi: 1D tensor of function values at xi, shape (N,)
    xq: 1D tensor of query points (all within [xi[0], xi[-1]]), shape (M,)
    
    Returns:
    yq: 1D tensor of interpolated values at xq, shape (M,)
    """
    # Locate indices such that xi[idx - 1] <= xq < xi[idx]
    # Since xi is strictly increasing and xq is in the range, this works well.
    indices = torch.searchsorted(xi, xq).clamp(1, xi.numel() - 1)
    
    # Get left and right neighbors for each xq.
    x0 = xi[indices - 1]
    x1 = xi[indices]
    y0 = yi[indices - 1]
    y1 = yi[indices]
    
    # Compute the interpolation weight (avoiding division by zero).
    epsilon = 1e-6
    w = (xq - x0) / (x1 - x0 + epsilon)
    
    # Compute the linear interpolation.
    yq = y0 * (1 - w) + y1 * w
    return yq

class PointProcessOperatorLearning(PointProcess):
    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        BayesianProblem.__init__(self, **kwargs)
        self.input_dim = 1
        self.output_dim = 1
        self.param_dim = 2
        self.opts = kwargs

        # charateristic parameters
        self.char_param = {'lmbd': 100.0, 'mu': 1.0, 'z': 1.0, 'L': 1.0}
        self.default_param = {'lmbd': 5.0, 'mu': 10.0, 'z': 0.5, 'L': 1.0}
    
        self.loss_dict = {'data': self.get_data_loss, 'tdata': self.get_test_loss, 'particle': self.nll_particle}

        self.arch = None
        self.method = None

    def config_traintype(self, traintype):
        arch, method = traintype.split('-')
        self.arch = arch
        self.method = method
        assert arch in {'fno', 'deeponet'}
        assert method in {'inv', 'init'}

        if method == 'inv':
            self.setup_parameters(**self.opts)
            

    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        # if X P U are in the data set, this is pretraining dataset
        # otherwise this is inverse dataset
        if self.method == 'init':
            self.dataset = MatDataset(self.opts['datafile'])
            self.setup_pretrain_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_pretrain
        else:
            # call super class method
            self.setup_inverse_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_inverse
    
    def setup_network(self, **kwargs):
        # called after config_traintype and setup_dataset
        if self.arch == 'deeponet':
            raise ValueError('DeepONet not supported for PointProcessOperatorLearning')

        elif self.arch == 'fno':
            self.lambda_transform = lambda X, u:  u * X * (1 - X)

            opnet = scalarFNO(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim,
            n_modes=kwargs['n_modes'], n_layers=kwargs['n_layers'], hidden_channels=kwargs['hidden_channels'],
            lambda_transform=self.lambda_transform)

        else:
            raise ValueError('Invalid network architecture')

        opnet.pde_params_list = ['lmbd', 'mu']
        if self.method == 'inv':
            # initialize pde_param in FNO
            lmbd = self.init_param['lmbd']
            mu = self.init_param['mu']
            pde_params_dict = {'lmbd': torch.nn.Parameter(torch.tensor([lmbd], dtype=torch.float32)),
            'mu': torch.nn.Parameter(torch.tensor([mu], dtype=torch.float32))}
            opnet.pde_params_dict = torch.nn.ParameterDict(pde_params_dict)
                
        return opnet
    
        

    def get_inverse_data(self):
        '''
        Return data for training inverse problem
        These are from BILO/PINN dataset. Need reshape
        '''
        U = self.dataset['u_dat_train']
        X = self.dataset['X_dat_train']
        return X, U
    
    

    def setup_pretrain_dataset(self, ds_opts, noise_opts=None, device='cuda'):

        split = ds_opts['split']
        self.dataset.to_device(device)

        X = self.dataset['X']
        # reshape as (Nx, 1)
        X = X.reshape(-1, 1)

        lmbd = self.dataset['lambda'].reshape(-1, 1)/self.char_param['lmbd']
        mu = self.dataset['mu'].reshape(-1, 1)/self.char_param['mu']

        # reshape and concatenate
        P = torch.cat([lmbd, mu], dim=1)

        # U is (Np, Nx)
        U = self.dataset['U']

        self.OpData = OpData(X, P, U)

        n_total = len(self.OpData)
        train_size = int(split * n_total)
        test_size = n_total - train_size

        print(f"Training size: {train_size}, Testing size: {test_size}")
        # Randomly shuffle the indices
        total_indices = torch.randperm(len(self.OpData))

        # Split indices into training and testing
        train_indices = total_indices[:train_size]
        test_indices = total_indices[train_size:]

        self.train_dataset = OpData(self.OpData.X, self.OpData.P[train_indices], self.OpData.U[train_indices])
        self.test_dataset = OpData(self.OpData.X, self.OpData.P[test_indices], self.OpData.U[test_indices])

        batch_size = ds_opts['batch_size']

        self.train_loader = cycle(DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True))
    
    def setup_inverse_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        # call super class method
        super().setup_dataset(ds_opts, noise_opts=noise_opts, device=device)
        self.X = self.dataset['X_dat_train']
        # change loss function
        self.loss_dict['data'] = self.get_data_loss_inverse
        self.loss_dict['particle'] = self.nll_particle
    
    def get_data_loss_inverse(self, net):
        # for FNO

        xi = self.X # (N, 1)
        upred_dat_train = net(net.pde_params_dict, xi) #(1,N)

        loss = torch.nn.functional.mse_loss(upred_dat_train.flatten(), self.dataset['u_dat_train'].flatten())
        
        return loss

    def get_data_loss(self, net):
        # each time evaluated on different batch
        P, U = next(self.train_loader)
        U_pred = net(P, self.train_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred.flatten(), U.flatten())
        return loss
    
    def nll_particle(self, net):
        '''compute likelihood of trajectory data'''
        xs = self.dataset['samples'].squeeze()
        M = self.dataset['n_snapshot']

        xi = self.X # (N, 1)
        u_xi = net(net.pde_params_dict, xi).squeeze() #(N,)

        # interpolate at 
        u = linear_interp(xi.squeeze(), u_xi, xs)
        
        # trapezoidal rule for integral using X and u_xi
        # assuming uniform grid and unit interval
        integral = torch.trapz(u_xi, xi.squeeze())
        
        ll = - M * integral + torch.sum(torch.log(u))
        nll = -ll
        return nll
    

    def get_test_loss(self, net):
        U_pred = net(self.test_dataset.P, self.test_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred, self.test_dataset.U)
        return loss

    @torch.no_grad()
    def validate_pretrain(self, net):
        # validate the network
        net.eval()
        loss = self.get_test_loss(net)
        return {'tdata': loss.item()}
    
    @torch.no_grad()
    def validate_inverse(self, net):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        return {'lmbd': net.pde_params_dict['lmbd'].item(), 'mu': net.pde_params_dict['mu'].item()}

    
    def visualize(self, savedir=None):
        # visualize the results
        if self.method == 'inv':
            self.visualize_inverse(savedir=savedir)
        else:
            self.visualize_pretrain(savedir=savedir)
    
    def make_prediction(self, net):
        if self.method == 'inv':
            self.make_prediction_inverse(net)
        else:
            self.make_prediction_pretrain(net)


    @torch.no_grad()
    def make_prediction_pretrain(self, net):
        # iterate over OP dataset and make prediction
        # for each P, U in the dataset
        X = self.OpData.X
        P = self.OpData.P

        U_pred = torch.zeros_like(self.OpData.U)
        mse = torch.zeros(P.shape[0])
        max_err = torch.zeros(P.shape[0])
        data_loss = torch.zeros(P.shape[0])

        # make predictioin at lambda = 500, mu = 10
        lmbd = torch.tensor([500/self.char_param['lmbd']], dtype=torch.float32).to(X.device)
        mu = torch.tensor([10/self.char_param['mu']], dtype=torch.float32).to(X.device)
        p = {'lmbd': lmbd, 'mu': mu}
        upred = net(p, X)
        self.dataset['upred_gt'] = upred.squeeze()

        z = torch.tensor([0.5], dtype=torch.float32).to(X.device)
        L = torch.tensor([1.0], dtype=torch.float32).to(X.device)
        self.dataset['u_gt'] = PDE.u_exact(X, lmbd=lmbd*self.char_param['lmbd'],
        mu = mu*self.char_param['mu'], z=z, L=L).squeeze()

        
        for i in range(len(P)):
            U_pred[i] = net(P[i:i+1], X)
            exact_U = self.OpData.U[i:i+1].squeeze()
            mse[i] = torch.nn.functional.mse_loss(U_pred[i].squeeze(), exact_U)
            max_err[i] = torch.max(torch.abs(U_pred[i].squeeze() - exact_U))
            # data loss w.r.t to gt
            data_loss[i] = torch.nn.functional.mse_loss(U_pred[i].flatten(), self.dataset['u_gt'].flatten())
        
        self.dataset['U_pred'] = U_pred
        self.dataset['mse'] = mse
        self.dataset['max_err'] = max_err
        self.dataset['data_loss'] = data_loss


        
    
    def plot_grid(self, field, savedir=None):
        # reshape error and plot
        n_lmbd = self.dataset['lambda_arr'].size
        n_mu = self.dataset['mu_arr'].size
        error = self.dataset[field].reshape(n_mu, n_lmbd,  order='F')
        fig, ax = plt.subplots()

        min_lmbd = self.dataset['lambda_arr'].flat[0]
        max_lmbd = self.dataset['lambda_arr'].flat[-1]
        min_mu = self.dataset['mu_arr'].flat[0]
        max_mu = self.dataset['mu_arr'].flat[-1]

        pos = ax.imshow(error, cmap='viridis', extent=[min_lmbd, max_lmbd, min_mu, max_mu], aspect='auto')
       
        # add colorbar
        fig.colorbar(pos, ax=ax)
        
        ax.set_xlabel('lambda')
        ax.set_ylabel('mu')

        if savedir is not None:
            path = os.path.join(savedir, f'fig_grid_{field}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
        plt.close(fig)
        
    @error_logging_decorator
    def plot_grid_error(self,  savedir=None):
        # reshape error and plot
        self.plot_grid('mse', savedir=savedir)
        self.plot_grid('max_err', savedir=savedir)
        self.plot_grid('data_loss', savedir=savedir)

    def plot_gt_pred_pretrain(self, savedir=None):
        # plot ground truth and prediction
        fig, ax = plt.subplots()
        ax.plot(self.dataset['X'].squeeze(), self.dataset['u_gt'], label='exact')
        ax.plot(self.dataset['X'].squeeze(), self.dataset['upred_gt'].flatten(), label='pred')
        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_gt_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
        plt.close(fig)

    def visualize_pretrain(self, savedir=None):
        # reshape error and plot
        self.plot_grid_error(savedir)
        self.plot_gt_pred_pretrain(savedir)

    @torch.no_grad()
    def make_prediction_inverse(self, net):
        # make prediction at original X_dat and X_res

        upred_dat_test = net(net.pde_params_dict, self.dataset['X_dat_test']).view(-1,1)
        self.dataset['upred_dat_test'] = upred_dat_test #(N_dat_train,)

    def plot_prediction_inverse(self, savedir=None):
        # plot prediction and ground truth
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], label='exact')
        ax.plot(self.dataset['X_dat_test'], self.dataset['upred_dat_test'], label='pred')
        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
        plt.close(fig)

    def visualize_inverse(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_prediction_inverse(savedir)