#!/usr/bin/env python
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from MatDataset import MatDataset 
from DeepONet import DeepONet, OpData, scalarFNO
from VarPoiProblem import VarPoiProblem
from util import griddata_subsample, generate_grf, add_noise, uniform_subsample_with_endpoint

from itertools import cycle
from torch.utils.data import Dataset, DataLoader


class VarPoiDeepONet(VarPoiProblem):
    def __init__(self, **kwargs):

        self.input_dim = 1
        self.output_dim = 1
        self.param_dim = kwargs['param_dim']
        
        self.lambda_transform = lambda x, u: u * x * (1.0 - x)
        self.testcase = kwargs.get('testcase', -1)
        self.dataset = MatDataset(kwargs['datafile'])
        self.loss_dict = {'data': self.get_data_loss, 'tdata': self.get_test_loss, 'l2grad': self.regularization_loss}

        self.is_pretraining = False
    
    def pad_pde_param(self, param):
        '''pad pde_param with 1s at the beginning and end'''
        assert param.shape[0] == 1

        # Create tensors for padding
        padding = torch.tensor([[1.0]], dtype=torch.float32, device=param.device, requires_grad=False)

        # Concatenate padding and param to keep it as a 1-by-(N+2) tensor
        full_d = torch.cat([
            padding,     # Padding at the beginning
            param,       # Original param
            padding      # Padding at the end
        ], dim=1)

        return full_d

    def regularization_loss(self, nn):
        '''l2 norm of gradient of pde_param'''
        # pad endpoint with 0
        full_d = self.pad_pde_param(nn.pde_params_dict['D'])
        n = full_d.shape[1]
        h = 1.0 / (n - 1)
        first_deri = (full_d[0,1:] - full_d[0,:-1])/h
        return torch.mean(torch.square(first_deri))

    def setup_network(self, **kwargs):

        if kwargs['arch'] == 'deeponet':
            opnet = DeepONet(param_dim=self.param_dim-2, X_dim=self.input_dim, output_dim=self.output_dim, 
            branch_depth=kwargs['branch_depth'], trunk_depth=kwargs['trunk_depth'], width=kwargs['width'],
            lambda_transform=self.lambda_transform)
        elif kwargs['arch'] == 'fno':
            opnet = scalarFNO(param_dim=self.param_dim-2, X_dim=self.input_dim, output_dim=self.output_dim,
            n_modes=kwargs['n_modes'], n_layers=kwargs['n_layers'], hidden_channels=kwargs['hidden_channels'],
            lambda_transform=self.lambda_transform)
        else:
            raise ValueError('Invalid network architecture')

        t = torch.ones(1, self.param_dim-2, dtype=torch.float32)
        # create tensor of all 1s, size 1-by-param_dim for initial guess
        opnet.pde_params_dict = torch.nn.ParameterDict({'D': torch.nn.Parameter(t)})

        return opnet

    def setup_pretrain_dataset(self, ds_opts, noise_opts=None, device='cuda'):

        split = ds_opts['split']
        
        # pat P by 1s at the beginning and end, P is batch-by-param_dim tensor
        # self.dataset['P'] = torch.nn.functional.pad(self.dataset['P'], (1, 1), value=1.0)
        self.dataset.to_device(device)

        self.OpData = OpData(self.dataset['X'], self.dataset['P'], self.dataset['U'])

        n_total = len(self.OpData)
        train_size = int(split * n_total)
        test_size = n_total - train_size

        print(f"Training size: {train_size}, Testing size: {test_size}")

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
        X_data, U_data = self.get_inverse_data()
        self.train_dataset = OpData(X_data, None, U_data)

        # change loss function
        self.loss_dict['data'] = self.get_data_loss_inverse


    def make_prediction(self, net):
        if self.is_pretraining:
            self.make_prediction_pretrain(net)
        else:
            self.make_prediction_inverse(net)

    def make_prediction_pretrain(self, net):
        pass

    def get_data_loss(self, net):
        # each time evaluated on different batch
        P, U = next(self.train_loader)
        U_pred = net(P, self.train_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred, U)
        return loss
    
    def get_data_loss_inverse(self, net):
        # for inverse, P input is pde_param, 
        U_pred = net(net.pde_params_dict['D'], self.train_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred,  self.train_dataset.U)
        return loss
    
    def get_inverse_data(self):
        '''return data for training inverse problem'''
        U = self.dataset['u_dat_train']
        U = torch.reshape(U, (1, -1))
        X = self.dataset['x_dat_train']

        return X, U

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
    def validate_inverse(self, nn):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        D = self.dataset['D_dat'] # shape (N,1)
        # include boundary
        n = self.param_dim
        D = uniform_subsample_with_endpoint(D, n).reshape(1, -1)

        Dpred = self.pad_pde_param(nn.pde_params_dict['D'])

        l2norm = torch.mean(torch.square(D - Dpred))
        linfnorm = torch.max(torch.abs(D - Dpred)) 
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        # if X P U are in the data set, this is pretraining dataset
        # otherwise this is inverse dataset
        if 'X' in self.dataset and 'P' in self.dataset and 'U' in self.dataset:
            self.is_pretraining = True
            self.setup_pretrain_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_pretrain
        else:
            # call super class method
            self.setup_inverse_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_inverse
    
    @torch.no_grad()
    def make_prediction_inverse(self, net):
        # make prediction for inverse problem using NO

        
        x = uniform_subsample_with_endpoint(self.dataset['x_dat'], self.param_dim)
        
        # operator evaluate with NN D
        upred_dat = net(net.pde_params_dict['D'], x)
        self.dataset['upred_dat'] = upred_dat.reshape(-1, 1)

        # operator evaluate with GT D
        D = self.dataset['D_dat'] # shape (1001,1)
        D = uniform_subsample_with_endpoint(D, self.param_dim) # shape (101,1)
        # trim boundary for inference
        D = D[1:-1,0:1] # shape (99,1)
        
        upred_dat = net(D.reshape(1,-1), x) # shape (1, 101)
        self.dataset['upred_gt_dat'] = upred_dat.reshape(-1, 1) # shape (101, 1)

        # predicted D
        D_pred = net.pde_params_dict['D']
        # pad boundary for predicted D
        self.dataset['func_dat'] = self.pad_pde_param(D_pred)
    
    def visualize(self, savedir=None):
        # visualize the results
        if not self.is_pretraining:
            super().visualize(savedir=savedir)
        else:
            pass

    def plot_Dpred(self, savedir=None):
        # Overwrite plot_Dpred to plot predicted D
        # Neural operator make predicted D at sub resolution
        
        n = self.dataset['func_dat'].shape[1]
        x = uniform_subsample_with_endpoint(self.dataset['x_dat'], n)

        fig, ax = plt.subplots()
        # plot ground truth
        ax.plot(self.dataset['x_dat'], self.dataset['D_dat'].flatten(), label='Exact')
        # plot predicted D at sub resolution
        ax.plot(x.flatten(), self.dataset['func_dat'].flatten() , label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_D_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    def plot_upred(self, savedir=None):
        # Overwrite plot_upred in VarPoiProblem.
        # Plot predicted u at sub resolution
        n = self.dataset['func_dat'].shape[1]
        x = uniform_subsample_with_endpoint(self.dataset['x_dat'], n)
        fig, ax = plt.subplots()
        # plot training data
        ax.scatter(self.dataset['x_dat_train'], self.dataset['u_dat_train'], label='data')
        # plot ground truth
        ax.plot(self.dataset['x_dat'], self.dataset['u_dat'], label='Exact')
        # plot predicted u at sub resolution
        ax.plot(x, self.dataset['upred_dat'], label='NN')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')