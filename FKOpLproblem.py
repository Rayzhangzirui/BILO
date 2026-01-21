#!/usr/bin/env python
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from MatDataset import MatDataset 
from DeepONet import DeepONet, OpData, scalarFNO
# from BaseOperator import BaseOperator
from FKproblem import FKproblem
from util import griddata_subsample, generate_grf, error_logging_decorator

from itertools import cycle
from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO

class scalarFNO2d(torch.nn.Module):
    # For comparision with DeepONet
    def __init__(self, param_dim=1, X_dim=1, output_dim=1, n_modes = 16, 
                hidden_channels=32,
                positional_embedding='grid',
                n_layers=2,
                lambda_transform=None):

        super(scalarFNO2d, self).__init__()

        self.param_dim = param_dim
        self.X_dim = X_dim
        self.output_dim = output_dim
        self.pde_param = None

        self.fno = FNO(n_modes=(n_modes,n_modes), 
            in_channels=param_dim, out_channels=output_dim,
            n_layers=n_layers,
            positional_embedding='grid',
            hidden_channels=hidden_channels)
            
        self.lambda_transform = lambda_transform

    def forward(self, P_input, X_input):
        """
        P_input: (B, k)  - PDE parameters
        X_input is (2, Nx, Nt) - spatial-temporal grid, X_int[0] is t, X_int[1] is x
        Output:   (B, N) or (B, out_channels, N)
        """
        B, k = P_input.shape  # (batch, k)
        _ , nx, nt = X_input.shape  # (Nx, Nt)
        
        # Expand to shape (batch_size, 2, 1, 1)
        params_reshaped = P_input.unsqueeze(-1).unsqueeze(-1)

        # Now broadcast to (batch_size, 2, N, N) by expanding
        P_input_expanded = params_reshaped.expand(-1, -1, nx, nt)

    
        out = self.fno(P_input_expanded)  # shape: (B, out_channels, N, N)

        out2 = out.squeeze(1)  # shape: (B, N, N)

        out3 = self.lambda_transform(X_input, out2)

        return out3



class FKOperatorLearning(FKproblem):
    def __init__(self, **kwargs):
        # super().__init__(**kwargs)

        self.input_dim = 2
        self.output_dim = 1
        self.param_dim = 2

        self.dataset = MatDataset(kwargs['datafile'])

        self.D = self.dataset['D']
        self.RHO = self.dataset['RHO']

        
        self.testcase = kwargs.get('testcase', -1)

        self.loss_dict = {'data': self.get_data_loss, 'tdata': self.get_test_loss}

        self.arch = None
        self.method = None

        self.dat_use_res = kwargs['dat_use_res']

    def setup_network(self, **kwargs):

        if self.arch == 'deeponet':
            self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]

            opnet = DeepONet(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim, 
            branch_depth=kwargs['branch_depth'], trunk_depth=kwargs['trunk_depth'], width=kwargs['width'],
            lambda_transform=self.lambda_transform)

            

        elif self.arch == 'fno':
            self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[1]) ** 2)+ u * X[1] * (1 - X[1]) * X[0]

            opnet = scalarFNO2d(param_dim=self.param_dim, X_dim=self.input_dim, output_dim=self.output_dim,
            n_modes=kwargs['n_modes'], n_layers=kwargs['n_layers'], hidden_channels=kwargs['hidden_channels'],
            lambda_transform=self.lambda_transform)

        else:
            raise ValueError('Invalid network architecture')

        D0 = 1.0
        rho0 = 1.0        
        opnet.pde_params_list = ['rD', 'rRHO']
        opnet.pde_params_dict = torch.nn.ParameterDict({'rD': torch.nn.Parameter(torch.tensor([[D0]], dtype=torch.float32)),
                                                        'rRHO': torch.nn.Parameter(torch.tensor([[rho0]], dtype=torch.float32))})

        return opnet

    def get_inverse_data(self):
        '''
        Return data for training inverse problem
        These are from BILO/PINN dataset. Need reshape
        '''
        U = self.dataset['u_dat_train']
        X = self.dataset['X_dat_train']
        return X, U
    
    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        # if X P U are in the data set, this is pretraining dataset
        # otherwise this is inverse dataset
        if self.method == 'init':
            self.setup_pretrain_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_pretrain
        else:
            # call super class method
            self.setup_inverse_dataset(ds_opts, noise_opts=noise_opts, device=device)
            self.validate = self.validate_inverse
    
    def X_to_grid(self, X, nt, nx):
        # for pretraining dataset
        # reshap X N-by-2 to (2, Nt, Nx)
        X_reshaped = X.t().reshape(2, nx, nt) 
        X2 = X_reshaped.transpose(1, 2) # swap to (2, Nt, Nx)
        return X2
    
    def config_traintype(self, traintype):
        arch, method = traintype.split('-')
        self.arch = arch
        self.method = method
        assert arch in {'fno', 'deeponet'}
        assert method in {'inv', 'init'}

    def setup_pretrain_dataset(self, ds_opts, noise_opts=None, device='cuda'):

        split = ds_opts['split']
        self.dataset.to_device(device)

        if self.arch == 'fno':
            # if FNO, reshape to 2D
            nx = self.dataset['x'].shape[1]
            nt = self.dataset['t'].shape[1]
            # reshap X N-by-2 to (2, Nt, Nx)
            X = self.dataset['X']
            
            self.dataset['X'] = self.X_to_grid(X, nt, nx)
            # reshape U from (batch, Nt*Nx) to (batch, nx, nt)
            U2 = torch.reshape(self.dataset['U'], (-1, nt, nx))
            self.dataset['U'] = U2.transpose(1, 2)  # swap to (batch,nt, nx)
            # end of FNO reshaping

        self.OpData = OpData(self.dataset['X'], self.dataset['P'], self.dataset['U'])

        n_total = min(len(self.OpData), ds_opts['N_example'])
        train_size = int(split * n_total)
        test_size = n_total - train_size

        # Randomly shuffle the indices
        total_indices = torch.randperm(len(self.OpData))

        # Split indices into training and testing
        train_indices = total_indices[:train_size]
        test_indices = total_indices[train_size:n_total]

        self.train_dataset = OpData(self.OpData.X, self.OpData.P[train_indices], self.OpData.U[train_indices])
        self.test_dataset = OpData(self.OpData.X, self.OpData.P[test_indices], self.OpData.U[test_indices])
        print(f"Training size: {len(self.train_dataset)}, Testing size: {len(self.test_dataset)}")

        batch_size = ds_opts['batch_size']

        self.train_loader = cycle(DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True))
    
    def setup_inverse_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        # call super class method
        super().setup_dataset(ds_opts, noise_opts=noise_opts, device=device)
        
        if self.arch == 'fno':
            self.X = torch.cat((self.dataset['X_res_train_gt'].unsqueeze(0), self.dataset['X_res_train_gx'].unsqueeze(0)), dim=0)
        else:
            self.X = self.dataset['X_res_train']
        
        # change loss function
        self.loss_dict['data'] = self.get_data_loss_inverse

    def get_data_loss(self, net):
        # each time evaluated on different batch
        P, U = next(self.train_loader)
        U_pred = net(P, self.train_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred, U)
        return loss

    
    def get_data_finalt(self, net):

        upred_res = net(net.pde_params_dict, self.X) #(batch=1, Nt, Nx)

        upred_res.squeeze_() # (Nt, Nx)
 
        # downsample upred_res to match U_data
        upred_finalt = upred_res[-1, :]

        # reshaped to (batch=1, channel=1, Nx)
        upred_finalt_reshaped = upred_finalt.reshape(1, 1, -1)

        u_dat_train = self.dataset['u_dat_train'].squeeze()

        # prediction at training data
        # upred_finalt needs to be (batch, channel, n1, n2 ...)
        # size should be (o1, o2, ...)
        upred_dat_train =  torch.nn.functional.interpolate(upred_finalt_reshaped, size=u_dat_train.shape, 
                            mode='linear',align_corners=True).squeeze()

        return upred_dat_train, u_dat_train, upred_res

    def get_data_loss_inverse(self, net):
        # for FNO
        if self.arch == 'fno':
            upred_dat_train, u_dat_train, upred_res = self.get_data_finalt(net)
            loss = torch.nn.functional.mse_loss(upred_dat_train, u_dat_train)
        else:
            upred_dat_train = net(net.pde_params_dict, self.dataset['X_dat_train']).squeeze()
            u_dat_train = self.dataset['u_dat_train'].squeeze()
            loss = torch.nn.functional.mse_loss(upred_dat_train, u_dat_train)
        return loss
    
    
    def get_test_loss(self, net):
        U_pred = net(self.test_dataset.P, self.test_dataset.X)
        loss = torch.nn.functional.mse_loss(U_pred, self.test_dataset.U)
        return loss

    def validate_pretrain(self, net):
        # validate the network
        net.eval()
        loss = self.get_test_loss(net)
        return {'tdata': loss.item()}
    
    def validate_inverse(self, net):
        # take pde_param, tensor of trainable parameters
        # return dictionary of metrics
        return {'rD': net.pde_params_dict['rD'].item(),
                'rRHO': net.pde_params_dict['rRHO'].item()}

    
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
        # Make prediction on ALL training examples
        P = self.dataset['P']
        U_pred = net(P, self.dataset['X'])
        err = U_pred - self.dataset['U']
        
        mse = torch.mean(err**2, dim=1)
        relmse = mse / torch.mean(self.dataset['U']**2, dim=1)

        # save prediction only
        original_dataset = self.dataset
        new_dataset = MatDataset()
        new_dataset['x'] = original_dataset['x']
        new_dataset['t'] = original_dataset['t']
        new_dataset['P'] = original_dataset['P']

        # find the solution corresponding to D=2 and RHO = 2
        idx_D2_RHO2 = ((original_dataset['P'][:, 0]-2).abs() < 1e-8) & ((original_dataset['P'][:, 1]-2).abs() < 1e-8)

        if torch.sum(idx_D2_RHO2) > 0:

            U_gt = original_dataset['U'][idx_D2_RHO2]

            # compute loss landscape
            mse_landscape = torch.mean((U_pred - U_gt)**2, dim=1)
            new_dataset['mse_landscape'] = mse_landscape.cpu().numpy()

            relmse_landscape = mse_landscape / torch.mean(U_gt**2, dim=1)
            new_dataset['relmse_landscape'] = relmse_landscape.cpu().numpy()


        # Save all predictions in dataset
        new_dataset['mse'] = mse.cpu().numpy()
        new_dataset['relmse'] = relmse.cpu().numpy()
        
        # Also select 9 evenly spaced samples for visualization
        n = P.shape[0]
        idx = torch.linspace(0, n-1, 9, dtype=torch.long)
        new_dataset['idx_vis'] = idx
        
        # Store individual predictions for visualization
        for i in range(9):
            new_dataset[f'U_{i}'] = U_pred[idx[i]].cpu().numpy()   
            new_dataset[f'U_{i}_gt'] = original_dataset['U'][idx[i]].cpu().numpy()
        
        self.dataset = new_dataset

    @error_logging_decorator
    def visualize_pretrain(self, savedir=None):
        # Convert dataset to numpy if needed
        self.dataset.to_np()
        idx = self.dataset['idx_vis']
        
        nx = self.dataset['x'].shape[1]
        nt = self.dataset['t'].shape[1]
        
        # Create a 3x3 grid of subplots for 9 examples
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        for i in range(9):
            # Create individual figure for each example with 3 subplots
            fig_individual, ax = plt.subplots(1, 3, figsize=(12, 4))
            
            Upred = self.dataset[f'U_{i}'].reshape(nt, nx)
            Ugt = self.dataset[f'U_{i}_gt'].reshape(nt, nx)
            # Compute error
            err = Upred - Ugt

            # Plot predicted U
            im_pred = ax[0].imshow(Upred, aspect='equal')
            ax[0].set_title(f'U_pred_{i}')

            # Plot ground-truth U
            im_true = ax[1].imshow(Ugt, aspect='equal')
            ax[1].set_title(f'U_{i}_gt')

            # Plot error with a distinct colormap and add a colorbar
            im_err = ax[2].imshow(err, aspect='equal', cmap='bwr')
            ax[2].set_title(f'Error_{i}')
            fig_individual.colorbar(im_err, ax=ax[2], fraction=0.046, pad=0.04)

            # Optionally save individual figures
            if savedir is not None:
                path = os.path.join(savedir, f'fig_U_{i}.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f'Figure saved to {path}')

            plt.close(fig_individual)

            
    
    @torch.no_grad()
    def make_prediction_inverse(self, net):
        # make prediction at original X_dat and X_res
        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']
        
        if self.arch == 'fno':
            upred_dat_train, u_dat_train, upred_res_train = self.get_data_finalt(net)
            self.dataset['upred_dat_train'] = upred_dat_train #(N_dat_train)
            self.dataset['upred_res_train'] = upred_res_train #(Nt, Nx)
        else:
            self.dataset['upred_dat'] =       net(net.pde_params_dict, x_dat)
            self.dataset['upred_res'] =       net(net.pde_params_dict, x_res)
            self.dataset['upred_dat_train'] = net(net.pde_params_dict, x_dat_train)
            self.dataset['upred_res_train'] = net(net.pde_params_dict, x_res_train)


    def visualize_inverse(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        # ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['upred_res'], fname = 'fig_upred_res.png', savedir=savedir)
        self.plot_scatter(self.dataset['X_dat_train'], self.dataset['upred_dat_train'], fname = 'fig_upred_dat_train.png', savedir=savedir)
        self.plot_scatter(self.dataset['X_res_train'], self.dataset['upred_res_train'].reshape(-1,1,order='F'), fname = 'fig_upred_res_train.png', savedir=savedir)
        self.plot_scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], fname = 'fig_u_dat_train.png', savedir=savedir)
        self.plot_upred_dat(savedir=savedir)

    @error_logging_decorator
    def plot_upred_dat(self, savedir=None):
        # plot prediciton at final time
        fig, ax = plt.subplots()
        
        # plot GT solution
        x = self.dataset['X_dat'][:, 1]
        ax.plot(x, self.dataset['u_dat'], label='exact')

        # plot NN prediction
        if self.arch == 'fno':
            ax.plot(self.dataset['X_res_train_gx'][-1,:].flatten(), self.dataset['upred_res_train'][-1,:].flatten(), label='pred')
        else:
            ax.plot(x, self.dataset['upred_dat'].flatten(), label='pred')
        
        # plot train data
        x_train = self.dataset['X_dat_train'][:, 1]
        t_train = self.dataset['X_dat_train'][:, 0]
        ax.scatter(x_train, self.dataset['u_dat_train'], c=t_train, cmap='viridis', label='train')
        
        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xdat.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
        plt.close(fig)

if __name__ == "__main__":
    import sys
    # test the OpMatDataset class
    filename  = sys.argv[1]
    fkoperator = FKOperatorLearning(datafile=filename)

    fkoperator.init(batch_size=1000)
    fkoperator.train(max_iter=1000, print_every=100)

    fkoperator.save_data('pred.mat')


    