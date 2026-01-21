#!/usr/bin/env python
# # define problems for PDE
import os
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from util import generate_grf, griddata_subsample

from BaseProblem import BaseProblem
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
    
class DarcyProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2 # x, y
        self.output_dim = 1
        self.opts=kwargs

        # self.loss_dict['l2grad'] = self.get_l2grad
        self.loss_dict['l1grad'] = self.get_l1grad


        self.dataset = MatDataset(kwargs['datafile'])
        self.testcase = kwargs['testcase']

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
        res = - (f_x + f_y) - 1
        
        
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
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_dat_train']
        y = net.pde_params_dict['f'](x)
        return torch.mean(torch.square(y - self.dataset['f_dat_train']))
    
    def get_grad(self, net):
        
        X = self.dataset['X_res_train']
        X.requires_grad_(True)

        x = X[:, 0:1]
        y = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((x,y), dim=1)

        f = net.pde_params_dict['f'](nn_input)

        f_x = torch.autograd.grad(f, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(f))[0]
        f_y = torch.autograd.grad(f, y,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(f))[0]
        
        return f_x, f_y

    def get_l1grad(self, net):
        '''regularization |\grad f(x)|_1
        = |f_x| + |f_y|
        '''
        f_x, f_y = self.get_grad(net)
        return torch.mean(torch.abs(f_x) + torch.abs(f_y))
    
    def get_l2grad(self, net):
        '''regularization |\grad f(x)|_2
        = (|f_x|^2 + |f_y|^2)
        '''
        f_x, f_y = self.get_grad(net)
        return torch.mean(torch.square(f_x) + torch.square(f_y))


    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        # f takes the form 9sigmoid(x) + 3, mainly 12 and 3
        self.param_fun = ParamFunction(input_dim=2, output_dim=1,
            fdepth=kwargs['fdepth'], fwidth=kwargs['fwidth'],fsiren=kwargs['fsiren'],
                                    activation=kwargs['activation'], output_activation=kwargs['output_activation'],
                                    # output_transform=lambda x, u: torch.exp(u))
                                    output_transform=lambda x, u: torch.sigmoid(u)*9+3)
                
        self.all_params_dict = {'f': self.param_fun}

        net = DarcyDenseNet(**kwargs,
                            all_params_dict= self.all_params_dict,
                            trainable_param = ['f'])
        
        return net

    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        x_dat = self.dataset['X_dat_train']
        x_res = self.dataset['X_res_train']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']
        
        self.dataset['upred_dat'] = net(x_dat, net.pde_params_dict)
        self.dataset['upred_res'] = net(x_res, net.pde_params_dict)
        self.dataset['fpred_res'] = net.pde_params_dict['f'](x_res)

        self.dataset['upred_dat_train'] = net(x_dat_train, net.pde_params_dict)
        self.dataset['upred_res_train'] = net(x_res_train, net.pde_params_dict)
        
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_meshgrid('u_res','upred_res',savedir=savedir)
        self.plot_meshgrid('f_res','fpred_res',savedir=savedir)

    def create_dataset_from_file(self, dsopt):
        # use all data for training
        dataset = self.dataset
        dataset.to_np()
        u = dataset['u'][:,:,self.testcase-1].squeeze()
        f = dataset['A'][:,:,self.testcase-1].squeeze()
        dataset.pop(f'u',None)
        dataset.pop(f'A',None)
        
        gx = dataset['gx']
        gy = dataset['gy']

        dataset['X_dat'] = np.column_stack((gx.reshape(-1, 1,order='F'), gy.reshape(-1, 1,order='F')))
        dataset['u_dat'] = u.reshape(-1, 1,order='F')
        dataset['f_dat'] = f.reshape(-1, 1,order='F')
        
        dataset['X_res'] = np.column_stack((gx.reshape(-1, 1,order='F'), gy.reshape(-1, 1,order='F')))
        dataset['u_res'] = u.reshape(-1, 1,order='F')
        dataset['f_res'] = f.reshape(-1, 1,order='F')

        # same for training
        dataset['X_dat_train'] = dataset['X_dat']
        dataset['u_dat_train'] = dataset['u_dat']
        dataset['f_dat_train'] = dataset['f_dat']
        
        dataset['X_res_train'] = dataset['X_res']
        dataset['u_res_train'] = dataset['u_res']
        dataset['f_res_train'] = dataset['f_res']
        
        dataset.printsummary()
    
    def validate(self, nn):
        '''compute l2 error and linf error of inferred f(t,x)'''
        
        x = self.dataset['X_res']
        f = self.dataset['f_res']
        with torch.no_grad():
            f_pred = nn.pde_params_dict['f'](x)
            err = f - f_pred
            l2norm = torch.mean(torch.square(err))
            linfnorm = torch.max(torch.abs(err)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        ''' downsample for training'''
        
        self.create_dataset_from_file(ds_opts)
        self.dataset.to_torch()

        if noise_opts['use_noise']:
            print('add noise to training data')
            x = self.dataset['X_dat_train'][:,1:2]
            noise = torch.zeros_like(self.dataset['u_dat_train'])
    
            tmp = generate_grf(x, noise_opts['std'], noise_opts['length_scale'])
            noise[:,0] = tmp.squeeze()

            self.dataset['noise'] = noise
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']
        
        self.dataset.to_device(device)
    
    def plot_meshgrid(self, name_true, name_pred, savedir=None):
        # plot u at X_res, 
        
        u = self.dataset[name_true]
        u_pred = self.dataset[name_pred]
        
        # reshape to 2D
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
    

if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'darcy'
    optobj.opts['pde_opts']['trainable_param'] = 'f'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = DarcyProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir='tmp')

