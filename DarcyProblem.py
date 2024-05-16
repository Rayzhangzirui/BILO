#!/usr/bin/env python
# # define problems for PDE
import os
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from util import generate_grf, griddata_subsample

from BaseProblem import BaseProblem
from DataSet import DataSet
from DenseNet import DenseNet, ParamFunction

class DarcyDenseNet(DenseNet):
    ''' override the embedding function of DenseNet
    This function represent the f
    - div(f grad u) = 1
    A takes the form 9sigmoid(x) + 3, mainly 12 and 3
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        fdepth = kwargs['fdepth']
        fwidth = kwargs['fwidth']
        activation = kwargs['activation']
        output_activation = kwargs['output_activation']
        
        # f(t,x) = 0 at x=0, x=1, t=0, t=1
        self.func_param = ParamFunction(input_dim=2, output_dim=1,fdepth=fdepth, fwidth=fwidth,
                                        activation=activation, output_activation= output_activation,
                                        output_transform=lambda x, u: torch.sigmoid(u) * 9 + 3)
        self.collect_trainable_param()


    def setup_embedding_layers(self):

        # embedding layer
        self.param_embeddings = nn.ModuleDict({'f': nn.Linear( 1, self.width, bias=False)})

        # set requires_grad to False
        for embedding_weights in self.param_embeddings.parameters():
            embedding_weights.requires_grad = False
    
    def output_transform(self, X, u):
        ''' impose 0 boundary condition'''
        # u(x,y) =u_NN(x,t) * x * (1-x) * y * (1-y)
        return u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1] * (1 - X[:,0:1])
    
    def embed_x(self, x):
        '''embed x to the input layer'''
        if self.fourier:
            x_embed = torch.sin(2 * torch.pi * self.fflayer(x))
        x_embed = self.input_layer(x)
        return x_embed

    def embedding(self, x):
        # override the embedding function
        # have to evaluate self.func_param(xcoord) inside the network
        # otherwise self.func_param is not in the computation graph
    
        t_x_coord = x
        x_embed = self.embed_x(x)

        if self.with_param:
            f = self.func_param(t_x_coord) #(n, 1) D at x_res_train
            self.f_eval = f
            self.params_expand['f'] = f
            y_embed = self.param_embeddings['f'](f)
            x_embed += y_embed
        else:
            # if varnilla, u(x), not function of unkown
            self.params_expand['f'] = self.func_param(t_x_coord)
            self.f_eval = self.params_expand['f']
            
        return x_embed

    def embedding_to_u(self, X):
        # X is the embedded input, linear combination of the "features"
        Xtmp = self.act(X)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = hidden_layer(Xtmp)
            if self.use_resnet:
                hidden_output += Xtmp  # ResNet connection
            hidden_output = self.act(hidden_output)
            Xtmp = hidden_output
        
        u = self.output_layer(Xtmp)
        
        return u

    def forward(self, x):
        
        X = self.embedding(x)
        
        u = self.embedding_to_u(X)

        u = self.output_transform(x, u)
        return u
    
class DarcyProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2 # x, y
        self.output_dim = 1
        self.opts=kwargs

        self.dataset = DataSet(kwargs['datafile'])
        self.testcase = kwargs['testcase']

    def residual(self, nn, X):
        ''' - div(f grad u) = 1'''

        X.requires_grad_(True)

        x = X[:, 0:1]
        y = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((x,y), dim=1)

        # Forward pass through the network
        u_pred = nn(nn_input)

        # Get the predicted f
        f = nn.f_eval

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
        u_pred = net(self.dataset['X_dat_train'])
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))
        return loss
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['X_dat_train']
        y = net.func_param(x)
        return torch.mean(torch.square(y - self.dataset['f_dat_train']))
    
    def get_grad(self, net):
        
        X = self.dataset['X_res_train']
        X.requires_grad_(True)

        x = X[:, 0:1]
        y = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((x,y), dim=1)

        f = net.func_param(nn_input)

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

        pde_param = self.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        net = DarcyDenseNet(**kwargs,
                            params_dict={},
                            trainable_param = ['f'])
        net.setup_embedding_layers()
        
        return net

    
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']
        
        with torch.no_grad():
            self.dataset['upred_dat'] = net(x_dat)
            self.dataset['upred_res'] = net(x_res)
            self.dataset['fpred_res'] = net.params_expand['f']

            self.dataset['upred_dat_train'] = net(x_dat_train)
            self.dataset['upred_res_train'] = net(x_res_train)
        
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_meshgrid('u_res','upred_res',savedir=savedir)
        self.plot_meshgrid('f_res','fpred_res',savedir=savedir)

    def create_dataset_from_file(self, dsopt):
        # use all data for training
        dataset = self.dataset
        
        uname = f'u{self.testcase}'
        fname = f'A{self.testcase}'

        u = dataset[uname]
        f = dataset[fname]
        
        # collect X and u from all time, for residual loss
        dataset['X_res'] = dataset['X']
        dataset['u_res'] = u
        dataset['f_res'] = f
        dataset['X_dat'] = dataset['X_res']
        dataset['u_dat'] = dataset['u_res']
        dataset['f_dat'] = dataset['f_res']
        
        dataset['X_res_train'] = dataset['X']
        dataset['u_res_train'] = u
        dataset['f_res_train'] = f

        # use all time data
        dataset['X_dat_train'] = dataset['X']
        dataset['u_dat_train'] = u
        dataset['f_dat_train'] = f
        
        # remove redundant data
        for i in range(10):
            if i != self.testcase:
                dataset.pop(f'u{i}',None)
                dataset.pop(f'f{i}',None)
        
        dataset.printsummary()
    
    def validate(self, nn):
        '''compute l2 error and linf error of inferred f(t,x)'''
        
        x = self.dataset['X_res']
        f = self.dataset['f_res']
        with torch.no_grad():
            f_pred = nn.func_param(x)
            err = f - f_pred
            l2norm = torch.mean(torch.square(err))
            linfnorm = torch.max(torch.abs(err)) 
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item()}

    def setup_dataset(self, ds_opts, noise_opts=None):
        ''' downsample for training'''
        
        self.create_dataset_from_file(ds_opts)
        self.dataset.to_torch()

        if noise_opts['use_noise']:
            print('add noise to training data')
            x = self.dataset['X_dat_train'][:,1:2]
            noise = torch.zeros_like(self.dataset['u_dat_train'])
    
            tmp = generate_grf(x, noise_opts['variance'], noise_opts['length_scale'])
            noise[:,0] = tmp.squeeze()

            self.dataset['noise'] = noise
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']
    
    def plot_meshgrid(self, name_true, name_pred, savedir=None):
        # plot u at X_res, 
        
        u = self.dataset[name_true]
        u_pred = self.dataset[name_pred]
        
        # reshape to 2D
        Nx = int(self.dataset['nx'])
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
    optobj.opts['nn_opts']['with_func'] = True
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

