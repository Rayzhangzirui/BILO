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

class varFKDenseNet(DenseNet):
    ''' override the embedding function of DenseNet
    This function represent the F
    u_t = D u_xx + F(x,t)
    impose the condition F(0) = F(1) = 0
    should also by possitive
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        fdepth = kwargs['fdepth']
        fwidth = kwargs['fwidth']
        activation = kwargs['activation']
        output_activation = kwargs['output_activation']
        
        # f(t,x) = 0 at x=0, x=1, t=0, t=1
        self.func_param = ParamFunction(input_dim=2, output_dim=1,fdepth=fdepth, fwidth=fwidth,
                                        activation=activation, output_activation=output_activation,
                                        output_transform=lambda x, u: u * x[:,1:2] * (1 - x[:,1:2]) * x[:,0:1] * (1 - x[:,0:1]))
        self.collect_trainable_param()


    def setup_embedding_layers(self):

        # embedding layer
        self.param_embeddings = nn.ModuleDict({'f': nn.Linear( 1, self.width, bias=False)})

        # set requires_grad to False
        for embedding_weights in self.param_embeddings.parameters():
            embedding_weights.requires_grad = False
    
    def output_transform(self, X, u):
        # u(x,t) = u0(x) + u_NN(x,t) * x * (1-x) * t
        return (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2) + u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]
    
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
            self.params_expand['f'] = f
            y_embed = self.param_embeddings['f'](f)
            x_embed += y_embed
        else:
            # if varnilla, u(x), not function of unkown
            self.params_expand['f'] = self.func_param(t_x_coord)
            
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
    
    # def variation(self, x, z):
    #     '''variation of u w.r.t u0
    #     '''
    #     # Need to update the params_expand['u0'] to z for both with_param=True and False
    #     # 
    #     self.params_expand['f'] = z

    #     if self.with_param:
    #         x_embed = self.embed_x(x)
    #         z_embed = self.param_embeddings['f'](z)
    #         X = x_embed + z_embed
    #     else:
    #         X = self.embed_x(x)

    #     u = self.embedding_to_u(X)
    #     u = self.output_transform(x, u)
    #     return u


class varFKproblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2 # x, t
        self.output_dim = 1
        self.opts=kwargs

        self.dataset = MatDataset(kwargs['datafile'])
        self.D = self.dataset['D']
        
        self.testcase = kwargs['testcase']

        # ic, u(x) = 0.5*sin(pi*x)^2
        # bc, u(t,0) = 0, u(t,1) = 0
        # transform: u(x,t) = u0(x) + u_NN(x,t) * x * (1-x) * t
        self.lambda_transform = lambda X, u: (0.5 * torch.sin(np.pi * X[:,1:2]) ** 2)+ u * X[:,1:2] * (1 - X[:,1:2]) * X[:,0:1]



    def residual(self, nn, X):
        X.requires_grad_(True)

        t = X[:, 0:1]
        x = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((t,x), dim=1)

        # Forward pass through the network
        u_pred = nn(nn_input)

        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u_pred)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=v, create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=v, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=v, create_graph=True)[0]

        # Compute the right-hand side of the PDE
        rhs =  self.D * u_xx + nn.params_expand['f']
        
        # Compute the residual
        res = u_t - rhs
        
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
    
    def get_l2grad(self, net):
        # regularization |D'(x)|^2
        X = self.dataset['X_dat_train']
        X.requires_grad_(True)

        t = X[:, 0:1]
        x = X[:, 1:2]

        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat((t,x), dim=1)

        f = net.func_param(nn_input)
        f_x = torch.autograd.grad(f, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(f))[0]
        f_t = torch.autograd.grad(f, t,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(f))[0]
        
        return torch.mean(torch.square(f_x) + torch.square(f_t))

        

    def print_info(self):
        # print parameter
        print('D = ', self.D)

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        pde_param = self.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        net = varFKDenseNet(**kwargs,
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
        
    def plot_scatter(self, X, u, fname = 'fig_scatter.png', savedir=None):
        ''' plot u vs x, color is t'''
        x = X[:,1]
        t = X[:,0]
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.scatter(x, u, c=t, cmap='viridis')

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_meshgrid('u_res','upred_res',savedir=savedir)
        self.plot_meshgrid('f_res','fpred_res',savedir=savedir)

        ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['upred_res'], fname = 'fig_upred_res.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_res'], self.dataset['fpred_res'], fname = 'fig_fpred_res.png', savedir=savedir)
        
        ax, fig = self.plot_scatter(self.dataset['X_dat'], self.dataset['u_dat'], fname = 'fig_u_dat.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_dat'], self.dataset['f_dat'], fname = 'fig_f_dat.png', savedir=savedir)

        ax, fig = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['u_dat_train'], fname = 'fig_u_dat_train.png', savedir=savedir)
        ax, fig = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['f_dat_train'], fname = 'fig_f_dat_train.png', savedir=savedir)

        self.plot_sample(savedir=savedir)

    def create_dataset_from_file(self, dsopt):
        dataset = self.dataset
        
        uname = f'u{self.testcase}'
        fname = f'f{self.testcase}'

        u = dataset[uname]
        f = dataset[fname]
        gt = dataset['gt']
        gx = dataset['gx']
        Nt_full, Nx_full = u.shape
        
        # downsample size
        Nt = dsopt['Nt']
        Nx = dsopt['Nx']
        dataset['Nt'] = Nt
        dataset['Nx'] = Nx

        # collect X and u from all time, for residual loss
        dataset['X_res'] = np.column_stack((gt.reshape(-1, 1), gx.reshape(-1, 1)))
        dataset['u_res'] = u.reshape(-1, 1)
        dataset['f_res'] = f.reshape(-1, 1)

        dataset['X_dat'] = dataset['X_res']
        dataset['u_dat'] = dataset['u_res']
        dataset['f_dat'] = dataset['f_res']

        # for training, downsample griddata and vectorize
        gt_sub, gx_sub, u = griddata_subsample(gt, gx, u, Nt, Nx)
        _, _, f = griddata_subsample(gt, gx, f, Nt, Nx)
        
        dataset['X_res_train'] = np.column_stack((gt_sub.reshape(-1, 1), gx_sub.reshape(-1, 1)))
        dataset['u_res_train'] = u.reshape(-1, 1)
        dataset['f_res_train'] = f.reshape(-1, 1)

        # use all time data
        dataset['X_dat_train'] = dataset['X_res_train']
        dataset['u_dat_train'] = dataset['u_res_train']
        dataset['f_dat_train'] = dataset['f_res_train']
        
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
    
            tmp = generate_grf(x, noise_opts['std'], noise_opts['length_scale'])
            noise[:,0] = tmp.squeeze()

            self.dataset['noise'] = noise
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']
    
    def plot_meshgrid(self, name_true, name_pred, savedir=None):
        # plot u at X_res, 
        
        u = self.dataset[name_true]
        u_pred = self.dataset[name_pred]
        
        # reshape to 2D
        Nt, Nx = self.dataset['gt'].shape
        u = u.reshape(Nt, Nx)
        u_pred = u_pred.reshape(Nt, Nx)
        err = u - u_pred

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # 2D plot
        cax = ax[0].imshow(u_pred , cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
        ax[0].set_title('NN')
        fig.colorbar(cax, ax=ax[0])

        cax = ax[1].imshow(u , cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
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
    
    def plot_sample(self, savedir=None):
        '''plot distribution of collocation points'''
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['X_res_train'][:, 1], self.dataset['X_res_train'][:, 0], s=2.0, marker='.', label='X_res_train')
        ax.scatter(self.dataset['X_dat_train'][:, 1], self.dataset['X_dat_train'][:, 0], s=2.0, marker='s', label='X_dat_train')
        # same xlim ylim
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        # legend
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_Xdist.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'varfk'
    optobj.opts['nn_opts']['with_func'] = True
    optobj.opts['pde_opts']['trainable_param'] = 'f'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = varFKproblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])

