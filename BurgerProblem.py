#!/usr/bin/env python
# Burger's equation, infer initial condition
import torch
from torch import nn
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from util import generate_grf, add_noise,  griddata_subsample, error_logging_decorator

from BaseProblem import BaseProblem
from MatDataset import MatDataset
from DenseNet import DenseNet, ParamFunction

class BurgerDenseNet(DenseNet):
    ''' override the embedding function of DenseNet'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_param_expand(self, x, pde_params_dict=None):
        '''
        get the embedding of the pde parameter
        n is the size of the input x
        '''
        for name, param in pde_params_dict.items():
            # expand the parameter to the same size as as n
            if isinstance(param, torch.Tensor):
                # might be x only for plotting variation
                n = x.shape[0]
                self.params_expand[name] = param.expand(n, -1)
            elif isinstance(param, nn.Module):
                # X = (t, x) is passed to the network
                self.params_expand[name] = param(x[:, 1:2])
            else:
                raise ValueError(f'Unknown type for parameter {name}')
    
    def output_transform(self, x, u, param):
        # override the output transform attribute of DenseNet
        u0 = self.params_expand['u0']

        return u * x[:, 1:2] * (1 - x[:, 1:2]) * x[:, 0:1] + u0



class BurgerProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.opts=kwargs
 
        self.testcase = kwargs['testcase']

        self.loss_dict['l2grad'] = self.get_l2grad
        
        self.dataset = MatDataset(kwargs['datafile'])
        self.v = self.dataset['v']
    
    def residual(self, nn, X_in):
        
        X_in.requires_grad_(True)

        t = X_in[:, 0:1]
        x = X_in[:, 1:2]
        
        # Concatenate sliced tensors to form the input for the network
        X = torch.cat((t,x), dim=1)

        u = nn(X, nn.pde_params_dict)
        
        u_t = torch.autograd.grad(u, t,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = torch.autograd.grad(u, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        
        
        res = u_t + self.v * u_x * u

        return res, u

    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['X_res_train'])
        return res, pred
    
    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['X_dat_train'], net.pde_params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))
        return loss
    
    def get_l2grad(self, net):
        # regularization |D'(x)|^2
        x = self.dataset['x_ic_train']
        x.requires_grad_(True)
        
        D = net.pde_params_dict['u0'](x)
        D_x = torch.autograd.grad(D, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(D))[0]
        return torch.mean(torch.square(D_x))

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        self.param_fun = ParamFunction(fdepth=kwargs['fdepth'], fwidth=kwargs['fwidth'],
                                    fsiren=kwargs['fsiren'],
                                    activation=kwargs['activation'], output_activation=kwargs['output_activation'],
                                    output_transform=lambda x, u: u * x * (1.0 - x) )
                
        self.all_params_dict = {'u0': self.param_fun}

        net = BurgerDenseNet(**kwargs,
                        all_params_dict= self.all_params_dict,
                        trainable_param = self.opts['trainable_param'])
        
        return net

    
    def griddata_to_tensor(self, gt, gx, u):
        '''convert grid data to tensor for training'''
        X_res = np.column_stack((gt.reshape(-1, 1), gx.reshape(-1, 1)))
        u_res = u.reshape(-1, 1)

        # X_dat is last row of gt(end,:), gx(end,:)
        X_dat = np.column_stack((gt[-1, :].reshape(-1, 1), gx[-1, :].reshape(-1, 1)))
        u_dat = u[-1, :].reshape(-1, 1)

        # x_ic is x coord only
        x_ic = gx[-1, :].reshape(-1, 1)
        u_ic = u[0, :].reshape(-1, 1)

        return X_res, u_res, X_dat, u_dat, x_ic, u_ic

    def create_dataset_from_file(self, dsopt):
        '''create dataset from file'''
        assert self.dataset is not None, 'datafile provide, dataset should not be None'
        dataset = self.dataset
        
        uname = f'u{self.testcase}'
        icname = f'ic{self.testcase}'
        
        N = 1001
        # get data from file
        u = dataset[uname]
        gt = dataset['gt']
        gx = dataset['gx']
        
        # downsample size
        Nt = dsopt['Nt']
        Nx = dsopt['Nx']
        dataset['Nt'] = Nt
        dataset['Nx'] = Nx

        # for training X_dat, can be different from Nx
        X_dat = np.column_stack((gt[-1, :].reshape(-1, 1), gx[-1, :].reshape(-1, 1)))
        u_dat = u[-1, :].reshape(-1, 1)
        idx = np.linspace(0, N-1, dsopt['N_dat_train'], dtype=int)
        dataset['X_dat_train'] = X_dat[idx, :]
        dataset['u_dat_train'] = u_dat[idx, :]

        # for x_ic, can be different from Nx, used to evalute l2grad
        idx = np.linspace(0, N-1, dsopt['N_ic_train'], dtype=int)
        dataset['x_ic_train'] = gx[-1, :].reshape(-1, 1)[idx, :]
        dataset['u_ic_train'] = u[0, :].reshape(-1, 1)[idx, :]


        # for testing and plotting
        dataset['X_res'], dataset['u_res'], dataset['X_dat'], dataset['u_dat'], dataset['x_ic'], dataset['u_ic'] = self.griddata_to_tensor(gt, gx, u)

        # for training
        gt, gx, u = griddata_subsample(gt, gx, u, Nt, Nx)
        dataset['X_res_train'], dataset['u_res_train'],_, _, _, _ = self.griddata_to_tensor(gt, gx, u)
        
        # remove redundant data
        for i in range(10):
            if i != self.testcase:
                dataset.pop(f'u{i}',None)
                dataset.pop(f'ic{i}',None)
        
        dataset.printsummary()

    def setup_dataset(self, dsopt, noise_opt, device='cuda'):
        '''add noise to dataset'''
        
        self.create_dataset_from_file(dsopt)
        
        self.dataset.to_torch()

        if noise_opt['use_noise']:
            x = self.dataset['X_dat_train'][:, 1]
            noise = generate_grf(x, noise_opt['std'], noise_opt['length_scale'])
            self.dataset['noise'] = noise.reshape(-1, 1)
            self.dataset['u_dat_train'] = self.dataset['u_dat_train'] + self.dataset['noise']
        
        self.dataset.to_device(device)
    
    def func_mse(self, net):
        '''mean square error of variable parameter'''
        x = self.dataset['x_ic']
        y = net.pde_params_dict['u0'](x)
        return torch.mean(torch.square(y - self.dataset['u_ic']))
    
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset['upred_res'] = net(self.dataset['X_res'], net.pde_params_dict)
        self.dataset['upred_dat'] = net(self.dataset['X_dat'], net.pde_params_dict)
        self.dataset['upred_ic'] = net.pde_params_dict['u0'](self.dataset['x_ic'])
        
        self.prediction_variation(net)

    @torch.no_grad()
    def validate(self, net):
        '''compute l2 error and linf error of inferred D(x)'''
        
        x  = self.dataset['x_ic']
        u0_exact = self.dataset['u_ic']
        u0_pred = net.pde_params_dict['u0'](x)
        err = u0_exact - u0_pred
        l2norm = torch.mean(torch.square(err))
        linfnorm = torch.max(torch.abs(err)) 

        # compute h1 norm
        u0_pred_x = torch.zeros_like(u0_pred)
        u0_pred_x[:-1] = torch.abs(u0_pred[1:] - u0_pred[:-1]) / (x[1:] - x[:-1])
        
        u0_exact_x = torch.zeros_like(u0_exact)
        u0_exact_x[:-1] = torch.abs(u0_exact[1:] - u0_exact[:-1]) / (x[1:] - x[:-1])
        
        h1 = torch.mean(torch.square(u0_pred_x - u0_exact_x))
        h1 = h1 + l2norm
        
        return {'l2err': l2norm.item(), 'linferr': linfnorm.item(), 'h1err': h1.item()}

    @error_logging_decorator
    def plot_upred_dat(self, savedir=None):
        fig, ax = plt.subplots()
        x = self.dataset['X_dat'][:, 1]
        x_train = self.dataset['X_dat_train'][:, 1]
        ax.plot(x, self.dataset['u_dat'], label='exact')
        ax.plot(x, self.dataset['upred_dat'], label='NN')
        ax.scatter(x_train, self.dataset['u_dat_train'], label='data')
        ax.legend(loc="best")
        ax.grid()
        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xdat.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    @error_logging_decorator
    def plot_upred_res_meshgrid(self, savedir=None):
        # plot u at X_res, 
        
        u = self.dataset['u_res']
        u_pred = self.dataset['upred_res']
        # length of u_pred
        n = u_pred.shape[0]

        nx = int(np.sqrt(n))
        ny = nx
        
        # reshape to 2D
        u = u.reshape(nx, ny)
        u_pred = u_pred.reshape(nx, ny)
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
            path = os.path.join(savedir, 'fig_upred_grid.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    @error_logging_decorator
    def plot_upred_res(self, savedir=None):
        # plot u at X_res,         
        u = self.dataset['u_res']
        u_pred = self.dataset['upred_res']
        err = u - u_pred
        
        # uniformly spaced interger between 0 and Nt
        N = 5
        tidx = np.linspace(0, 1001-1, N, dtype=int)

        # reshape to 2D
        u = u.reshape(1001,1001)
        u_pred = u_pred.reshape(1001,1001)
        fig, ax = plt.subplots()
        
        # get colororder
        C = plt.rcParams['axes.prop_cycle'].by_key()['color']
        k = 0
        for i in tidx:
            # plot u at each t
            ax.plot(u[i,:], label=f'Exact t={i/1000:.2f}', color=C[k])
            ax.plot(u_pred[i,:], label=f'NN t={i/1000:.2f}', linestyle='--', color=C[k])
            k += 1
        ax.legend(loc="best")
        ax.grid()

        if savedir is not None:
            path = os.path.join(savedir, 'fig_upred_xres.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    

    @error_logging_decorator
    def plot_ic_pred(self, savedir=None):
        ''' plot predicted d and exact d'''
        fig, ax = plt.subplots()
        ax.plot(self.dataset['x_ic'], self.dataset['upred_ic'], label='NN')
        ax.plot(self.dataset['x_ic'], self.dataset['u_ic'], label='Exact')
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_ic_pred.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')
    
    @error_logging_decorator
    def plot_sample(self, savedir=None):
        '''plot distribution of collocation points'''
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['X_res_train'][:, 1], self.dataset['X_res_train'][:, 0], s=2.0, marker='.', label='X_res_train')
        ax.scatter(self.dataset['X_dat_train'][:, 1], self.dataset['X_dat_train'][:, 0], s=2.0, marker='s', label='X_dat_train')
        zeros = np.zeros_like(self.dataset['x_ic_train'])
        ax.scatter(self.dataset['x_ic_train'], zeros , s=2.0, marker='x', label='X_ic_train')
        # same xlim ylim
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        # legend
        ax.legend(loc="best")
        if savedir is not None:
            path = os.path.join(savedir, 'fig_Xdist.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'fig saved to {path}')

    @torch.no_grad()
    def prediction_variation(self, net):
        # make prediction with different parameters
        X = self.dataset['X_dat']
        
        # first variation, D+0.1
        funs = {}
        u0 = lambda x: torch.sin(torch.pi * x)
        funs['plus']= lambda x: u0(x) + 0.1 * u0(x)
        funs['minus']= lambda x: u0(x) - 0.1 * u0(x)
        funs['left']= lambda x: torch.sin(torch.pi * x + x *(1-x))

        for funkey, fun in funs.items():
            # replace parameter
            
            z = fun(X[:, 1:2])
            u = net(X, {'u0':z})
                
            key = f'uvar_{funkey}_dat'
            var = f'icvar_{funkey}_dat'
            self.dataset[key] = u
            self.dataset[var] = z
    
    @error_logging_decorator
    def plot_variation(self, savedir=None):
        # go through uvar and var
        def get_funkey(key):
            return key.split('_')[1]
            
        for ukey in self.dataset.keys():
            if ukey.startswith('uvar'):
                fig, ax = plt.subplots(2,1)

                funkey = get_funkey(ukey)
                ickey = ukey.replace('uvar', 'icvar')
                # plot u
                ax[0].plot(self.dataset['X_dat'][:,1], self.dataset['u_dat'], label='u')
                ax[0].plot(self.dataset['X_dat'][:,1], self.dataset[ukey], label=f'u-{funkey}')
                ax[0].legend(loc="best")
                # plot var, icvar is evaluated at X_dat,
                # but u_ic is provided at x_ic
                ax[1].plot(self.dataset['x_ic'], self.dataset['u_ic'], label='ic')
                ax[1].plot(self.dataset['X_dat'][:,1], self.dataset[ickey], label=f'ic-{funkey}')
                ax[1].legend(loc="best")

                if savedir is not None:
                    path = os.path.join(savedir, f'fig_var_{funkey}.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    print(f'fig saved to {path}')
    
    def visualize(self, savedir=None):
        '''visualize the problem'''
        self.plot_upred_res_meshgrid(savedir=savedir)
        self.plot_upred_res(savedir=savedir)
        self.plot_upred_dat(savedir=savedir)
        self.plot_ic_pred(savedir=savedir)
        self.plot_sample(savedir=savedir)
        self.plot_variation(savedir=savedir)
                            
        
        


if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'heat'
    optobj.opts['pde_opts']['trainable_param'] = 'u0'


    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = BurgerProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])


