#!/usr/bin/env python
# define problems for PDE
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from BaseProblem import BaseProblem
from MatDataset import MatDataset
from util import generate_grf, add_noise


''' 
Based on Supporting Information for:
Discovering governing equations from data: Sparse identification of nonlinear dynamical systems
Steven L. Brunton1, Joshua L. Proctor2, J. Nathan Kutz
damped oscillator nonlinear
dxdt = a11  x^3 + a12  y^3
dydt = -2   x^3 + -0.1   y^3
a11 = -0.1;
a12 = 2;
t0 = 0;
tstop = 25
y0 = [2 0];
'''

class SimpleODEProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 2
        self.opts = kwargs

        self.dataset = MatDataset(kwargs['datafile'])
        # get parameter from mat file
        self.param = {}
        Aname = f'A{kwargs["testcase"]}'
        self.param['a11'] = self.dataset[Aname][0,0]
        self.param['a12'] = self.dataset[Aname][0,1]
        self.param['a21'] = self.dataset[Aname][1,0]
        self.param['a22'] = self.dataset[Aname][1,1]
        self.testcase = kwargs['testcase']
        self.p = self.dataset['p']
        self.y0 = self.dataset['y0']
        
        y0 = torch.tensor(self.y0)
        
        # this allow u0 follow the device of Neuralnet
        self.lambda_transform = torch.nn.Module()
        self.lambda_transform.register_buffer('u0', y0)
        self.lambda_transform.forward = lambda x, u: self.lambda_transform.u0 + u*x


    def residual(self, nn, x):
        x.requires_grad_(True)
        
        u_pred = nn(x, nn.pde_params_dict)  # Assuming x.shape is (batch, 1)
        # Initialize tensors
        u_t = torch.zeros_like(u_pred)
        res = torch.zeros_like(u_pred)

        # Compute gradients for each output dimension and adjust dimensions
        for i in range(u_pred.shape[1]):
            grad_outputs = torch.ones_like(u_pred[:, i])
            u_t_i = torch.autograd.grad(u_pred[:, i], x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            u_t[:, i] = u_t_i[:, 0]  # Adjust dimensions
        
        # Perform your operations
        res[:, 0:1] = u_t[:, 0:1] - (nn.params_expand['a11'] * torch.pow(u_pred[:, 0:1],self.p) + nn.params_expand['a12'] * torch.pow(u_pred[:, 1:2],self.p))
        res[:, 1:2] = u_t[:, 1:2] - (nn.params_expand['a21'] * torch.pow(u_pred[:, 0:1],self.p) + nn.params_expand['a22'] * torch.pow(u_pred[:, 1:2],self.p))
        
        return res, u_pred

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')
        print(f'p = {self.p}')
        print(f'y0 = {self.y0}')



    def solve_ode(self, param, tend = 1.0, num_points=1001, t_eval=None):
        """
        Solves the ODE using Scipy's solve_ivp with high accuracy.
        
        Args:
        tend (double): end time
        num_points (int): Number of time points to include in the solution.

        Returns:
        sol: A `OdeResult` object representing the solution.
        """
        # Define the ODE system
        def ode_system(t, y):
            x, y = y
            dxdt = param['a11'] * x**self.p + param['a12'] * y**self.p
            dydt = param['a21'] * x**self.p + param['a22'] * y**self.p
            return [dxdt, dydt]


        # Time points where the solution is computed
        if t_eval is None:
            t_eval = np.linspace(0.0, tend, num_points)
        t_span = (0.0, tend)

        # Initial conditions, self.yo is 2-1 tensor, need to convert to 1-dim numpy array
        y0 = self.y0.numpy().reshape(-1)

        # Solve the ODE
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='DOP853', rtol=1e-9, atol=1e-9)

        # sol.y is (2, n) array, n is number of time points
        return sol.y.T 

    def setup_dataset(self, dsopt, noise_opt):
        # data loss
        self.create_dataset_from_file(dsopt)

        if noise_opt['use_noise']:
            print('add noise to training data')
            add_noise(self.dataset, noise_opt)

    
    def create_dataset_from_file(self, dsopt):
        '''create dataset from file'''
        assert self.dataset is not None, 'datafile provide, dataset should not be None'
        Aname = f'A{self.testcase}'
        uname = f'u{self.testcase}'

    
        self.dataset['x_dat'] = self.dataset['ts']
        self.dataset['u_dat'] = self.dataset[uname]

        self.dataset['x_res'] = self.dataset['ts']
        self.dataset['u_res'] = self.dataset[uname]
        
        self.dataset.subsample_evenly_astrain(dsopt['N_res_train'], ['x_res', 'u_res'])
        self.dataset.subsample_evenly_astrain(dsopt['N_dat_train'], ['x_dat', 'u_dat'])
        
        # remove other test case
        for i in range(10):
            if i != self.testcase:
                self.dataset.pop(f'A{i}',None)
                self.dataset.pop(f'u{i}',None)
        
        self.dataset.printsummary()

    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        with torch.no_grad():
            self.dataset['upred_res'] = net(self.dataset['x_res'], net.pde_params_dict)
            self.dataset['upred_dat'] = net(self.dataset['x_dat'], net.pde_params_dict)
            params = {k: v.item() for k, v in net.pde_params_dict.items()}
            self.dataset['ufdm_dat'] = self.solve_ode(params)
        
        self.prediction_variation(net)

    def plot_pred_comp(self, savedir=None):
        ''' plot prediction at x_dat_train
        '''

        # scatter plot of training data, might be noisy
        x_dat_train = self.dataset['X_dat_train']
        u_dat_train = self.dataset['u_dat_train']

        # line plot gt solution and prediction
        u_test = self.dataset['u_dat']
        upred = self.dataset['upred_dat']
        x_dat = self.dataset['x_dat']

        # visualize the results
        fig, ax = plt.subplots()

        # Get the number of dimensions
        d = upred.shape[1]
        # get color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Plot each dimension
        for i in range(d):
            color = color_cycle[i % len(color_cycle)]
            coord = chr(120 + i)
            ax.plot(x_dat, upred[:, i], label=f'pred {coord}', color=color)
            ax.plot(x_dat, u_test[:, i], label=f'test {coord}', linestyle='--', color=color)
            ax.scatter(x_dat_train, u_dat_train[:, i], label=f'train {coord}',color=color,marker='.')
            if 'ufdm_dat' in self.dataset:
                ax.plot(x_dat, self.dataset['ufdm_dat'][:, i], label=f'inf {coord}', linestyle=':', color=color)

        ax.legend()

        if savedir is not None:
            fpath = os.path.join(savedir, 'fig_pred_comp.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def plot_pred_traj(self, savedir=None):
        ''' plot prediction at x_dat_train
        '''

        # scatter plot of training data, might be noisy
        x_dat_train = self.dataset['X_dat_train']
        u_dat_train = self.dataset['u_dat_train']

        # line plot gt solution and prediction
        u_test = self.dataset['u_dat']
        upred = self.dataset['upred_dat']
        x_dat = self.dataset['x_dat']

        # visualize the results
        fig, ax = plt.subplots()
        ax.plot(upred[:, 0], upred[:, 1], label=f'pred')
        ax.plot(u_test[:, 0], u_test[:, 1], label=f'gt')
        ax.scatter(u_dat_train[:, 0], u_dat_train[:, 1], label=f'data')
        if 'ufdm_dat' in self.dataset:
            ax.plot(self.dataset['ufdm_dat'][:, 0],self.dataset['ufdm_dat'][:, 1], label=f'fdm')

        ax.legend()

        if savedir is not None:
            fpath = os.path.join(savedir, 'fig_pred_traj.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_pred_comp(savedir=savedir)
        self.plot_pred_traj(savedir=savedir)
        self.plot_variation(savedir=savedir)




if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'simpleode'

    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = SimpleODEProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])


    # save dataset
    fpath = os.path.join(optobj.opts['logger_opts']['save_dir'], 'dataset.mat')
    prob.dataset.save(fpath)