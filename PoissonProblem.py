#!/usr/bin/env python
# define problems for PDE
import torch
import os
import numpy as np

from util import generate_grf, add_noise, error_logging_decorator

from BaseProblem import BaseProblem
from MatDataset import MatDataset
from PoissonTest import PoissonTest
import matplotlib.pyplot as plt

class PoissonProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = kwargs.get('dim', 1)
        print(f'Poisson problem in {self.input_dim}D')
        self.output_dim = 1
        self.opts=kwargs

        # dictionary of parameters, keys are D0, D1, D2
        self.pde_params = [f'D{i}' for i in range(self.input_dim)]
        self.init_param = kwargs['init_param']
        self.gt_param = kwargs['gt_param']

        self.poisson_test = PoissonTest(dim=self.input_dim, D = self.gt_param)

        # initial guess
        self.all_params_dict = self.gt_param
        

        if self.input_dim == 1:
            self.lambda_transform = lambda X, u, param: u * X * (1 - X)
        elif self.input_dim == 2:
            self.lambda_transform = lambda X, u, param: u * X[:,0:1] * (1 - X[:,0:1]) * X[:,1:2] * (1 - X[:,1:2])
        elif self.input_dim == 3:
            self.lambda_transform = lambda X, u, param: u * X[:,0:1] * (1 - X[:,0:1]) * X[:,1:2] * (1 - X[:,1:2]) * X[:,2:3] * (1 - X[:,2:3])
        else:
            raise NotImplementedError("Only 1D and 2D cases are implemented.")


    def residual(self, nn, Xcoord):
        Xcoord.requires_grad_(True)
        coords = [Xcoord[:, i:i+1] for i in range(self.input_dim)]

        X = torch.cat(coords, dim=1)
        
        u_pred = nn(X, nn.pde_params_dict)
        lap = 0
        for i in range(len(coords)):
            du = torch.autograd.grad(u_pred, coords[i],
                create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
            d2u = torch.autograd.grad(du, coords[i],
                create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(du))[0]
            lap += nn.params_expand[f'D{i}'] * d2u

        res = -lap - self.dataset['f_res_train']

        return res, u_pred

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.all_params_dict.items():
            print(f'{k} = {v}')

    
    def create_dataset_from_pde(self, dsopt):
        # create dataset from pde using datset option and noise option
        dataset = MatDataset()

        poisson_dataset = self.poisson_test.generate_grid_data(dsopt['N_res_train'])

        # copy all items
        for k,v in poisson_dataset.items():
            dataset[k] = v
        
        # Create meshgrid
        coord_list_1d = [coord.squeeze() for coord in dataset['coord_list']]
        grids = torch.meshgrid(*coord_list_1d, indexing='ij')
        # Flatten
        X_list = []
        for grid in grids:
            x_flat = grid.flatten().view(-1, 1)
            X_list.append(x_flat)
        
        # Compute ground truth on grid and flatten
        u_gt_flat = dataset['u_gt'].view(-1, 1)
        f_gt_flat = dataset['f_gt'].view(-1, 1)


        dataset['X_res_train'] = torch.cat(X_list, dim=1)
        dataset['f_res_train'] = f_gt_flat

        # data col-pt, for testing, use exact param
        dataset['X_dat_test'] = dataset['X_res_train']
        dataset['u_dat_test'] = u_gt_flat


        # data col-pt, for initialization use init_param, for training use exact_param
        dataset['X_dat_train'] = dataset['X_res_train']
        dataset['u_dat_train'] = u_gt_flat

        self.dataset = dataset


    def setup_dataset(self, dsopt, noise_opt, device='cuda'):
        '''add noise to dataset'''
        self.create_dataset_from_pde(dsopt)
        if noise_opt['use_noise']:
            add_noise(self.dataset, noise_opt)
        self.dataset.to_device(device)

    def u_exact(self, x, param:dict):
        # 1D only
        return torch.sin(torch.pi *  x) / param['D0']
    
    # override base class methods to handle visualization locally
    @torch.no_grad()
    def make_prediction(self, net):
        """Make predictions on test data"""
        xtest = self.dataset['X_dat_test']
        upred = net(xtest, net.pde_params_dict)
        self.dataset['upred_dat_test'] = upred.reshape(self.dataset['shape'])

        # if 1D,predict variation
        if self.input_dim == 1:
            deltas = [0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5]
            self.dataset['deltas'] = deltas
            
            for k in self.pde_params:
                # copy the parameters, DO NOT modify the original parameters
                # need to be in the loop, reset each time
                # tmp_param_dict = {k: v.clone() for k, v in net.pde_params_dict.items()}
                tmp_param_dict = {}
                for k_, v_ in net.pde_params_dict.items():
                    if isinstance(v_, torch.Tensor):
                        tmp_param_dict[k_] = v_.clone()
                    else:
                        raise ValueError('unknown type of pde parameter')
                    
                # original value
                param_value = tmp_param_dict[k]
                param_name = k

                for delta_i, delta in enumerate(deltas):
                    new_value = param_value * (1 + delta)
                    
                    tmp_param_dict[param_name] = new_value
                    # to(x_test.device)

                    u_test = net(xtest, tmp_param_dict)
                    vname = f'var_{param_name}_{delta_i}_pred'
                    self.dataset[vname] = u_test
                    
                    u_exact = self.u_exact(xtest, tmp_param_dict)
                    vname = f'var_{param_name}_{delta_i}_exact'
                    self.dataset[vname] = u_exact


    
    def plot_prediction(self, savedir=None, vname='u'):
        """Override base class method - handled in visualize"""
        pass
    
    @error_logging_decorator
    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter
        if 'deltas' not in self.dataset:
            print('no deltas found in dataset, skip plotting variation')
            return
        x_test = self.dataset['X_dat_test']

        deltas = self.dataset['deltas']

        var_names = self.dataset.filter('var_')
        # find unique parameter names
        param_names = list(set([v.split('_')[1] for v in var_names]))

        # for each varname, plot the solution and variation
        for varname in param_names:
            fig, ax = plt.subplots()

            # for each delta
            for i_delta,delta in enumerate(deltas):

                vname_pred = f'var_{varname}_{i_delta}_pred'
                # plot prediction
                u_pred = self.dataset[vname_pred]
                ax.plot(x_test, u_pred, label=f'NN $\Delta${varname} = {delta:.2f}')

                # plot exact if available
                if hasattr(self, 'u_exact'):
                    vname_exact = f'var_{varname}_{i_delta}_exact'
                    color = ax.lines[-1].get_color()
                    u_exact = self.dataset[vname_exact]
                    ax.plot(x_test, u_exact, label=f'exact $\Delta${varname} = {delta:.2f}',color=color,linestyle='--')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            if savedir is not None:
                fname = f'fig_var_{varname}.png'
                fpath = os.path.join(savedir, fname)
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')

        return

    @torch.no_grad()
    def visualize(self, savedir=None):
        # Convert dataset to numpy for plotting
        self.dataset.to_np()
        
        if self.input_dim == 1:

            self.plot_variation(savedir=savedir)


            # 1D visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x_grid = self.dataset['coord_list'][0]
            
            # Plot prediction
            ax.plot(x_grid, self.dataset['upred_dat_test'], 
                   label='$u_{pred}$', color='#FF69B4', linewidth=2)
            
            # Plot ground truth
            ax.plot(x_grid, self.dataset['u_gt'], 
                   label='$u_{exact}$', linestyle='--', color='black', linewidth=2)
            
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title('1D Poisson Problem Solution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            
        
        elif self.input_dim == 2:
            # 2D visualization
            # Get grid shape from dataset
            grid_shape = self.dataset['shape']
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot prediction
            u_pred = self.dataset['upred_dat_test']
            im0 = axs[0].imshow(u_pred, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
            axs[0].set_title('$u_{pred}$')
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
            fig.colorbar(im0, ax=axs[0])
            
            # Plot ground truth
            u_exact = self.dataset['u_gt']
            im1 = axs[1].imshow(u_exact, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
            axs[1].set_title('$u_{exact}$')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('y')
            fig.colorbar(im1, ax=axs[1])
            
            # Plot error
            error = np.abs(u_pred - u_exact)
            im2 = axs[2].imshow(error, extent=(0, 1, 0, 1), origin='lower', cmap='hot')
            axs[2].set_title('Absolute Error')
            axs[2].set_xlabel('x')
            axs[2].set_ylabel('y')
            fig.colorbar(im2, ax=axs[2])
            
            # Print error statistics
            print(f"Max error: {np.max(error):.6e}")
            print(f"Mean error: {np.mean(error):.6e}")
            print(f"L2 error: {np.linalg.norm(error):.6e}")
            
            plt.tight_layout()
            
        
        elif self.input_dim == 3:
            # 3D visualization - plot middle slice along z-axis
            # Get grid shape from dataset
            grid_shape = self.dataset['shape']
            
            # Get middle slice index
            mid_z = grid_shape[2] // 2
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot prediction - middle z-slice
            u_pred = self.dataset['upred_dat_test']
            u_pred_slice = u_pred[:, :, mid_z]
            im0 = axs[0].imshow(u_pred_slice, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
            axs[0].set_title(f'$u_{{pred}}$ (z={mid_z}/{grid_shape[2]-1})')
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
            fig.colorbar(im0, ax=axs[0])
            
            # Plot ground truth - middle z-slice
            u_exact = self.dataset['u_gt']
            u_exact_slice = u_exact[:, :, mid_z]
            im1 = axs[1].imshow(u_exact_slice, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
            axs[1].set_title(f'$u_{{exact}}$ (z={mid_z}/{grid_shape[2]-1})')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('y')
            fig.colorbar(im1, ax=axs[1])
            
            # Plot error - middle z-slice
            error = np.abs(self.dataset['upred_dat_test'] - self.dataset['u_gt'])
            error_slice = error[:, :, mid_z]
            im2 = axs[2].imshow(error_slice, extent=(0, 1, 0, 1), origin='lower', cmap='hot')
            axs[2].set_title(f'Absolute Error (z={mid_z}/{grid_shape[2]-1})')
            axs[2].set_xlabel('x')
            axs[2].set_ylabel('y')
            fig.colorbar(im2, ax=axs[2])
            
            # Print error statistics
            print(f"Max error: {np.max(error):.6e}")
            print(f"Mean error: {np.mean(error):.6e}")
            print(f"L2 error: {np.linalg.norm(error):.6e}")
            
            plt.tight_layout()
            
        
        else:
            raise NotImplementedError(f"Visualization for {self.input_dim}D is not implemented")

        if savedir is not None:
            fpath = os.path.join(savedir, f'fig_poisson_{self.input_dim}d.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'{self.input_dim}D figure saved to {fpath}')

        plt.show()


        
 
if __name__ == "__main__":
    import sys
    from Options import *
    from DenseNet import *
    from Problems import *


    optobj = Options()
    optobj.opts['pde_opts']['problem'] = 'poisson'

    optobj.parse_args(*sys.argv[1:])
    
    
    device = set_device('cuda')
    set_seed(0)
    
    print(optobj.opts)

    prob = PoissonProblem(**optobj.opts['pde_opts'])
    pdenet = prob.setup_network(**optobj.opts['nn_opts'])
    prob.setup_dataset(optobj.opts['dataset_opts'], optobj.opts['noise_opts'])

    prob.make_prediction(pdenet)
    prob.visualize(savedir=optobj.opts['logger_opts']['save_dir'])

    # save dataset
    fpath = os.path.join(optobj.opts['logger_opts']['save_dir'], 'dataset.mat')
    prob.dataset.save(fpath)


