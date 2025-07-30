#!/usr/bin/env python
# define problems for PDE
import os
import torch

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import griddata

from Options import *
from util import *
from DenseNet import DenseNet
from MatDataset import MatDataset
from BaseProblem import BaseProblem

def sumcol(A):
    # sum along column
    return torch.sum(A, dim=1, keepdim=True)


def sigmoid_binarize(x, th):
    # smooth heaviside function using a sigmoid
    K = 20
    return torch.nn.functional.sigmoid(K*(x-th))

def phimse(x,y,phi):
    return torch.mean(((x-y)*phi)**2)

def segmseloss(upred, udat, phi, th):    
    '''spatial segmentation loss by mse'''
    uth = sigmoid_binarize(upred,th)
    return phimse(uth, udat, phi)

def dice(seg_pred, seg_gt):
    # dice
    tp = torch.sum(seg_pred * seg_gt)
    fp = torch.sum(seg_pred * (1-seg_gt))
    fn = torch.sum((1-seg_pred) * seg_gt)
    dice = 2*tp/(2*tp+fp+fn)
    return dice

def r2(X):
    # return r^2 in non-dimensional scale
    r2 = sumcol(torch.square((X[:, 1:])))
    return r2

def range_penalty(x, xmin, xmax):
    # relu loss
    return torch.nn.functional.relu(x - xmax)**2 + torch.nn.functional.relu(xmin - x)**2

class GBMproblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.dataset = MatDataset(kwargs['datafile'])
        
        self.opts = kwargs
        ### GBM custom loss
        self.loss_dict['seg1'] = self.get_seg1_loss
        self.loss_dict['seg2'] = self.get_seg2_loss

        self.loss_dict['dice1'] = self.get_dice1_metric
        self.loss_dict['dice2'] = self.get_dice2_metric

        self.loss_dict['uchar_res'] = self.get_uchar_res_loss
        self.loss_dict['uchar_dat'] = self.get_uchar_dat_loss
        self.loss_dict['ugt_res'] = self.get_ugt_res_loss
        self.loss_dict['ugt_dat'] = self.get_ugt_dat_loss

        self.loss_dict['thrange'] = self.range_constraint

        ### GBM specific options
        self.whichdata = kwargs['whichdata']
        self.usewdat = kwargs['usewdat']
        # put X_dat to X_res for residual loss
        self.combinedat = kwargs['combinedat']
        # force the network to be a positive
        self.pos_trans = kwargs['pos_trans']
        # force the network not change at boundary
        self.force_bc = kwargs['force_bc']

        # range of th1 and th2
        self.th1_range = kwargs['th1_range']
        self.th2_range = kwargs['th2_range']

        ### dimension
        self.xdim = int(self.dataset['xdim'])
        self.dim = self.xdim + 1 # add time dimension
        self.input_dim = self.dim
        self.output_dim = 1


        # get parameter from mat file
        # check empty string
        # inititalize parameters 
        self.all_params_dict['rD'] = 1.0
        self.all_params_dict['rRHO'] = 1.0
        self.all_params_dict['th1'] = self.opts['init_param']['th1'] if 'th1' in self.opts['init_param'] else 0.35
        self.all_params_dict['th2'] = self.opts['init_param']['th2'] if 'th2' in self.opts['init_param'] else 0.6


        self.pde_params = ['rD', 'rRHO']

        # ground truth parameters
        self.param_gt = {}
        if 'gt' in self.whichdata:
            self.param_gt['rD'] = self.dataset['rDe']
            self.param_gt['rRHO'] = self.dataset['rRHOe']
        else:
            self.param_gt['rD'] = 1.0
            self.param_gt['rRHO'] = 1.0

        
        self.DW = self.dataset['DW']
        self.RHO = self.dataset['RHO']
        self.L = self.dataset['L']

        # if rmax is defined, use it
        self.r2max = r2(self.dataset['X_bc']).max().item()
        
        # DO NOT USE self.x0
        # assuming x0 = [0,0,0]
        # self.x0 = self.dataset['x0']

        # concat DxPphi and DyPphi to form gradPphi
        self.dataset['gradPphi_res'] = torch.cat([self.dataset['DxPphi_res'], self.dataset['DyPphi_res']], dim=1)
        if self.xdim == 3:
            self.dataset['gradPphi_res'] = torch.cat([self.dataset['gradPphi_res'], self.dataset['DzPphi_res']], dim=1)
        
        self.dataset['gradPphi_dat'] = torch.cat([self.dataset['DxPphi_dat'], self.dataset['DyPphi_dat']], dim=1)
        if self.xdim == 3:
            self.dataset['gradPphi_dat'] = torch.cat([self.dataset['gradPphi_dat'], self.dataset['DzPphi_dat']], dim=1)


        # for transformation
        self.lambda_transform = torch.nn.Module()
        self.lambda_transform.register_buffer('L', torch.tensor(self.L))
        if self.pos_trans == True:
            # force the network to be positive
            self.lambda_transform.forward = lambda X, u: self.ic(X, self.lambda_transform.L) + torch.nn.functional.sigmoid(u) * X[:,0:1]
        elif self.force_bc == True:
            # force bc same as ic at rmax
            print(f'force bc at r2max = {self.r2max}')
            self.lambda_transform.register_buffer('rmax', torch.tensor(self.r2max))
            self.lambda_transform.forward = lambda X, u, param: self.ic(X, self.lambda_transform.L) + u * X[:,0:1] * (self.lambda_transform.rmax - r2(X))/self.lambda_transform.rmax
        else:
            self.lambda_transform.forward = lambda X, u, param: self.ic(X, self.lambda_transform.L) + u * X[:,0:1]


    # for testing purpose, still need to set whichdata
    def get_uchar_res_loss(self, net):
        # mse of uchar_res
        data = self.dataset.batch['res']
        X = data['X_res_train']
        u = data['uchar_res_train']
        phi = data['phi_res_train']
        upred = net(X, net.pde_params_dict)
        return phimse(upred, u, phi)
    
    def get_uchar_dat_loss(self, net):
        # mse of uchar_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['uchar_dat_train']
        phi = data['phi_dat_train']
        upred = net(X, net.pde_params_dict)
        return phimse(upred, u, phi)
    
    def get_ugt_res_loss(self, net):
        # mse of ugt_res
        data = self.dataset.batch['res']
        X = data['X_res_train']
        u = data['ugt_res_train']
        phi = data['phi_res_train']
        upred = net(X, net.pde_params_dict)
        return phimse(upred, u, phi)
    
    def get_ugt_dat_loss(self, net):
        # mse of ugt_dat
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        u = data['ugt_dat_train']
        phi = data['phi_dat_train']
        upred = net(X, net.pde_params_dict)
        return phimse(upred, u, phi)

    # compute validation statistics
    @torch.no_grad()
    def validate(self, nn):
        '''compute err '''
        v_dict = {}
        for vname in nn.trainable_param:
            v_dict[vname] = nn.all_params_dict[vname]
            if vname in self.param_gt and vname in nn.pde_params_dict:
                err = torch.abs(nn.pde_params_dict[vname] - self.param_gt[vname])
                v_dict[f'abserr_{vname}'] = err
        return v_dict

    def ic(self, X, L):
        # initial condition
        r2 = sumcol(torch.square((X[:, 1:self.dim])*L)) # this is in pixel scale, unit mm, 
        return 0.1*torch.exp(-0.1*r2)

    def residual(self, nn, X, phi, P, gradPphi):
        
        # Get the number of dimensions
        n = X.shape[0]

        # split each column of X into a separate tensor of size (n, 1)
        vars = [X[:, d:d+1] for d in range(self.dim)]
        
        
        # Concatenate sliced tensors to form the input for the network if necessary
        nn_input = torch.cat(vars, dim=1)
       
        # Forward pass through the network
        u = nn(nn_input, nn.pde_params_dict)
        # Define a tensor of ones for grad_outputs
        v = torch.ones_like(u)
        
        # Compute gradients with respect to the sliced tensors
        u_t = torch.autograd.grad(u, vars[0], grad_outputs=v, create_graph=True)[0]

        # n by d matrix
        u_x = torch.zeros(n, self.xdim, device=X.device)
        u_xx = torch.zeros(n, self.xdim, device=X.device)

        for d in range(0, self.xdim):
            u_x[:,d:d+1] = torch.autograd.grad(u, vars[d+1], grad_outputs=v, create_graph=True)[0]
            u_xx[:,d:d+1] = torch.autograd.grad(u_x[:,d:d+1], vars[d+1], grad_outputs=v, create_graph=True)[0]
        
        prof = nn.params_expand['rRHO'] * self.RHO * phi * u * ( 1 - u)
        diff = nn.params_expand['rD'] * self.DW * (P * phi * sumcol(u_xx) + self.L * sumcol(gradPphi * u_x))
        res = phi * u_t - (prof + diff)
        return res, u
    
    
    def get_res_pred(self, net):
        # get residual and prediction
        data = self.dataset.batch['res']
        X = data['X_res_train']
        X.requires_grad_(True)
        phi = data['phi_res_train']
        P = data['P_res_train']
        gradPphi = data['gradPphi_res_train']
        res, u_pred = self.residual(net, X, phi, P, gradPphi)
        self.res = res
        self.upred_res = u_pred
        return res, u_pred

    
    # def get_res_data(self, net):
    #     # get residual at final time
    #     self.dataset['X_dat_train'].requires_grad_(True)
        
    #     res, u_pred = self.residual(net, self.dataset['X_dat_train'], self.dataset['phi_dat_train'], self.dataset['P_dat_train'], self.dataset['gradPphi_dat_train'])
    #     return torch.mean(torch.square(res))
        

    def get_data_loss(self, net):
        # get data loss
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u = data['uchar_dat_train']
        upred = net(X, net.pde_params_dict)

        # pointwise loss
        loss = phimse(upred, u, phi)
        return loss
    
    def get_seg1_loss(self, net):
        # get segmentation loss of u1
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u1 = data['u1_dat_train']
        upred = net(X, net.pde_params_dict)
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th1'])
        loss = phimse(pred_seg, u1, phi)/torch.mean(u1**2)
        return loss
    
    def get_seg2_loss(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u2 = data['u2_dat_train']
        upred = net(X, net.pde_params_dict)
        pred_seg = sigmoid_binarize(upred, net.all_params_dict['th2'])
        loss = phimse(pred_seg, u2, phi)/torch.mean(u2**2)
        return loss

    def get_dice1_metric(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u1 = data['u1_dat_train']
        upred = net(X, net.pde_params_dict)
        pred_seg = upred > net.all_params_dict['th1']
        dice1 = dice(pred_seg, u1)
        return dice1
    
    def get_dice2_metric(self, net):
        data = self.dataset.batch['dat']
        X = data['X_dat_train']
        phi = data['phi_dat_train']
        u2 = data['u2_dat_train']
        upred = net(X, net.pde_params_dict)
        pred_seg = upred > net.all_params_dict['th2']
        dice2 = dice(pred_seg, u2)
        return dice2
        
    
    def bcloss(self, net):
        # get dirichlet boundary condition loss
        u_pred = net(self.dataset['X_bc_train'], net.pde_params_dict)
        phi = self.dataset['phi_bc_train']

        loss = torch.mean(torch.square((u_pred * phi - self.dataset['zero_bc_train'])))
        return loss

    def range_constraint(self, net):
        # range constraint for th1 and th2

        th1 = net.all_params_dict['th1'].squeeze()
        th2 = net.all_params_dict['th2'].squeeze()

        loss = range_penalty(th1, self.th1_range[0], self.th1_range[1]) + range_penalty(th2, self.th2_range[0], self.th2_range[1])
        return loss
    
    def print_info(self):
        pass
    
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset.to_device(self.dataset.device)
        net.to(self.dataset.device)

        x_dat = self.dataset['X_dat']
        x_res = self.dataset['X_res']
        
        x_dat_train = self.dataset['X_dat_train']
        x_res_train = self.dataset['X_res_train']

        
        with torch.no_grad():
            # write prediction by batch
            self.dataset['upred_dat'] = net(x_dat, net.pde_params_dict)
            self.dataset['upred_dat_train'] = net(x_dat_train, net.pde_params_dict)

            # Prediction by batching for large tensors x_res and x_res_train
            def batch_predict(x, batch_size):
                predictions = []
                for i in range(0, x.shape[0], batch_size):
                    batch = x[i:i + batch_size]
                    pred = net(batch, net.pde_params_dict)
                    predictions.append(pred)
                return torch.cat(predictions, dim=0)

            # Perform batched predictions to avoid OOM errors
            self.dataset['upred_res'] = batch_predict(x_res, 20000)
            self.dataset['upred_res_train'] = batch_predict(x_res_train, 20000)

        
            # prediction of parameters
            self.dataset['rD_pred'] = net.pde_params_dict['rD']
            self.dataset['rRHO_pred'] = net.pde_params_dict['rRHO']
            self.dataset['th1_pred'] = net.all_params_dict['th1']
            self.dataset['th2_pred'] = net.all_params_dict['th2']

        self.prediction_variation(net)

    @error_logging_decorator
    def plot_scatter_2d(self, X, u, fname = 'fig_2dscatter.png', savedir=None):

        x = X[:,1:2]
        y = X[:,2:3]
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        scatter = ax.scatter(x, y, c=u, cmap='viridis', s = 12)
        # xlabel
        ax.set_xlabel('x')
        # ylabel
        ax.set_ylabel('y')
        # colorbar
        cbar = fig.colorbar(scatter, ax=ax)

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    @error_logging_decorator
    def plot_2d_contour(self, X, u, v, fname = 'fig_2dcontour.png', savedir=None, title = '', levels = None):
        # compare contour plot of u and v
        # u is dashed, v is solid
        x = X[:,1:2]
        y = X[:,2:3]
        # reshape to 1D
        x = x.reshape(-1)
        y = y.reshape(-1)
        u = u.reshape(-1)
        v = v.reshape(-1)
        
        if levels is None:
            levels = np.array([0.01, 0.1, 0.3, 0.6])

        fig, ax = plt.subplots()

        ax.tricontour(x, y, u, levels=levels, cmap='viridis', linestyles='dashed')
        ax.tricontour(x, y, v, levels=levels, cmap='viridis', linestyles='solid')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if title != '':
            ax.set_title(title)

        # set axis equal
        ax.axis('equal')
        

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')
        
        return fig, ax
    
    @error_logging_decorator
    def plot_3d_contour(self, X, u, v, plane='z', k=0, fname_prefix='fig_3d', savedir=None, title='', levels=None):
        # Interpolates data to a 2D grid for x and y, then plots contour
        # X has columns [t, x, y, z] for unstructured data

        n = X.shape[0]
        maxn = 100000
        if n > maxn:
            # plot first maxn points
            X = X[:maxn]
            u = u[:maxn]
            v = v[:maxn]

        x = X[:, 1]
        y = X[:, 2]
        z = X[:, 3]

        # Create grid for interpolation based on the specified plane
        if plane == 'x':
            grid_x, grid_y, grid_z = np.meshgrid(
                [k],
                np.linspace(min(y), max(y), 100),
                np.linspace(min(z), max(z), 100)
            )
            grid_x_slice = grid_y[:, 0, :]
            grid_y_slice = grid_z[:, 0, :]
            
            xlabel = 'y'
            ylabel = 'z'
        elif plane == 'y':
            grid_x, grid_y, grid_z = np.meshgrid(
                np.linspace(min(x), max(x), 100),
                [k],
                np.linspace(min(z), max(z), 100)
            )
            grid_x_slice = grid_x[0, :, :]
            grid_y_slice = grid_z[0, :, :]
            xlabel = 'x'
            ylabel = 'z'
        elif plane == 'z':
            grid_x, grid_y, grid_z = np.meshgrid(
                np.linspace(min(x), max(x), 100),
                np.linspace(min(y), max(y), 100),
                [k]
            )
            grid_x_slice = grid_x[:, :, 0]
            grid_y_slice = grid_y[:, :, 0]
            xlabel = 'x'
            ylabel = 'y'
        else:
            raise ValueError("Plane must be 'x', 'y', or 'z'")

        # Reshape grid to 2D
        grid_x_flat = grid_x.reshape(-1)
        grid_y_flat = grid_y.reshape(-1)
        grid_z_flat = grid_z.reshape(-1)

        # Stack the flattened grids to create an n-by-3 array
        grid_points = np.stack((grid_x_flat, grid_y_flat, grid_z_flat), axis=-1)

        # Interpolate using griddata
        u_grid = griddata((x, y, z), u, grid_points, method='linear').reshape(grid_x.shape)
        v_grid = griddata((x, y, z), v, grid_points, method='linear').reshape(grid_x.shape)

        if levels is None:
            levels = np.array([0.01, 0.1, 0.3, 0.6])

        fig, ax = plt.subplots()

        # Plot contour for u and v
        if plane == 'z':
            CS_u = ax.contour(grid_x_slice, grid_y_slice, u_grid[:, :, 0], levels=levels, cmap='viridis', linestyles='solid')
            CS_v = ax.contour(grid_x_slice, grid_y_slice, v_grid[:, :, 0], levels=levels, cmap='viridis', linestyles='dashed')
        elif plane == 'y':
            CS_u = ax.contour(grid_x_slice, grid_y_slice, u_grid[0, :, :], levels=levels, cmap='viridis', linestyles='solid')
            CS_v = ax.contour(grid_x_slice, grid_y_slice, v_grid[0, :, :], levels=levels, cmap='viridis', linestyles='dashed')
        elif plane == 'x':
            CS_u = ax.contour(grid_x_slice, grid_y_slice, u_grid[:, 0, :], levels=levels, cmap='viridis', linestyles='solid')
            CS_v = ax.contour(grid_x_slice, grid_y_slice, v_grid[:, 0, :], levels=levels, cmap='viridis', linestyles='dashed')
        else:
            raise ValueError("Plane must be 'x', 'y', or 'z'")



        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


        if title != '':
            ax.set_title(title)

        # Set axis equal
        ax.axis('equal')

        if savedir is not None:
            # add label to fname
            fname = f'{fname_prefix}_{plane}_{k}.png'
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    @error_logging_decorator
    def plot_scatter(self, X, u, fname = 'fig_scatter.png', savedir=None, title = ''):
        
        maxn = 100000
        if X.shape[0] > maxn:
            # plot first maxn points
            X = X[:maxn]
            u = u[:maxn]

        x = X[:,1:]
        t = X[:,0]
        
        # if dim >2, only plot radial
        if x.shape[1] > 1:
            r = np.linalg.norm(x,axis=1)
        else:
            r = x
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.scatter(r, u, c=t, cmap='viridis')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(True)
        ax.set_title(title)

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    @error_logging_decorator
    def plot_1d_profile(self, X, upred, ubase,  fname = 'fig_1d_profile.png', savedir=None, title = ''):
        # for variation of 1d problem

        x = X[:,1:]
        # reshape to 1D
        x = x.reshape(-1)
        upred = upred.reshape(-1)
        ubase = ubase.reshape(-1)

        # sort by x
        idx = np.argsort(x, axis=0)
        x = x[idx]
        upred = upred[idx]
        ubase = ubase[idx]
        
        # visualize the results
        fig, ax = plt.subplots()
        
        # scatter plot, color is upred
        ax.plot(x, ubase, label='base', linestyle='solid')
        ax.plot(x, upred, label='pred', linestyle='dashed')

        ax.set_title(title)

        if savedir is not None:
            fpath = os.path.join(savedir, fname)
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax
    
    @error_logging_decorator
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        phiupred_dat = self.dataset['upred_dat']  * self.dataset['phi_dat']


        # 3 types of data: patient, char, gt

        if 'gt' in self.whichdata:
            if 'ugt_dat' in self.dataset:
                phiu_dat = self.dataset['ugt_dat']  * self.dataset['phi_dat']
                u_dat_train = self.dataset['ugt_dat_train'] * self.dataset['phi_dat_train']
        elif 'char' in self.whichdata:
            phiu_dat = self.dataset['uchar_dat']  * self.dataset['phi_dat']
            u_dat_train = self.dataset['uchar_dat_train'] * self.dataset['phi_dat_train']
        else:
            # plot segmentation
            u_dat_train = self.dataset['u1_dat_train'] + self.dataset['u2_dat_train']


        # prediction can always be visualized
        fig, ax = self.plot_scatter(self.dataset['X_dat'], phiupred_dat, fname = 'fig_phiupred_dat.png', savedir=savedir)
        fig, ax = self.plot_scatter(self.dataset['X_dat_train'], self.dataset['upred_dat_train'] * self.dataset['phi_dat_train'], fname = 'fig_phiupred_dat_train.png', savedir=savedir)
        fig, ax = self.plot_scatter(self.dataset['X_res_train'], self.dataset['upred_res_train'] * self.dataset['phi_res_train'], fname = 'fig_phiupred_res_train.png', savedir=savedir)
        plt.close('all')

        fig, ax = self.plot_scatter(self.dataset['X_dat_train'], u_dat_train , fname = 'fig_phiudat_train.png', savedir=savedir)
        
        # if 1d, plot profile
        if self.xdim == 1:
            self.plot_variation(savedir=savedir)
            fig, ax = self.plot_1d_profile(self.dataset['X_dat'], phiupred_dat, phiu_dat, fname = 'fig_1d_phiupred_dat.png', savedir=savedir)
            plt.close('all')

        # if 2d, plot scatter plot
        if self.xdim == 2:

            fig, ax = self.plot_scatter_2d(self.dataset['X_dat'], phiupred_dat, fname = 'fig_2d_phiupred_dat.png', savedir=savedir)

            # if gt or char, plot the difference
            if 'gt' in self.whichdata or 'char' in self.whichdata:
                err = np.abs(phiupred_dat - phiu_dat)
                fig, ax = self.plot_scatter_2d(self.dataset['X_dat'], err, fname = 'fig_2d_err_dat.png', savedir=savedir)
                fig, ax = self.plot_2d_contour(self.dataset['X_dat'], phiupred_dat, phiu_dat, fname = 'fig_contour_phiupred_dat.png', savedir=savedir, title="pred --, data -")
                plt.close('all')
                self.plot_variation(savedir=savedir)
            else:
                # plot patient prediction not implemented yet
                pass
        
        if self.xdim == 3:
            if 'gt' in self.whichdata or 'char' in self.whichdata:
                # plot 3d contour
                fig, ax = self.plot_3d_contour(self.dataset['X_dat'], phiupred_dat, phiu_dat,  plane='x', k=0, fname_prefix='fig_3d_phiupred_dat', savedir=savedir)
                fig, ax = self.plot_3d_contour(self.dataset['X_dat'], phiupred_dat, phiu_dat,  plane='y', k=0, fname_prefix='fig_3d_phiupred_dat', savedir=savedir)
                fig, ax = self.plot_3d_contour(self.dataset['X_dat'], phiupred_dat, phiu_dat,  plane='z', k=0, fname_prefix='fig_3d_phiupred_dat', savedir=savedir)
                plt.close('all')


        
        

    
    @error_logging_decorator
    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter compare with 0 variation
        # only 2d
        x_dat = self.dataset['X_dat']

        deltas = self.dataset['deltas']

        var_names = self.dataset.filter('var_')
        # find unique parameter names
        param_names = list(set([v.split('_')[1] for v in var_names]))


        # for each varname, plot the solution and variation
        for varname in param_names:
            # for each delta
            for i_delta,delta in enumerate(deltas):

                vname_pred = f'var_{varname}_{i_delta}_pred'
                # plot prediction
                u_pred = self.dataset[vname_pred] * self.dataset['phi_dat']
                u_pred_base = self.dataset[f'var_{varname}_0_pred'] * self.dataset['phi_dat']

                title_str = f'NN $\Delta${varname} = {delta:.2f} --'
                # if 2d
                if self.xdim == 2:
                    fig, ax = self.plot_2d_contour(x_dat, u_pred, u_pred_base, fname = f'{vname_pred}.png', savedir=savedir, title=title_str)
                
                if self.xdim == 1:
                    fig, ax = self.plot_1d_profile(x_dat, u_pred, u_pred_base, fname = f'{vname_pred}.png', savedir=savedir, title=title_str)
                
        
        plt.close('all')

        return

    
    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        ''' downsample for training'''
        
        # data loss, process variables ends with _dat
        ndat_train = min(ds_opts['N_dat_train'], self.dataset['X_dat'].shape[0])
        vars = self.dataset.filter('_dat')
        dat_vars = self.dataset.subsample_unif_astrain(ndat_train, vars)
        print('downsample ', vars, ' to ', ndat_train)
        ds_opts['N_dat_train'] = ndat_train
        

        # res loss
        nres_train = min(ds_opts['N_res_train'], self.dataset['X_res'].shape[0])
        vars = self.dataset.filter('_res')
        res_vars = self.dataset.subsample_unif_astrain(nres_train, vars)
        print('downsample ', vars, ' to ', nres_train)
        ds_opts['N_res_train'] = nres_train
        

        # bc loss
        n = min(ds_opts['N_bc_train'], self.dataset['X_bc'].shape[0])
        vars = self.dataset.filter('_bc')
        self.dataset.subsample_unif_astrain(n, vars)
        print('downsample ', vars, ' to ', n)
        ds_opts['N_bc_train'] = n

        # put all variables to device
        self.dataset.to_device(device)

        res_batch_size = ds_opts['net_batch_size']  
        dat_batch_size = ds_opts['pde_batch_size']
        
        if ndat_train < dat_batch_size:
            dat_batch_size = ndat_train
        if nres_train < res_batch_size:
            res_batch_size = nres_train

        
        # if GPU tensor, pin_memeroy has to be false, and num_workers has to be 0
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        self.dataset.configure_dataloader('dat', dat_vars, batch_size=dat_batch_size, shuffle=True)
        self.dataset.configure_dataloader('res', res_vars, batch_size=res_batch_size, shuffle=True)
       

# load model and visualize
if __name__ == "__main__":

    from MlflowHelper import *
    

    # commandline read exp name and run name
    import argparse
    import sys
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some parameters.')
    
    # Add arguments
    parser.add_argument('exp_name', type=str, help='Experiment name')
    parser.add_argument('run_name', type=str, help='Run name')
    parser.add_argument('save_dir', type=str, help='Save directory')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Assign the parsed arguments to variables
    exp_name = args.exp_name
    run_name = args.run_name
    
    
    # read artifact from mlflow
    helper = MlflowHelper()
    run_id = helper.get_id_by_name(exp_name, run_name)
    artifact_dict = helper.get_artifact_dict_by_id(run_id)
    restore_opts = read_json(artifact_dict['options.json'])
    
    # if save_dir is 'same', save to artifact_dir
    if args.save_dir == 'same':
        save_dir = artifact_dict['artifacts_dir']
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)


    # copy load options
    optobj = Options()
    optobj.opts = restore_opts

    device = set_device('cuda')
    set_seed(0)
    
    prob = GBMproblem(**optobj.opts['pde_opts'])
    prob.print_info()

    prob.setup_dataset(optobj.opts['dataset_opts'], device=device)
    net = prob.setup_network(**optobj.opts['nn_opts'])

    # restore network
    net.load_state_dict(torch.load(artifact_dict['net.pth']))


    prob.make_prediction(net)
    prob.visualize(savedir= save_dir)
