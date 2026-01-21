#!/usr/bin/env python
# define problems for PDE
import os
import torch

from Options import *
from util import *
from DenseNet import DenseNet
from GBMDataset import GBMDataset
from BaseProblem import BaseProblem

from GBMplot import *

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

def reg_penalty(x, ref):
    # relu loss
    return (x - ref)**2

class GBMproblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
        
        self.opts = kwargs

        self.prior_th1 = kwargs.get('prior_th1', 0.3)
        self.prior_th2 = kwargs.get('prior_th2', 0.6)
        
        ### GBM custom loss
        self.loss_dict['seg1'] = self.get_seg1_loss
        self.loss_dict['seg2'] = self.get_seg2_loss

        self.loss_dict['dice1'] = self.get_dice1_metric
        self.loss_dict['dice2'] = self.get_dice2_metric

        self.loss_dict['uchar_res'] = self.get_uchar_res_loss
        self.loss_dict['uchar_dat'] = self.get_uchar_dat_loss
        self.loss_dict['ugt_res'] = self.get_ugt_res_loss
        self.loss_dict['ugt_dat'] = self.get_ugt_dat_loss
        
        # finite difference
        self.loss_dict['fdmbc'] = self.fdmbcloss
        self.loss_dict['res_fdm'] = self.resloss_fdm

        # regularization of the parameters
        self.loss_dict['rD_reg'] = self.regD_loss
        self.loss_dict['rRHO_reg'] = self.regRHO_loss
        self.loss_dict['th1_reg'] = self.regth1_loss
        self.loss_dict['th2_reg'] = self.regth2_loss


        self.loss_dict['thrange'] = self.range_constraint

        ### GBM specific options
        self.whichdata = kwargs['whichdata']
        # assert whichdata in ['char', 'gt', 'pat']
        assert self.whichdata in ['char', 'gt', 'pat'], f'whichdata {self.whichdata} not recognized'

        # for patient simulation, no need to load ugt and uchar into memory
        if self.whichdata == 'pat':
            ignore_fields = ['ugt','uchar']
        else:
            ignore_fields = []

        self.dataset = GBMDataset(kwargs['datafile'],ignore = ignore_fields)
        
        # finite difference method options
        self.use_fdm = kwargs.get('use_fdm', False)  # default to automatic differentiation
        self.fdm_Nt = kwargs.get('fdm_Nt', 100)      # default time intervals for FDM

        # range of th1 and th2
        self.th1_range = kwargs['th1_range']
        self.th2_range = kwargs['th2_range']
        self.rD_range = kwargs['rD_range']
        self.rRHO_range = kwargs['rRHO_range']

        ### dimension
        self.xdim = int(self.dataset['xdim'])
        self.dim = self.xdim + 1 # add time dimension
        self.input_dim = self.dim
        self.output_dim = 1


        # get parameter from mat file
        # check empty string
        # inititalize parameters 
        self.init_param = {}
        self.init_param['rD'] = 1.0
        self.init_param['rRHO'] = 1.0
        self.init_param['th1'] = self.opts['init_param']['th1'] if 'th1' in self.opts['init_param'] else self.prior_th1
        self.init_param['th2'] = self.opts['init_param']['th2'] if 'th2' in self.opts['init_param'] else self.prior_th2

        self.all_params_dict['rD'] = 1.0
        self.all_params_dict['rRHO'] = 1.0
        self.all_params_dict['th1'] = self.opts['init_param']['th1'] if 'th1' in self.opts['init_param'] else self.prior_th1
        self.all_params_dict['th2'] = self.opts['init_param']['th2'] if 'th2' in self.opts['init_param'] else self.prior_th2


        self.pde_params = ['rD', 'rRHO']

        # ground truth parameters
        self.gt_param = {}
        if 'gt' in self.whichdata:
            self.gt_param['rD'] = self.dataset['rDe']
            self.gt_param['rRHO'] = self.dataset['rRHOe']
        else:
            self.gt_param['rD'] = 1.0
            self.gt_param['rRHO'] = 1.0


        self.DW = self.dataset['DW']
        self.RHO = self.dataset['RHO']
        self.L = self.dataset['L']
        self.h_init = self.dataset.get('h_init', 0.1)
        self.r_init = self.dataset.get('r_init', 0.1)

        # if rmax is defined, use it
        self.r2max = (self.dataset['rmax']/self.L)**2
        
        # DO NOT USE self.x0
        # assuming x0 = [0,0,0]
        # self.x0 = self.dataset['x0']


        # for transformation
        self.lambda_transform = torch.nn.Module()
        self.lambda_transform.register_buffer('L', torch.tensor(self.L))
        if self.opts['pos_trans'] == True:
            # force the network to be positive
            self.lambda_transform.forward = lambda X, u: self.ic(X, self.lambda_transform.L) + torch.nn.functional.sigmoid(u) * X[:,0:1]
        elif self.opts['force_bc'] == True:
            # force bc same as ic at rmax
            print(f'force bc at r2max = {self.r2max}')
            self.lambda_transform.register_buffer('r2max', torch.tensor(self.r2max))
            self.lambda_transform.forward = lambda X, u, param: (self.ic(X, self.lambda_transform.L) + u * X[:,0:1]) * (self.lambda_transform.r2max - r2(X))/self.lambda_transform.r2max
            # logistic growth
            # self.lambda_transform.forward = lambda X, u, param: self.ic(X, self.lambda_transform.L)* torch.exp(X[:,0:1] * param['rRHO'])/(1 - self.ic(X, self.lambda_transform.L) + self.ic(X, self.lambda_transform.L)* torch.exp(X[:,0:1] * param['rRHO'])) + u * X[:,0:1] * (self.lambda_transform.rmax - r2(X))/self.lambda_transform.rmax
        elif self.opts['fixdebug'] == True:
            # fixed transformation for debugging
            self.lambda_transform.forward = lambda X, u, param: X[:,0:1]**0 + param['rD']
        else:
            self.lambda_transform.forward = lambda X, u, param: self.ic(X, self.lambda_transform.L) + u * X[:,0:1]


    # for testing purpose, still need to set whichdata
    def get_uchar_res_loss(self, net):
        # mse of uchar_res
        data = self.dataset.batch['st']
        X = data['X_st']
        u = data['uchar_st']
        phi = data['phi_st']
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
        data = self.dataset.batch['st']
        X = data['X_st']
        u = data['ugt_st']
        phi = data['phi_st']
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

    def regD_loss(self, net):
        # regularization loss for rD
        rD = net.pde_params_dict['rD'].squeeze()
        return reg_penalty(rD, 1.0)

    def regRHO_loss(self, net):
        # regularization loss for rRHO
        rRHO = net.pde_params_dict['rRHO'].squeeze()
        return reg_penalty(rRHO, 1.0)
    
    def regth1_loss(self, net):
        # regularization loss for th1
        th1 = net.all_params_dict['th1'].squeeze()
        return reg_penalty(th1, self.prior_th1)

    def regth2_loss(self, net):
        # regularization loss for th2
        th2 = net.all_params_dict['th2'].squeeze()
        return reg_penalty(th2, self.prior_th2)

    # compute validation statistics
    @torch.no_grad()
    def validate(self, nn):
        '''compute err '''
        v_dict = {}
        for vname in nn.trainable_param:
            v_dict[vname] = nn.all_params_dict[vname]
            if vname in self.gt_param and vname in nn.pde_params_dict:
                err = torch.abs(nn.pde_params_dict[vname] - self.gt_param[vname])
                v_dict[f'abserr_{vname}'] = err
        return v_dict

    def ic(self, X, L):
        # initial condition
        r2 = sumcol(torch.square((X[:, 1:self.dim])*L)) # this is in pixel scale, unit mm, 
        return self.h_init*torch.exp(-self.r_init*r2)


    def residual(self, nn, X, phi, P, gradPphi):
        
        # Get the number of dimensions
        n = X.shape[0]

        # split each column of X into a separate tensor of size (n, 1)
        vars = [X[:, d:d+1] for d in range(self.dim)]
        
        t = vars[0].detach()  # time variable detached for weighting if needed
        
        
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

        w = 1.0
        if self.opts['causal_weight']:
            # res_grid_sqr = res_grid**2
            res_p2 = res.detach()**2
            # cumulative sum along t, this is okay because t is sorted
            res_cumsum = torch.cumsum(res_p2, dim=0)
            w_t = torch.sqrt(torch.exp(-self.opts['texp_weight'] * res_cumsum))

        else:
            w_t = torch.sqrt(torch.exp(-self.opts['texp_weight'] * t))  # time weighting exp(-w*t), sqrt because res is squared later
        

        r = torch.sqrt(r2(X[:, 1:])).detach()  # r detached for weighting if needed
        r = r/torch.max(r)
        w_r = 1e-3 + torch.pow(r, self.opts['rpow_weight'])  # radial weighting 1/(1e-3 + r^p)
        w_r = torch.pow(w_r, -0.5)
        
        res = res * w_t * w_r

        return res, u

    
    
    def get_res_pred(self, net):
        # get residual and prediction
        if self.use_fdm:
            # Use finite difference method
            return self.get_res_pred_fdm(net, self.fdm_Nt)
        else:
            # Use automatic differentiation (original method)
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

    def get_res_pred_fdm(self, net, Nt):
        """
        Get residual and prediction using finite difference method
        Alternative to get_res_pred that uses automatic differentiation
        
        Args:
            net: neural network
            Nt: number of time intervals
            
        Returns:
            res: flattened residual for compatibility with existing loss functions
            u_pred: prediction at residual points for compatibility
        """
        # Compute FDM residual on grid
        res_grid, u_grid = self.fdm_residual(net, Nt)
        
        # Store for later use
        self.res = res_grid.reshape(-1, 1)  # Flattened residual
        self.upred_res = u_grid.reshape(-1, 1)  # Flattened prediction
        
        return self.res, self.upred_res

    def resloss_fdm(self, net):
        """
        Residual loss using finite difference method
        Alternative to the standard resloss that uses automatic differentiation
        """
        self.res, self.upred_res = self.get_res_pred_fdm(net, self.fdm_Nt)
        val_loss_res = mse(self.res)
        return val_loss_res

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
    

    def fdmbcloss(self, net):
        # get dirichlet boundary condition loss for FDM
        # Zero boundary conditions - just square the boundary values
        
        # Get network predictions on grid
        u_grid = self.u_grid 
        
        if self.xdim == 1:
            # Collect boundary values: left and right edges for all time
            u_boundary = torch.cat([u_grid[0, :], u_grid[-1, :]])  # (2*nt,)
            
        elif self.xdim == 2:
            # Collect boundary values: all four edges for all time
            u_left  = u_grid[0, :, :].flatten()     # Left boundary
            u_right = u_grid[-1, :, :].flatten()   # Right boundary  
            u_bot   = u_grid[:, 0, :].flatten()   # Bottom boundary
            u_top   = u_grid[:, -1, :].flatten()     # Top boundary
            
            u_boundary = torch.cat([u_left, u_right, u_bot, u_top])
        else:
            raise ValueError(f"Unsupported spatial dimension: {self.xdim}. Only 1D and 2D are supported.")
        
        # Zero boundary condition: just square the boundary values
        loss = torch.mean(u_boundary**2)
        
        return loss

    def range_constraint(self, net):
        # range constraint for th1 and th2

        th1 = net.all_params_dict['th1'].squeeze()
        th2 = net.all_params_dict['th2'].squeeze()

        loss = range_penalty(th1, self.th1_range[0], self.th1_range[1]) + range_penalty(th2, self.th2_range[0], self.th2_range[1])
        return loss
    
    def print_info(self):
        pass

    
    
    @torch.no_grad()
    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        self.dataset.to_device(self.dataset.device)
        net.to(self.dataset.device)

        upred_grid = self.grid_forward(net, self.dataset['t_scaled'])

        self.names2save = ['upred', 'rD_pred', 'rRHO_pred', 'th1_pred', 'th2_pred']
        self.dataset['upred'] = upred_grid    
        # prediction of parameters
        self.dataset['rD_pred'] = net.pde_params_dict['rD']
        self.dataset['rRHO_pred'] = net.pde_params_dict['rRHO']
        self.dataset['th1_pred'] = net.all_params_dict['th1']
        self.dataset['th2_pred'] = net.all_params_dict['th2']

        self.prediction_variation(net, list_params=['rD', 'rRHO'])

    def grid_forward(self, net, t_array):
        """
        Transform grid tensor to coordinate arrays for network forward pass
        
        Args:
            t_array: nt x 1 time points
            net: neural network
            
        Returns:
            u_grid: (nx, nt) for 1D, (nx, ny, nt) for 2D, (nx, ny, nz, nt) for 3D
                    Dimension order follows MATLAB ndgrid convention: (x, y, z, t)
        """
        t_array = t_array.view(-1, 1)  # ensure shape (nt, 1)
        nt = int(t_array.shape[0])

        # Spatial axes
        gx = self.dataset['gx_scaled']
        nx = int(gx.shape[0])

        if self.xdim >= 2:
            gy = self.dataset['gy_scaled']
            ny = int(gy.shape[0])
        if self.xdim >= 3:
            gz = self.dataset['gz_scaled']
            nz = int(gz.shape[0])

        # Precompute flattened spatial coordinates 
        # gx/gy/gz_mesh defiend in GBMDataset following MATLAB ndgrid (Fortran order).
        # note that reshape(-1,1) here give order that is different from matlab, but okay when we do the same reshap back
        if self.xdim == 1:
            space_coords = self.dataset['gx_mesh'].reshape(-1, 1)  # (nx, 1)
            n_space = nx
        elif self.xdim == 2:
            Xg = self.dataset['gx_mesh']  # (nx, ny)
            Yg = self.dataset['gy_mesh']  # (nx, ny)
            # Fortran-order flattening: permute dims then flatten in C-order
            perm = (1, 0)
            x_flat = Xg.reshape(-1, 1)
            y_flat = Yg.reshape(-1, 1)
            space_coords = torch.cat([x_flat, y_flat], dim=1)  # (nx*ny, 2)
            n_space = nx * ny
        elif self.xdim == 3:
            Xg = self.dataset['gx_mesh']  # (nx, ny, nz)
            Yg = self.dataset['gy_mesh']  # (nx, ny, nz)
            Zg = self.dataset['gz_mesh']  # (nx, ny, nz)
            perm = (2, 1, 0)
            x_flat = Xg.reshape(-1, 1)
            y_flat = Yg.reshape(-1, 1)
            z_flat = Zg.reshape(-1, 1)
            space_coords = torch.cat([x_flat, y_flat, z_flat], dim=1)  # (nx*ny*nz, 3)
            n_space = nx * ny * nz
        else:
            raise ValueError(f"Unsupported spatial dimension: {self.xdim}. Only 1D/2D/3D are supported.")

    
        # Order preserved: for each time, list all space points.
        # Avoid materializing the full X_input (and even full X_input_t) to reduce peak memory.
        batch_size = 1_000_000
        u_chunks = []
        for it in range(nt):
            t_val = t_array[it:it + 1]  # (1, 1)
            for s in range(0, n_space, batch_size):
                e = min(s + batch_size, n_space)
                xs = space_coords[s:e]
                t_rep = t_val.expand(int(xs.shape[0]), 1)
                X = torch.cat([t_rep, xs], dim=1)
                u_chunks.append(net(X, net.pde_params_dict))

        u_flat = torch.cat(u_chunks, dim=0)

        
        if self.xdim == 1:
            u_grid = u_flat.view(nt,nx).permute(1, 0)  # (nx, nt)
            # set boundary values to zero
            u_grid[0, :] = 0.0
            u_grid[-1, :] = 0.0
        elif self.xdim == 2:
            u_grid = u_flat.view(nt,nx,ny).permute(1, 2, 0)  # (nx, ny, nt)
            u_grid[0, :, :] = 0.0
            u_grid[-1, :, :] = 0.0
            u_grid[:, 0, :] = 0.0
            u_grid[:, -1, :] = 0.0
        else:
            u_grid = u_flat.view(nt,nx,ny,nz).permute(1, 2, 3, 0)  # (nx, ny, nz, nt)
            u_grid[0, :, :, :] = 0.0
            u_grid[-1, :, :, :] = 0.0
            u_grid[:, 0, :, :] = 0.0
            u_grid[:, -1, :, :] = 0.0
            u_grid[:, :, 0, :] = 0.0
            u_grid[:, :, -1, :] = 0.0

        return u_grid

    def compute_spatial_derivatives_fd(self, u_grid):
        """
        Compute spatial derivatives using central difference
        
        Args:
            u_grid: (x, y ,z, t)
        """
        h = (self.dataset['gx_scaled'][1] - self.dataset['gx_scaled'][0]).item()  # assuming uniform grid
        if self.xdim == 1:
            nx, nt = u_grid.shape
            
            # Initialize derivative tensors
            u_x = torch.zeros_like(u_grid)
            u_xx = torch.zeros_like(u_grid)
            
            # Central difference for first derivatives (skip boundaries)
            # u_x = (u[i+1] - u[i-1]) / (2*dx)
            u_x[1:-1, :] = (u_grid[2:,:] - u_grid[:-2, :]) / (2 * h)
            
            # Central difference for second derivatives (skip boundaries)
            # u_xx = (u[i+1] - 2*u[i] + u[i-1]) / h^2
            u_xx[1:-1, :] = (u_grid[2:,:] - 2*u_grid[1:-1, :] + u_grid[:-2, :]) / (h**2)
            
            return u_x, u_xx
            
        elif self.xdim == 2:
            nx, ny, nt = u_grid.shape
            
            # Initialize derivative tensors
            u_x = torch.zeros_like(u_grid)
            u_y = torch.zeros_like(u_grid)
            u_xx = torch.zeros_like(u_grid)
            u_yy = torch.zeros_like(u_grid)
            
            # Central difference for first derivatives (skip boundaries)
            # u_x = (u[i+1] - u[i-1]) / (2*dx)
            u_x[1:-1, :, :] = (u_grid[2:, :, :] - u_grid[:-2, :, :]) / (2 * h)
            u_y[:, 1:-1, :] = (u_grid[:, 2:, :] - u_grid[:, :-2, :]) / (2 * h)
            
            # Central difference for second derivatives (skip boundaries)
            # u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx^2
            u_xx[1:-1, :, :] = (u_grid[2:, :, :] - 2*u_grid[1:-1, :, :] + u_grid[:-2, :, :]) / (h**2)
            u_yy[:, 1:-1, :] = (u_grid[:, 2:, :] - 2*u_grid[:, 1:-1, :] + u_grid[:, :-2, :]) / (h**2)
            
            return u_x, u_y, u_xx, u_yy
        else:
            nx, ny, nz, nt = u_grid.shape
            # Initialize derivative tensors
            u_x = torch.zeros_like(u_grid)
            u_y = torch.zeros_like(u_grid)
            u_z = torch.zeros_like(u_grid)
            u_xx = torch.zeros_like(u_grid)
            u_yy = torch.zeros_like(u_grid)
            u_zz = torch.zeros_like(u_grid)

            # Central difference for first derivatives (skip boundaries)
            u_x[1:-1, :, :, :] = (u_grid[2:, :, :, :] - u_grid[:-2, :, :, :]) / (2 * h)
            u_y[:, 1:-1, :, :] = (u_grid[:, 2:, :, :] - u_grid[:, :-2, :, :]) / (2 * h)
            u_z[:, :, 1:-1, :] = (u_grid[:, :, 2:, :] - u_grid[:, :, :-2, :]) / (2 * h)

            # Central difference for second derivatives (skip boundaries)
            u_xx[1:-1, :, :, :] = (u_grid[2:, :, :, :] - 2*u_grid[1:-1, :, :, :] + u_grid[:-2, :, :, :]) / (h**2)
            u_yy[:, 1:-1, :, :] = (u_grid[:, 2:, :, :] - 2*u_grid[:, 1:-1, :, :] + u_grid[:, :-2, :, :]) / (h**2)
            u_zz[:, :, 1:-1, :] = (u_grid[:, :, 2:, :] - 2*u_grid[:, :, 1:-1, :] + u_grid[:, :, :-2, :]) / (h**2)
            return u_x, u_y, u_z, u_xx, u_yy, u_zz

    def fdm_residual(self, nn, Nt):
        """
        Compute residual using finite difference method
        Alternative to the current automatic differentiation approach
        
        Args:
            nn: neural network
            Nt: number of time intervals (if None, use self.fdm_Nt or t_array from dataset)
            
        Returns:
            res_grid: nx (,ny (,nz)) nt-1 tensor of residuals (excluding boundaries)
            u_grid: nx (,ny (,nz)) nt tensor of predictions
        """
        # Use pre-computed time array if available, otherwise create one
        dt = 1.0 / Nt
        t_array = torch.linspace(0, 1, Nt + 1, device=nn.pde_params_dict['rD'].device).view(-1, 1)
        
        # Get network predictions on grid
        u_grid = self.grid_forward(nn, t_array)
        
        # Get grid spacing from coordinates
        gx = self.dataset['gx_scaled']
        
        if self.xdim == 1:
            # Compute spatial derivatives for 1D
            u_x, u_xx = self.compute_spatial_derivatives_fd(u_grid)
            
            nt, nx = u_grid.shape
            
            # backward difference for time derivative, u_t = 0 at t=0
            u_t = torch.zeros_like(u_grid)
            u_t[:, 1:] = (u_grid[:,1:] - u_grid[:, :-1]) / dt
            
            
            # Get problem parameters and grid data, for 1D, data is (nx,1)
            phi = self.dataset['phi'].reshape(-1, 1) # (nx, 1)
            P = self.dataset['P'].reshape(-1,1)    # (nx, 1)
            DxPphi = self.dataset['DxPphi'].reshape(-1,1) # (nx, 1)

            
            # Compute residual terms
            # Proliferation term: rRHO * RHO * phi * u * (1 - u)
            # need to reshap params_expand to same as as u
            
            prof = nn.params_expand['rRHO'].reshape(nt, nx) * self.RHO * phi * u_grid * (1 - u_grid)
            
            # Diffusion term: rD * DW * (P * phi * u_xx + L * DxPphi * u_x)
            diff = nn.params_expand['rD'].reshape(nt, nx) * self.DW * (P * phi * u_xx + self.L * DxPphi * u_x)
            
            # Full residual: phi * u_t - (prof + diff)
            res_grid = phi * u_t - (prof + diff)
            
            # Set boundary residuals to zero (don't evaluate at boundaries)
            res_grid[:, 0] = 0    # t=0
            res_grid[0, :] = 0    # left boundary
            res_grid[-1,:] = 0   # right boundary
            
        elif self.xdim == 2:
            gy = self.dataset['gy']
            nx, ny, nt = u_grid.shape
            
            # Compute spatial derivatives for 2D
            u_x, u_y, u_xx, u_yy = self.compute_spatial_derivatives_fd(u_grid)
            
            # Compute time derivative (results in nt-1 time points)
            u_t = torch.zeros_like(u_grid)

            # backward difference for time derivative
            u_t[:, :, 1:] = (u_grid[:, :, 1:] - u_grid[:, :, :-1]) / dt
            u_t[:, :, 0] = u_t[:, :, 1]  # forward diff at t=0

            # forward difference for time derivative
            # u_t[:, :, :-1] = (u_grid[:, :, 1:] - u_grid[:, :, :-1]) / dt
            
            
            # Get problem parameters and grid data
            phi = self.dataset['phi'].unsqueeze(-1)  # (nx,ny)
            P = self.dataset['P'].unsqueeze(-1)      # (nx,ny)
            DxPphi = self.dataset['DxPphi'].unsqueeze(-1)  # (nx,ny)
            DyPphi = self.dataset['DyPphi'].unsqueeze(-1)  # (nx,ny)
            
            # Compute residual terms
            # Proliferation term: rRHO * RHO * phi * u * (1 - u)
            prof = nn.params_expand['rRHO'].reshape_as(u_grid) * self.RHO * phi * u_grid * (1 - u_grid)
            
            # Diffusion term: rD * DW * (P * phi * (u_xx + u_yy) + L * (DxPphi * u_x + DyPphi * u_y))
            laplacian = u_xx + u_yy
            gradient_term = DxPphi * u_x + DyPphi * u_y
            diff = nn.params_expand['rD'].reshape_as(u_grid) * self.DW * (P * phi * laplacian + self.L * gradient_term)
            
            # Full residual: phi * u_t - (prof + diff)
            res_grid = phi * u_t - (prof + diff)
            
            # Set boundary residuals to zero (don't evaluate at boundaries)
            res_grid[:, :, 0] = 0    # t=0
            res_grid[0,  :,  :] = 0    # left boundary
            res_grid[-1, :, :] = 0   # right boundary
            res_grid[:,  0, :] = 0    # bottom boundary
            res_grid[:, -1, :] = 0   # top boundary

            # simple scheme
            # weight residual by exp(- w t)
            time_weights = torch.exp(- self.opts['texp_weight']*t_array).view(1, 1, -1).expand(nx, ny, -1)
            
            # Causal training
            # res_grid_sqr = res_grid**2
            # cumulative sum along t
            # time_weights = torch.cumsum(res_grid_sqr, dim=0)
            # time_weights = torch.exp(-time_weights)

            res_grid = res_grid * time_weights
        else:
            gz = self.dataset['gz']
            nx, ny, nz, nt = u_grid.shape
            
            # Compute spatial derivatives for 3D
            u_x, u_y, u_z, u_xx, u_yy, u_zz = self.compute_spatial_derivatives_fd(u_grid)
            
            # Compute time derivative (results in nt-1 time points)
            u_t = torch.zeros_like(u_grid)

            # backward difference for time derivative
            u_t[:, :, :, 1:] = (u_grid[:, :, :, 1:] - u_grid[:, :, :, :-1]) / dt
            u_t[:, :, :, 0] = u_t[:, :, :, 1]  # forward diff at t=0

            # Get problem parameters and grid data
            phi = self.dataset['phi'].unsqueeze(-1)  # (nx,ny,nz)
            P = self.dataset['P'].unsqueeze(-1)      # (nx,ny,nz)
            DxPphi = self.dataset['DxPphi'].unsqueeze(-1)  # (nx,ny,nz)
            DyPphi = self.dataset['DyPphi'].unsqueeze(-1)  # (nx,ny,nz)
            DzPphi = self.dataset['DzPphi'].unsqueeze(-1)  # (nx,ny,nz)
            
            # Compute residual terms
            prof = nn.params_expand['rRHO'].reshape_as(u_grid) * self.RHO * phi * u_grid * (1 - u_grid)
            
            laplacian = u_xx + u_yy + u_zz
            gradient_term = DxPphi * u_x + DyPphi * u_y + DzPphi * u_z
            diff = nn.params_expand['rD'].reshape_as(u_grid) * self.DW * (P * phi * laplacian + self.L * gradient_term)
            
            res_grid = phi * u_t - (prof + diff)
            
            # Set boundary residuals to zero (don't evaluate at boundaries)
            res_grid[:, :, :, 0] = 0    # t=0
            res_grid[0,  :,  :, :] = 0    # left boundary
            res_grid[-1, :,  :, :] = 0  # right boundary
            res_grid[:,  0,  :, :] = 0    # bottom boundary
            res_grid[:, -1,  :, :] = 0   # top boundary
            res_grid[:, :,  0, :] = 0    # front boundary
            res_grid[:, :, -1, :] = 0   # back boundary
        
        return res_grid, u_grid

    
    
    @error_logging_decorator
    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.dataset.visualize_sampling(savedir)

        th1_pred = self.dataset.get('th1_pred', None)
        th2_pred = self.dataset.get('th2_pred', None)

        # Prediction grid uses ndgrid convention with dims (x[,y[,z]], t)
        ugrid = self.dataset['upred']  # shape (nx[,ny[,nz]], nt)
        phiugrid = ugrid * self.dataset['phi'][..., np.newaxis]  # apply phi mask
        nt_pred = int(phiugrid.shape[-1])
        t_indices = np.linspace(0, nt_pred - 1, 5).astype(int)

        # Optional reference on the same grid (gt/char only)
        ref_key = None
        if 'gt' in self.whichdata and 'ugt' in self.dataset:
            ref_key = 'ugt'
        elif 'char' in self.whichdata and 'uchar' in self.dataset:
            ref_key = 'uchar'
        uref = self.dataset[ref_key] if ref_key is not None else None

        # scatter plot of solution at t_indices vs radius
        space_mesh = ("gx_mesh", "gy_mesh", "gz_mesh")[:self.xdim]
        X_space_full = np.stack([self.dataset[m].ravel(order="F") for m in space_mesh], axis=1)
        r_full = np.linalg.norm(X_space_full, axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('viridis')
        for it in t_indices:
            phiu_it = phiugrid[..., it].ravel(order="F")
            uref_it = uref[..., it].ravel(order="F") if uref is not None else None
            t = self.dataset['t_scaled'][it]
            color = cmap(t)
            ax.scatter(r_full, phiu_it, color=color, marker='o',label=f'pred t={t:.2f}')
            if uref_it is not None:
                ax.scatter(r_full, uref_it, color=color, marker='x',label=f'ref t={t:.2f}')
            ax.legend(loc='upper right')
            # add grid
            ax.grid(True)
            ax.set_xlabel('Radius r')
            ax.set_ylabel('u')
            min_u = min(np.min(phiu_it), np.min(uref_it) if uref_it is not None else np.inf)
            ax.set_ylim([min_u, 1.0])
            ax.set_title('Scatter Plot of u vs Radius')
        
        if savedir is not None:
            fpath = os.path.join(savedir, 'fig_scatter_u_vs_r.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        
        # contour and imshow plotting functions
        if self.xdim == 1:
            # For 1D, plot line profiles at different times
            xgrid = self.dataset['gx'].squeeze()
                
            fig, ax = plt.subplots(figsize=(8, 6))
            for it in t_indices:
                t_val = it / max(1, (nt_pred - 1))  # normalize time to [0,1]
                ax.plot(xgrid, phiugrid[:, it], label=f't={t_val:.2f}')
                if uref is not None:
                    ax.plot(xgrid, uref[:, it], '--', label=f't={t_val:.2f} (ref)')
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title('Grid Solution at Different Times')
            ax.legend()
            ax.grid(True)
            
            if savedir is not None:
                fpath = os.path.join(savedir, 'fig_1d.png')
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')
            plt.close(fig)
                
        elif self.xdim == 2:

            # Overlay predicted segmentation contours over GT u1/u2 fills
            if self.whichdata in ['pat', 'gt']:
                pred2d = phiugrid[:, :, -1]
                plot_grid_segmentation_overlay(
                    pred2d,
                    th1=th1_pred,
                    th2=th2_pred,
                    u1=self.dataset.get('u1', None),
                    u2=self.dataset.get('u2', None),
                    fname=f'fig_2d_seg_overlay_t{t_val:.2f}.png',
                    savedir=savedir,
                    title=f'Segmentation overlay t={t_val:.2f}',
                )

        
            # For 2D: 5 time points, imshow panels + contour overlays.
            levels = np.array([0.01, 0.1, 0.3, 0.6])
            for it in t_indices:
                t_val = it / max(1, (nt_pred - 1))
                pred2d = phiugrid[:, :, it]

                ref2d = None
                if uref is not None:
                    ref2d = uref[:, :, it]

                title = f'Grid (2D) t={t_val:.2f}'

                plot_grid_imshow_panels(
                    pred2d,
                    ref2d,
                    fname=f'fig_2d_panels_t{t_val:.2f}.png',
                    savedir=savedir,
                    title=title,
                )

                # contour overlay (same levels)
                plot_grid_contour_overlay(
                    pred2d,
                    ref2d,
                    levels=levels,
                    fname=f'fig_2d_contour_t{t_val:.2f}.png',
                    savedir=savedir,
                    title=f'Contours t={t_val:.2f}' if ref2d is not None else f'Contours (pred) t={t_val:.2f}',
                )
        else:
            # For 3D: same as 2D, but use middle z-slice
            if self.xdim == 3:
                
                levels = np.array([0.01, 0.1, 0.3, 0.6])
                nz = int(phiugrid.shape[2])
                kz = nz // 2

                if self.whichdata in ['gt', 'pat']:
                    pred2d = phiugrid[:, :, kz, -1]
                    plot_grid_segmentation_overlay(
                        pred2d,
                        th1=th1_pred,
                        th2=th2_pred,
                        u1=self.dataset['u1'][:, :, kz],
                        u2=self.dataset['u2'][:, :, kz],
                        fname=f'fig_3d_midZ{kz}_seg_overlay.png',
                        savedir=savedir,
                        title=f'Segmentation overlay mid-z={kz}',
                    )



                for it in t_indices:
                    t_val = it / max(1, (nt_pred - 1))
                    pred2d = phiugrid[:, :, kz, it]

                    ref2d = None
                    if uref is not None:
                        ref2d = uref[:, :, kz, it]

                    title = f'Grid (3D mid-z={kz}) t={t_val:.2f}'

                    plot_grid_imshow_panels(
                        pred2d,
                        ref2d,
                        fname=f'fig_3d_midZ{kz}_panels_t{t_val:.2f}.png',
                        savedir=savedir,
                        title=title,
                    )

                    plot_grid_contour_overlay(
                        pred2d,
                        ref2d,
                        levels=levels,
                        fname=f'fig_3d_midZ{kz}_contour_t{t_val:.2f}.png',
                        savedir=savedir,
                        title=f'Contours mid-z={kz} t={t_val:.2f}' if ref2d is not None else f'Contours mid-z={kz} (pred) t={t_val:.2f}',
                    )
            else:
                # higher dimensions not implemented
                pass

        plt.close('all')
    
    @error_logging_decorator
    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter compare with 0 variation
        # only 2d
        if self.xdim not in [1,2]:
            print('plot_variation only supports 1d and 2d')
            return
        if 'deltas' not in self.dataset:
            print('no variation data found in dataset')
            return
        
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
                    fig, ax = plot_2d_contour(x_dat, u_pred, u_pred_base, fname = f'{vname_pred}.png', savedir=savedir, title=title_str)
                
                if self.xdim == 1:
                    fig, ax = plot_1d_profile(x_dat, u_pred, u_pred_base, fname = f'{vname_pred}.png', savedir=savedir, title=title_str)
                
        
        plt.close('all')

        return
    
    def setup_dataset(self, ds_opts, noise_opts=None, device='cuda'):
        ''' downsample for training'''
        self.dataset.prepare_all_space_field()
        if self.whichdata in ['gt', 'char']:
            include_st = True
            self.dataset.prepare_all_spacetime_field(m_time=ds_opts.get('Nt_train', 10))
        else:
            include_st = False
        

        # put all variables to device (iterable samplers sample indices on GPU)
        self.dataset.to_device(device)

        # Create time array for FDM if using finite difference method
        if self.use_fdm:
            dt = 1.0 / self.fdm_Nt
            t_array = torch.linspace(0, 1, self.fdm_Nt + 1, device=device).view(-1, 1)
            self.dataset['t_array'] = t_array
            self.dataset['dt'] = dt
            print(f'Created time array for FDM: Nt={self.fdm_Nt}, dt={dt:.6f}, t_array shape={t_array.shape}')
        

        res_batch_size = int(ds_opts['net_batch_size'])
        dat_batch_size = int(ds_opts['pde_batch_size'])
        
        # for residual batch size, the larger the better, no limit
        # for data loss batch size, cannot be larger than dataset size
        
        if dat_batch_size > self.dataset['X_space_flat'].shape[0]:
            dat_batch_size = self.dataset['X_space_flat'].shape[0]
            ds_opts['pde_batch_size'] = dat_batch_size
            print(f'Adjusted dat_batch_size to {dat_batch_size} due to dataset size.')

        # Iterable samplers already randomize; keep workers=0 since tensors live on GPU.
        self.dataset.configure_dataloader(res_batch_size, dat_batch_size, include_st)
            
       

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
