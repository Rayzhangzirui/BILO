import torch

class PoissonTest:
    """
    Synthetic dataset generator for Poisson equation: -(Dx uxx + Dy uyy + Dz uzz) = f
    Supports multiple test cases and dimensions (1D/2D/3D)
    """
    
    def __init__(self, dim, D):
        self.dim = dim
        self.D = D  # Diffusion coefficients as dictionary

    def _u_gt_1d(self, x):
        """u = sin(π*x)"""
        x = x.view(-1, 1)
        return torch.sin(torch.pi * x)/self.D['D0']

    def _u_gt_2d(self, x, y):
        """u = sin(π*x)sin(2π*y)/(Dx + Dy)"""
        x = x.view(-1, 1)
        y = y.view(1, -1)
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) / (self.D['D0'] + self.D['D1'])

    def _u_gt_3d(self, x, y, z):
        """u = sin(π*x)sin(π*y)sin(π*z)/(Dx + Dy + Dz)"""
        x = x.view(-1, 1, 1)
        y = y.view(1, -1, 1)
        z = z.view(1, 1, -1)
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) / (self.D['D0'] + self.D['D1'] + self.D['D2'])
    
    def _f_gt_1d(self, x):
        """f = -D*uxx for u = sin(π*x)/D: f = π²*sin(π*x)"""
        x = x.view(-1,1)
        return torch.pi**2 * torch.sin(torch.pi * x)

    def _f_gt_2d(self, x, y):
        """f = -D₁*u_xx - D₂*u_yy for case 1"""
        x = x.view(-1, 1)
        y = y.view(1, -1)
        # u_xx = -π²*sin(πx)*sin(πy)/(Dx + Dy), 
        # u_yy = -π²*sin(πx)*sin(πy)/(Dx + Dy)
        # - (Dx u_xx + D_y u_yy)
        return torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    def _f_gt_3d(self, x, y, z):
        """f = -D₁*u_xx - D₂*u_yy - D₃*u_zz for case 1"""
        x = x.view(-1, 1, 1)
        y = y.view(1, -1, 1)
        z = z.view(1, 1, -1)
        # u_xx = -π²*sin(πx)*sin(πy)*sin(πz), 
        # u_yy = -π²*sin(πx)*sin(πy)*sin(πz)
        # u_zz = -π²*sin(πx)*sin(πy)*sin(πz)
        # Dx u_xx + Dy u_yy + Dz u_zz
        return torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)

    def generate_gt_grid(self, *coord_list):
        if self.dim == 1:
            u = self._u_gt_1d(coord_list[0])
            f = self._f_gt_1d(coord_list[0])
        elif self.dim == 2:
            u = self._u_gt_2d(coord_list[0], coord_list[1])
            f = self._f_gt_2d(coord_list[0], coord_list[1])
        elif self.dim == 3:
            u = self._u_gt_3d(coord_list[0], coord_list[1], coord_list[2])
            f = self._f_gt_3d(coord_list[0], coord_list[1], coord_list[2])
        else:
            raise NotImplementedError("Only 1D, 2D and 3D cases are implemented.")
        return u, f
    
    def generate_grid_data(self, N:int):
        # Generate coordinate list for any dimension
        coord_list = [torch.linspace(0, 1, N).view(-1, 1) for _ in range(self.dim)]
        
        u_gt, f_gt = self.generate_gt_grid(*coord_list)

        dataset = {}
        dataset['shape'] = u_gt.shape
        dataset['coord_list'] = coord_list
        dataset['u_gt'] = u_gt
        dataset['f_gt'] = f_gt
        return dataset