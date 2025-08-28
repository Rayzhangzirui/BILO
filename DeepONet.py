#!/usr/bin/env python
import sys
import torch
import torch.nn as nn

from MatDataset import MatDataset

from neuralop.models import FNO


class scalarFNO(nn.Module):
    # 1d FNO
    # take coordinate X as a channel. Expand the PDE parameter to the same size as X as different channels
    def __init__(self, param_dim=1, X_dim=1, output_dim=1, n_modes = 16, **kwargs):
        super(scalarFNO, self).__init__()

        self.param_dim = param_dim
        self.X_dim = X_dim
        self.output_dim = output_dim
        self.pde_params_dict = {}
        self.pde_params_list = []

        self.fno = FNO(n_modes=(n_modes,), hidden_channels=kwargs['hidden_channels'],
                in_channels=X_dim+param_dim, out_channels=output_dim,positional_embedding=None)
            
        self.lambda_transform = kwargs.get('lambda_transform', lambda x, u: u)

    def forward(self, P_input, X_input):
        """
        P_input: (B, k)  - PDE parameters
        X_input: (N, d)  - Spatial points
        Output:   (B, N) or (B, out_channels, N)
        """
        if isinstance(P_input, torch.nn.ParameterDict) or isinstance(P_input, dict):
            # if dictionary, concat to (1, k) tensor
            # dict for evlauating with tensor, ParameterDict for training
            P_input = torch.cat([P_input[key] for key in self.pde_params_list]).view(1, -1)

        B, k = P_input.shape  # (batch, k)
        N, d = X_input.shape  # (N, d)

        # 1) Transpose X_input -> (d, N)
        X_input_t = X_input.t()  # shape: (d, N)

        # 2) Broadcast to batch dimension -> (B, d, N)
        X_input_expanded = X_input_t.unsqueeze(0).expand(B, d, N)

        # 3) Broadcast P_input -> (B, k, N)
        #    We unsqueeze at dim=-1 and then expand along N.
        P_input_expanded = P_input.unsqueeze(-1).expand(B, k, N)

        # 4) Concatenate along channel dimension (dim=1): (B, d + k, N)
        X = torch.cat([X_input_expanded, P_input_expanded], dim=1)

        # Pass through your FNO which expects shape (B, in_channels, N).
        # Suppose FNO returns (B, out_channels, N). With out_channels=1, that's (B, 1, N).
        out = self.fno(X)

        # If you want a final shape (B, N), you can squeeze out_channels=1:
        out = out.squeeze(1)  # shape: (B, N)

        # Apply the lambda transform to each batch of output
        out = self.lambda_transform(X_input, out.t()).t() # (B, N)

        return out
        
    
    

class DeepONet(nn.Module):
    def __init__(self, 
    param_dim=1, X_dim=1, output_dim=1,
    width=64,  branch_depth=4, trunk_depth=4,
    lambda_transform=lambda x, u: u):

        super(DeepONet, self).__init__()
        # param_dim is the dimension of the PDE parameter space
        # X_dim is the dimension of the PDE domain

        # branch net is a neural network that processes the parameter set
        # trunk net is a neural network that processes the coordinates
        
        self.width = width
        self.param_dim = param_dim
        self.X_dim = X_dim
        self.output_dim = output_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.lambda_transform = lambda_transform
        self.pde_param = None

        # list or pde parameter in order of P
        self.pde_params_list = []


        self.branch_net = self.build_subnet(param_dim, branch_depth)
        self.trunk_net = self.build_subnet(X_dim, trunk_depth)
        
    def freeze(self):
        # freeze the parameters of the network
        for param in self.parameters():
            param.requires_grad = False
    
    def build_subnet(self, input_dim, depth):

        layers = [input_dim] + [self.width]*depth  # List of layer widths

        layers_list = []
        for i in range(len(layers) - 1):
            layers_list.append(nn.Linear(layers[i], layers[i+1]))  # Add linear layer
            layers_list.append(nn.Tanh())  # Add activation layer
        return nn.Sequential(*layers_list)

    def forward(self, P_input, X_input):
        # Process each parameter set in the branch network
        # P_input: (B, k)  - PDE parameters or dictionary of parameters

        if isinstance(P_input, dict) or isinstance(P_input, torch.nn.ParameterDict):
            # if dictionary, concat to (1, k) tensor
            P_input = torch.cat([P_input[key] for key in self.pde_params_list], dim=1).view(1, -1)

        branch_out = self.branch_net(P_input)  # [batch_size, width]

        # Process the fixed grid in the trunk network
        trunk_out = self.trunk_net(X_input)  # [num_points, width]

        # Compute the output as num_points x batch_size
        output = torch.mm(trunk_out, branch_out.t())  # [num_points, batch_size]

        # Apply the lambda transform to each batch of output
        output = self.lambda_transform(X_input, output)  # [num_points, batch_size]

        # tranpose the output to batch_size x num_points
        output = output.t()

        return output

# Note on output: The final `.squeeze()` is used to remove any unnecessary dimensions if `output_dim` is 1.

# operator learning dataset
class OpData(torch.utils.data.Dataset):
    #  N = gt*gx
    #  X = N-by-2 matrix of t and x
    #  U = m by N matrix of solutions
    #  P = m by 2 matrix of parameters
    def __init__(self, X, P, U):

        self.X = X
        self.P = P
        self.U = U
        
    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, idx):
        # get the idx-th item of P, X, and U
        
        return self.P[idx], self.U[idx]

# simple test of the network
if __name__ == "__main__":
    import sys
    # test the FKMatDataset class
    filename  = sys.argv[1]
    

    dataset = MatDataset(filename)
    
    dataset.printsummary()

    fkdata = OpData( dataset['X'], dataset['P'], dataset['U'])
    
    data_loader = torch.utils.data.DataLoader(fkdata, batch_size=10, shuffle=True)

    deeponet = DeepONet(2, 2, 1)

    # test dimension
    for data in data_loader:
        P, U = data
        u = deeponet(P, fkdata.X)
        print(u.shape)
        print(U.shape)
        break
