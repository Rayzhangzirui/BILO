#!/usr/bin/env python
import sys
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import loralib as lora

from util import *
from config import *
from MlflowHelper import MlflowHelper

from Options import *

# set module to be trainable or not
def set_module_require_grad(m, requires_grad):
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(requires_grad)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(requires_grad)
        
# the pde and the neural net is combined in one class
# the PDE parameter is also part of the network

# all_params_dict include all problem parameters, including PDE parameters and additioinal parameters
# pde_params include only PDE parameters in BILO
# trainable_param is subset of all_params_dict, only train part of the parameters
class DenseNet(nn.Module):
    def __init__(self, depth=4, width=8, input_dim=1, output_dim=1, 
                lambda_transform=lambda x, u, param: u,
                with_param: bool = False, 
                all_params_dict = None,
                pde_params: List[str] = [],
                trainable_param:List[str] = [],
                # for architecture
                use_resnet: bool = False, 
                fourier: bool = False,
                rff_trainable: bool = False,
                sigma: float = 1.0,
                siren: bool = False,
                omega_0: int = 30,
                act: str = 'tanh',
                modifiedmlp: bool = False,
                train_embed: bool = False,
                rank: int = 4,
                lora_alpha: float = 1.0,
                **kwargs):
        super().__init__()
        
        self.depth = depth
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_transform = lambda_transform # transform the output of the network, default is identity
        self.use_resnet = use_resnet
        self.with_param = with_param # if True, then the pde parameter is part of the network
        # random fourier feature, smaller sigma, smoother features.
        self.fourier = fourier
        self.rff_trainable = rff_trainable
        self.sigma = sigma
        # SIREN
        self.siren = siren
        self.omega_0 = omega_0
        # activation function
        self.act = act
        # modified mlp
        self.modifiedmlp = modifiedmlp
        # train the embedding weight of the PDE parameter
        self.train_embed = train_embed
        # for LoRA
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        self.with_func = False

        
        
        ### handle scalar parameters
        # convert float to tensor of size (1,1)
        # need ParameterDict to make it registered, otherwise to(device) will not automatically move it to device
        # tmp = {k: nn.Parameter(torch.tensor([[v]])) for k, v in all_params_dict.items()}
        tmp = {}
        for k, v in all_params_dict.items():
            if isinstance(v, torch.Tensor):
                tmp[k] = nn.Parameter(v)
            elif isinstance(v, float) or isinstance(v, int):
                tmp[k] = nn.Parameter(torch.tensor([[v]], dtype=torch.float32))
            elif isinstance(v, nn.Module):
                tmp[k] = v
                self.with_func = True
            else:
                raise ValueError(f'Unknown type for parameter {k}')

        self.all_params_dict = nn.ParameterDict(tmp)
        self.params_expand = {}
        
        # if pde_params is empty, then pde_params is the same as params_dict
        if len(pde_params) == 0:
            self.pde_params_dict = self.all_params_dict
        else:
            self.pde_params_dict = {k: self.all_params_dict[k] for k in pde_params}

        # list of pde/net parameters for upper/lower optimization
        self.param_net = []
        self.param_pde = []

        ### setup architecture
        if self.modifiedmlp:
            self.setup_modified_mlp()
        else:
            self.setup_basic_architecture()

        
        ### setup trainable parameters

        # trainable_param is a list of parameter names
        # only train part of the parameters

        # For now, set all PDE parameters to be trainable, for initialization, set lr=0
        # self.trainable_param = list(self.all_params_dict.keys())

        self.trainable_param = trainable_param
        for name, param in self.all_params_dict.items():
            if name not in self.trainable_param:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # temporary solution, if with_func, this part will be overwritten, hence don't run this part here
        self.collect_trainable_param()

    def setup_basic_architecture(self):
        
        input_dim = self.input_dim
        width = self.width
        output_dim = self.output_dim
        depth = self.depth

        if self.fourier:
            self.fflayer = nn.Linear(input_dim, width)
            
            # initialize with N(0,1) weights
            torch.nn.init.normal_(self.fflayer.weight, mean=0, std=1)
            # init the bias to be unif[0,1]
            torch.nn.init.uniform_(self.fflayer.bias, a=0, b=2*torch.pi)

            set_module_require_grad(self.fflayer, self.rff_trainable)
            self.input_layer = lora.Linear(width, width, r=self.rank, lora_alpha=self.lora_alpha)
        else:
            self.input_layer = lora.Linear(input_dim, width, r=self.rank, lora_alpha=self.lora_alpha)

        # depth = input + hidden + output
        self.hidden_layers = nn.ModuleList([lora.Linear(width, width,r=self.rank, lora_alpha=self.lora_alpha) for _ in range(depth - 2)])
        self.output_layer = lora.Linear(width, output_dim, r=self.rank, lora_alpha=self.lora_alpha)

        # initailize the weights with glorot uniform
        # for layer in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
        #     torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))

        # activation function
        if self.siren:
            self.act_fun = torch.sin
            self.siren_init()
        else:
            if self.act == 'tanh':
                self.act_fun = torch.tanh
            elif self.act == 'softplus':
                self.act_fun = torch.nn.functional.softplus
            else:
                # not suported error
                raise ValueError(f'activation {self.act} not supported')
        

        # for with_param version, pde parameter is not part of the network (but part of module)
        # for inverse problem, create embedding layer for pde parameter
        if self.with_param:
            # Create embedding layers for each parameter
            self.param_embeddings = nn.ModuleDict({
                name: nn.Linear(1, width, bias=False) for name, param in self.pde_params_dict.items()
            })
            
            # set requires_grad to False
            for embedding_weights in self.param_embeddings.parameters():
                # use positive initialization
                
                embedding_weights.requires_grad = self.train_embed
    
    def setup_modified_mlp(self):
        # see Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks 
        # equation 2.33 - 2.37
        input_dim = self.input_dim
        width = self.width
        output_dim = self.output_dim
        depth = self.depth

        if self.fourier:
            print('Using Random Fourier Features, not trainable')
            self.fflayer = nn.Linear(input_dim, width)
            set_module_require_grad(self.fflayer, self.rff_trainable)
            self.input_layer = lora.Linear(width, width, r = self.rank)
            self.input_layer_u = lora.Linear(width, width, r = self.rank)
            self.input_layer_v = lora.Linear(width, width, r = self.rank)
        else:
            self.input_layer = lora.Linear(input_dim, width, r = self.rank)
            self.input_layer_u = lora.Linear(input_dim, width, r = self.rank)
            self.input_layer_v = lora.Linear(input_dim, width, r = self.rank)

        # depth = input + k hidden + output
        self.hidden_layers = nn.ModuleList([lora.Linear(width, width,r = self.rank) for _ in range(depth - 2)])
        self.output_layer = lora.Linear(width, output_dim, r = self.rank)
        
        # activation function
        if self.siren:
            self.act_fun = torch.sin
            self.siren_init()
        else:
            if self.act == 'tanh':
                self.act_fun = torch.tanh
            elif self.act == 'softplus':
                self.act_fun = torch.nn.functional.softplus
            else:
                # not suported error
                raise ValueError(f'activation {self.act} not supported')
        

        # for with_param version, pde parameter is not part of the network (but part of module)
        # for inverse problem, create embedding layer for pde parameter
        if self.with_param:
            # Create embedding layers for each parameter
            self.param_embeddings = nn.ModuleDict({
                name: nn.Linear(1, width, bias=False) for name, param in self.pde_params_dict.items()
            })
            
            # set requires_grad to False
            for embedding_weights in self.param_embeddings.parameters():
                # use positive initialization
                # embedding_weights.data.uniform_(0, 1)
                embedding_weights.requires_grad = self.train_embed

    def forward(self, x, pde_params_dict=None):
        if self.modifiedmlp:
            u = self.forward_modified(x, pde_params_dict)
        else:
            u = self.basic_forward(x, pde_params_dict)
        
        u = self.output_transform(x, u, pde_params_dict)
        return u
       
    def collect_trainable_param(self):
        '''setup trianable parameter'''
        # separate parameters for the neural net and the PDE parameters
        # neural net parameter exclude parameter embedding and fourier feature embedding layer
        # Initialize lists
        self.param_net = []
        self.param_lora = []  # This will hold only LoRA parameters

        list_input_layers = [self.input_layer]
        if self.modifiedmlp:
            list_input_layers += [self.input_layer_u, self.input_layer_v]

        for layer in list_input_layers + list(self.hidden_layers) + [self.output_layer]:
            # Check if this is a LoRA-enhanced linear layer
            if isinstance(layer, lora.Linear):
                # Separate LoRA and non-LoRA parameters by name
                for name, param in layer.named_parameters():
                    if 'lora' in name.lower():
                        self.param_lora.append(param)
                    else:
                        self.param_net.append(param)
            else:
                # Normal layer, add all parameters
                self.param_net.extend(list(layer.parameters()))

        # only train the low-rank parameters
        if self.rank > 0:
            # set all param_net to be not trainable
            for param in self.param_net:
                param.requires_grad = False
            self.param_net = self.param_lora
            

        if self.fourier and self.rff_trainable:
            self.param_net += list(self.fflayer.parameters())
        
        if self.with_param and self.train_embed:
            for name, param in self.param_embeddings.items():
                self.param_net += list(param.parameters())


        for k in self.trainable_param:
            if isinstance(self.all_params_dict[k], nn.Parameter):
                self.param_pde.append(self.all_params_dict[k])
            elif isinstance(self.all_params_dict[k], nn.Module):
                self.param_pde += list(self.all_params_dict[k].parameters())
            else:
                raise ValueError(f'Unknown type for parameter {k}')

        # For PINN, optimizer include all parameters
        # include untrainable parameters, so that the optimizer have the parameter in state_dict
        if not self.with_param:
            self.param_all = self.param_net + self.param_pde
        # for BILO, has parameter embedding
        else:
            # collection of trainable parameters
            self.param_pde_trainable = [param for param in self.param_pde if param.requires_grad]
        

    def output_transform(self, x, u, pde_params_dict=None):
        '''
        transform the output of the network
        '''
        return self.lambda_transform(x, u, pde_params_dict)
    
    def siren_init(self):
        '''
        initialize weights for siren
        '''
        with torch.no_grad():
            self.input_layer.weight.uniform_(-1 / self.input_layer.in_features, 1 / self.input_layer.in_features)
            for layer in self.hidden_layers:
                layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                             np.sqrt(6 / layer.in_features) / self.omega_0)

    def embedding_modified(self, x, pde_params_dict=None):
        '''
        '''
        xcoord = x
        self.get_param_expand(xcoord, pde_params_dict)
        
        if self.with_param :
            # go through each parameter and do embedding
            param_embedding = self.get_param_embedding(x, pde_params_dict)
        else:
            # for vanilla version, no parameter embedding
            param_embedding = 0

        # fourier feature embedding
        if self.fourier:
            x = self.fflayer(x) + param_embedding
            x = torch.sin(2 * torch.pi * self.sigma * x)

            h = self.input_layer(x) 
            u = self.input_layer_u(x)
            v = self.input_layer_v(x)
        
        else:
            h = self.input_layer(x) + param_embedding
            u = self.input_layer_u(x) + param_embedding
            v = self.input_layer_v(x) + param_embedding
        
        return h, u, v

    def get_param_expand(self, x, pde_params_dict=None):
        '''
        get the embedding of the pde parameter
        n is the size of the input x
        '''
        for name, param in pde_params_dict.items():
            # expand the parameter to the same size as as n
            if isinstance(param, torch.Tensor):
                n = x.shape[0]
                self.params_expand[name] = param.expand(n, 1)
            elif isinstance(param, nn.Module):
                self.params_expand[name] = param(x)
            elif isinstance(param, torch.Tensor) and param.shape[0] == x.shape[0]:
                # this is z = f(x)
                self.params_expand[name] = param
            else:
                raise ValueError(f'Unknown type for parameter {name}')

    def get_param_embedding(self, x, pde_params_dict=None):
        '''
        get the embedding of the pde parameter
        n is the size of the input x
        '''
        z = 0        
        for name, param in pde_params_dict.items():
            param_embedding = self.param_embeddings[name](self.params_expand[name])
            z = z + param_embedding
        return z

    def embedding(self, x, pde_params_dict=None):
        '''
        No fourier feature embedding:
            then y = Wx+b = input_layer(x)
        if fourier feature embedding:
            then z = sin(2*pi* (Wx+b))
            then y = Wz+b = input_layer(z)
        
        if with_param, then Wy+b + W'p+b' (pde parameter embedding)
        This is the same as concat [x, pde_param] and then do a linear layer
        otherwise, then Wy+b

        '''
        
        xcoord = x
        # fourier feature embedding
        if self.fourier:
            x = torch.sin(2 * torch.pi * self.sigma * self.fflayer(x))
        x = self.input_layer(x)
        
        self.get_param_expand(xcoord, pde_params_dict)

        if self.with_param :
            # go through each parameter and do embedding
            param_embedding = self.get_param_embedding(xcoord, pde_params_dict)
            x += param_embedding
            
        return x
    
    def basic_forward(self, x, pde_params_dict=None):
        
        X = self.embedding(x, pde_params_dict)
        Xtmp = self.act_fun(X)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_output = hidden_layer(Xtmp)
            if self.use_resnet:
                hidden_output += Xtmp  # ResNet connection
            hidden_output = self.act_fun(hidden_output)
            Xtmp = hidden_output
        
        u = self.output_layer(Xtmp)    
        return u
    
    def forward_modified(self, x, pde_params_dict=None):
        x, u, v = self.embedding_modified(x, pde_params_dict)
        H = self.act_fun(x)
        U = self.act_fun(u)
        V = self.act_fun(v)
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            z = hidden_layer(H)
            Z = self.act_fun(z)
            H = (1 - Z) * U + Z * V
        u = self.output_layer(H)
        return u
    
    def reset_weights(self):
        '''
        Resetting model weights
        '''
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def freeze_layers_except(self, n):
        '''
        Freeze model, only the last n layers are trainable
        '''
        self.param_net = []
        all_layers = []  # To flatten all layers and sub-layers

        # Recursive function to flatten layers, skipping ParameterDict
        def flatten_layers(module):
            for child in module.children():
                if isinstance(child, torch.nn.ParameterDict):
                    print(f"Skipping ParameterDict: {child}")
                    continue
                elif isinstance(child, torch.nn.ModuleList):
                    # Handle layers inside ModuleList
                    flatten_layers(child)
                else:
                    all_layers.append(child)

        flatten_layers(self)  # Populate all_layers with valid layers

        total = len(all_layers)
        print(f'Total {total} layers, freeze {total - n} layers')

        # Freeze all but the last `n` layers
        for i, layer in enumerate(all_layers):
            if i < total - n:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                self.param_net += list(layer.parameters())
    
    def print_named_module(self):
        for name, layer in self.named_modules():
            print(name, layer)



class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)

class ParamFunction(nn.Module):
    '''represent unknown f(x) to be learned, diffusion field or initial condition'''
    def __init__(self, input_dim=1, output_dim=1, fdepth=4, fwidth=16, 
                 activation='tanh', output_activation='id', 
                 fsiren=False, 
                 ffourier=False,
                 output_transform=lambda x, u: u):
        super(ParamFunction, self).__init__()
        
        # represent single variable function

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fsiren = fsiren
        self.fourier = ffourier

        if activation == 'tanh':
            nn_activation = nn.Tanh
        elif activation == 'relu':
            nn_activation = nn.ReLU
        elif activation == 'sigmoid':
            nn_activation = nn.Sigmoid
        elif activation == 'id':
            nn_activation = nn.Identity
        else:
            raise ValueError(f'activation {activation} not supported')

        if self.fsiren:
            activation = Sin

        if output_activation == 'softplus':
            nn_output_activation = nn.Softplus
        elif output_activation == 'id':
            nn_output_activation = nn.Identity
        elif output_activation == 'relu':
            nn_output_activation = nn.ReLU
        elif output_activation == 'sigmoid':
            nn_output_activation = nn.Sigmoid
        else:
            raise ValueError(f'output activation {output_activation} not supported')
        
        # random fourier feature
        if self.fourier:
            self.fflayer = nn.Linear(input_dim, fwidth)
            self.fflayer.requires_grad = False
            input_layer = nn.Linear(fwidth, fwidth)
        else:
            input_layer = nn.Linear(input_dim, fwidth)

        # Create the layers of the neural network
        layers = []
        if fdepth == 1:
            # Only one layer followed by output_activation if fdepth is 1
            layers.append(input_layer)
            layers.append(nn_output_activation())
        else:
            # input layer
            layers.append(nn.Linear(input_dim, fwidth))
            layers.append(nn_activation())

            # hidden layers (fdepth - 2 because we already have input and output layers)
            for _ in range(fdepth - 2):
                layers.append(nn.Linear(fwidth, fwidth))
                layers.append(nn_activation())

            # output layer
            layers.append(nn.Linear(fwidth, output_dim))
            layers.append(nn_output_activation())

        # Store the layers as a sequential module
        self.layers = nn.Sequential(*layers)

        # Store the output transformation function
        self.output_transform = output_transform

        # Initialize the weights
        if self.fsiren:
            self.siren_init()


    def siren_init(self):
        '''
        initialize weights for siren
        '''
        self.omega_0 = 30
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    if i==0:
                        layer.weight.uniform_(-1 / layer.in_features, 1 / layer.in_features)
                    else:
                        layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                                    np.sqrt(6 / layer.in_features) / self.omega_0)

    def forward(self, x):
        # Define the forward pass
        u = self.layers(x)
        return self.output_transform(x, u)



class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, trainable=False):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=trainable)
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features), requires_grad=trainable)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        phi = torch.exp(-distances.pow(2))
        return phi


# simple test of the network
# creat a network, compute residual, compute loss, no training
# if __name__ == "__main__":

    
#     optobj = Options()
#     optobj.parse_args(*sys.argv[1:])
    

#     device = set_device('cuda')
#     set_seed(0)
    
#     prob = create_pde_problem(**optobj.opts['pde_opts'])
#     prob.print_info()

#     optobj.opts['nn_opts']['input_dim'] = prob.input_dim
#     optobj.opts['nn_opts']['output_dim'] = prob.output_dim

#     net = DenseNet(**optobj.opts['nn_opts'],
#                 output_transform=prob.output_transform, 
#                 params_dict=prob.param).to(device)
    
#     dataset = {}
#     x = torch.linspace(0, 1, 20).view(-1, 1).to(device)
#     x.requires_grad_(True)
#     y = prob.u_exact(x, prob.param)
#     res, u_pred = prob.residual(net, x, net.all_params_dict)

#     jac  = prob.compute_jacobian(net, x, net.all_params_dict)


#     # print 2 norm of res
#     print('res = ',torch.norm(res))
#     print('jac = ',torch.norm(jac)) 
    

    