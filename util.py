import torch
import numpy as np
import json
import inspect
import types
import traceback


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, types.FunctionType):
            # for simple lambda function
            return inspect.getsource(obj)
        else:
            return super(MyEncoder, self).default(obj)

def savedict(dict, fpath):
    json.dump( dict, open( fpath, 'w' ), indent=4, cls=MyEncoder, sort_keys=True)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        
def mse(x, y = 0):
    # mean square error
    return torch.mean((x - y)**2)

def print_dict(d):
    print(json.dumps(d, indent=4,cls=MyEncoder,sort_keys=True))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_statistics(epoch, **losses):
    """
    Prints the epoch number and an arbitrary number of loss values.

    Args:
        epoch (int): The current epoch number.
        **losses (dict): A dictionary where the keys are descriptive names of the loss values,
                         and the values are the loss values themselves.
    """
    loss_strs = ', '.join(f'{name}: {value:.3e}' for name, value in losses.items())
    print(f'Epoch {epoch}, {loss_strs}')



# Function to set device priority: CUDA > MPS > CPU
def set_device(name = None):

    if name is not None:
        device = torch.device(name)

    # auto select device 
    # Check for CUDA availability
    else: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Check for MPS (Metal) availability
        elif is_metal_available():
            device = torch.device('metal')
        # Default to CPU
        else:
            device = torch.device('cpu')

    print(f'Using device: {device}')

    return device


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
# turn a nested dict to a single dict
# def flatten(d, parent_key=''):
#     items = []
#     for k, v in d.items():
#         new_key = k
#         if isinstance(v, dict):
#             items.extend(flatten(v, new_key).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

#
#  mlflow.log_params
#  Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
def flatten(nested_dict):
    """
    Flatten a nested dictionary into a single dictionary.
    For a nested dictionary, use key1.key2.....keyN as the concatenated key.

    Args:
    - nested_dict (dict): The nested dictionary to flatten.

    Returns:
    - dict: A flattened dictionary.
    """
    def flatten_helper(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_helper(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten_helper(nested_dict)



def generate_grf(x, std, l):
    """
    If l is small, it's iid Gaussian, mean 0, std 
    If l is large, it's a Gaussian random field with mean 0, cov a, and correlation length l.
    
    """
    a = std**2
    # Ensure x is a torch tensor, if not, convert it to a torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # Convert x to a 1D numpy array for processing with numpy
    x_numpy = x.view(-1).numpy()

    # Meshgrid for covariance matrix calculation
    x1, x2 = np.meshgrid(x_numpy, x_numpy, indexing='ij')

    if abs(l) > 1e-6:
        # grf with length scale l
        K = a * np.exp(-0.5 * ((x1 - x2)**2) / (l**2))
        grf_numpy = np.random.multivariate_normal(mean=np.zeros_like(x_numpy), cov=K)
    else:
        # iid Gaussian
        grf_numpy = np.random.normal(loc=0.0, scale=std, size=x_numpy.shape)

    # Convert grf_numpy back to a torch tensor, reshaping to match the original shape of x
    grf_torch = torch.tensor(grf_numpy, dtype=torch.float32).view(x.shape)

    return grf_torch


def to_double(x):
    """
    Converts a torch tensor to double
    if dictionary, convert each value to double
    """
    
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, torch.nn.ParameterDict) or isinstance(x, dict):
        return {key: to_double(value) for key, value in x.items()}

    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, str):
        # if native python type, return as is
        return x
        
    else:
        raise ValueError(f'Unknown type: {type(x)}')
    
        
def add_noise(dataset, noise_opts, x_name='X_dat_train', u_name='u_dat_train'):
    '''
    add noise to each coordinate of u_dat_train
    For now, assuming x is 1-dim, u is d-dim. 
    For ODE problem, this means the noise is correlated in time (if add length scale)
    '''
    dim = dataset[u_name].shape[1]
    x = dataset[x_name] # (N,1)
    noise = torch.zeros_like(dataset[u_name])
    for i in range(dim):
        tmp = generate_grf(x, noise_opts['std'], noise_opts['length_scale'])
        noise[:,i] = tmp.squeeze()

    dataset['noise'] = noise
    dataset[u_name] = dataset[u_name] + dataset['noise']
    
    return dataset

def get_mem_stats(device=None):
    ''' 
    Get cuda memory usage for the current process, return a dictionary of memory alloc, peak, reserved in MB
    
    Args:
        device: torch.device or int, optional. If None, uses current device.
    '''
    mem = {}
    if torch.cuda.is_available():
        # Specify device explicitly to avoid confusion
        if device is None:
            device = torch.cuda.current_device()
        
        mem['mem_alloc'] = torch.cuda.memory_allocated(device) / 1024**2
        mem['mem_alloc_max'] = torch.cuda.max_memory_allocated(device) / 1024**2
        mem['mem_reserved'] = torch.cuda.memory_reserved(device) / 1024**2
        mem['mem_reserved_max'] = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        mem['mem_alloc'] = 0
        mem['mem_alloc_max'] = 0
        mem['mem_reserved'] = 0
        mem['mem_reserved_max'] = 0

    return mem

def griddata_subsample(gt, gx, u, Nt, Nx):
    '''downsample grid data'''
    nt, nx = gt.shape
    tidx = np.linspace(0, nt-1, Nt, dtype=int)
    xidx = np.linspace(0, nx-1, Nx, dtype=int)

    su = u[np.ix_(tidx, xidx)]
    sgt = gt[np.ix_(tidx, xidx)]
    sgx = gx[np.ix_(tidx, xidx)]

    return sgt, sgx, su

def uniform_subsample_with_endpoint(x, N):
    '''uniformly subsample x, with the first and last point included'''
    # assert first dimension is larger than N
    assert x.shape[0] >= N
    n = x.shape[0]
    idx = np.linspace(0, n-1, N, dtype=int)
    if idx[0] != 0:
        idx[0] = 0
    if idx[-1] != n-1:
        idx[-1] = n-1
    return x[idx]


def cart_to_pol(X):
    """
    Converts spatial coordinates in X to polar (2D) or spherical (3D) coordinates.

    Parameters:
    - X (torch.Tensor): Input tensor of shape (n, 1 + xdim), where the first column is time,
      and the remaining columns are spatial coordinates (xdim = 2 or 3).

    Returns:
    - torch.Tensor: Output tensor of shape (n, 1 + xdim), containing time, radius, and angles.
    """
    time = X[:, 0]
    coords = X[:, 1:]
    xdim = coords.shape[1]

    eps = 1e-6
    
    if xdim == 2:
        # 2D case: Convert to polar coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        r = torch.sqrt(x**2 + y**2 + eps)
        # r = torch.exp(- (x**2 + y**2))
        theta = torch.atan2(y, x + eps)  # Angle in radians
        # Stack the results into a tensor
        result = torch.stack((time, r, theta), dim=1)
    elif xdim == 3:
        # 3D case: Convert to spherical coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero
        theta = torch.acos(z / (r + eps))  # Colatitude angle
        phi = torch.atan2(y, x)            # Azimuthal angle
        # Stack the results into a tensor
        result = torch.stack((time, r, theta, phi), dim=1)
    else:
        raise ValueError("xdim must be 2 or 3")

    return result


# decorator for error logging
def error_logging_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in function '{func.__name__}': {e}")
            print(tb)
    return wrapper



