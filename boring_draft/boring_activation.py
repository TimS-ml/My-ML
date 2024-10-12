import torch
from torch import nn
import numpy as np

# From t3-1
act_fn_by_name = {}

class Sigmoid(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
    '''
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

act_fn_by_name['sigmoid'] = Sigmoid


class StableSigmoid(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
    '''
    def forward(self, x):
        return 4 / (1 + torch.exp(-x)) - 2

act_fn_by_name['stablesigmoid'] = StableSigmoid


class Softmax(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Softmax.html#torch.nn.Softmax
    '''
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x
        # return F.softmax(x, self.dim, _stacklevel=5)

# act_fn_by_name['softmax'] = Softmax  # let's skip this rn :(


class Tanh(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Tanh.html#torch.nn.Tanh
    '''
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

act_fn_by_name['tanh'] = Tanh


class Tanhshrink(nn.Module):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html
    '''
    def forward(self, x):
        return x - (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

act_fn_by_name['tanhshrink'] = Tanhshrink


class Softplus(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Softplus.html#torch.nn.Softplus
    '''
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        softplus =  1/self.beta * torch.log(1 + torch.exp(self.beta * x))
        return torch.where(x > self.threshold, x, softplus)

act_fn_by_name['softplus'] = Softplus


class Mish(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.Mish.html#torch.nn.Mish
    '''
    def forward(self, x):
        soft_plus = torch.nn.functional.softplus(x)
        tanh_x = torch.tanh(soft_plus)
        return x * tanh_x

act_fn_by_name['mish'] = Mish

class ReLU(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.ReLU.html#torch.nn.ReLU
    '''
    def forward(self, x):
        # return torch.max(0, x)
        return x * (x > 0).float()

act_fn_by_name['relu'] = ReLU


class LeakyReLU(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU
    '''
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.neg_slop = negative_slope
        
    def forward(self, x):
        return torch.where(x > 0, x, self.neg_slop * x)

act_fn_by_name['leakyrelu'] = LeakyReLU


class PReLU(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.PReLU.html#torch.nn.PReLU
    '''
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x):
        return torch.where(x > 0, x, self.weight * x)

# act_fn_by_name['prelu'] = PReLU  # also skip this :(


class ELU(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.ELU.html#torch.nn.ELU
    '''
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

act_fn_by_name['elu'] = ELU
    

class SiLU(nn.Module):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

act_fn_by_name['silu'] = SiLU


class GELU(nn.Module):
    '''
    https://pytorch.org/docs/master/generated/torch.nn.GELU.html#torch.nn.GELU
    '''
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                np.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )

act_fn_by_name['gelu'] = GELU
