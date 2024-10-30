import torch
from torch.autograd.functional import jacobian
import numpy as np

class BasicFunc():
    def __init__(self, negative=False):
        self.negative = negative
    
    def func(self, x):
        pass
    
    def __call__(self, x):
        output = self.func(x)
        sign = -1. if self.negative else 1.
        return sign * output.view(-1, 1)

class Linear(BasicFunc):
    def __init__(self, theta, negative=False):
        super().__init__(negative)
        self.theta = torch.from_numpy(theta)

    def func(self, x):
        label = torch.matmul(x.to(self.theta.dtype), self.theta.to(x.device)).float()
        return label    

class QuadraticOnLinear(Linear):
    def func(self, x):
        label = super().func(x)
        label = 10 - (label - 3) ** 2
        return label  

class NegNormPlus(BasicFunc):
    def __init__(self, b: np.array, negative=False):
        super().__init__(negative)
        self.b = torch.from_numpy(b)

    def func(self, x):
        output = 5. - 0.5 * torch.norm(x - self.b.view(1, -1).to(x.device), dim=1)
        return output

def BatchGrad(X, func):
    '''
    Get the gradients of the batch data torch.Tensor(Batch_size, input_dim)
    Return:
        Gradient: torch.Tensor[Batch_size, input_dim]
    '''
    gradient_bs = jacobian(func, X)
    gradient_bs = torch.diagonal(
        gradient_bs, offset=0, dim1=0, dim2=2
        ).permute(2, 0, 1).squeeze(1)
    return gradient_bs, func(X)

def BatchGradV2(X, func):
    '''
    An alternate implementation of BatchGrad (return the same results)

    Get the gradients of the batch data torch.Tensor(Batch_size, input_dim)
    Return:
        Gradient: torch.Tensor[Batch_size, input_dim]
    '''
    X.requires_grad = True
    value = func(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]
    X.requires_grad = False
    return gradient, value
