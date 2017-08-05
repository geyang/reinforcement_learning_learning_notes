import numpy as np
import torch
from torch.autograd import Variable

# debug helpers
def show_params(self: torch.nn.Module):
    for p in self.parameters():
        print(p)

def freeze(self: torch.nn.Module):
    for p in self.parameters():
        p.requires_grad = False


def thaw(self: torch.nn.Module):
    for p in self.parameters():
        p.requires_grad = True


def tensorify(d, type='float', **kwargs) -> torch._TensorBase:
    d = np.array(d)
    if type == 'float':
        tensor_type = torch.FloatTensor
    elif type == 'int':
        tensor_type = torch.LongTensor
    else:
        raise Exception('tensor type "{}" is not supported'.format(type))
    return tensor_type(d)


def varify(d, type='float', **kwargs) -> Variable:
    d = np.array(d)
    if type == 'float':
        tensor_type = torch.FloatTensor
    elif type == 'int':
        tensor_type = torch.LongTensor
    return Variable(tensor_type(d), **kwargs)


def one_hot(t: torch.Tensor, n: int, type: str = 'float', varify=True):
    size = list(t.size())
    shape = size + [n]
    if type == 'int':
        oh = torch.LongTensor(*shape).zero_()
    elif type == 'float':
        oh = torch.FloatTensor(*shape).zero_()
    else:
        raise Exception('Type {} is not supported.'.format(type))
    oh.scatter_(-1, t.unsqueeze(dim=len(size)), 1)
    if varify:
        return Variable(oh)
    return oh
