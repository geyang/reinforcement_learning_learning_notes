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


def varify(d, dtype='float', **kwargs) -> Variable:
    d = np.array(d)
    t = tensorify(d, dtype)
    return Variable(t, **kwargs)


def sample_probs(probs, sampled, dim=-1):
    """
    :param probs: Size(batch_n, feat_n)
    :param sampled: Size(batch_n, ), Variable(LongTensor)
    :param dim: integer from probs.size inclusive.
    :return: sampled_probability: Size(batch_n)

    Use scatter and allow the gradient to flow from output -> probs.
    """
    # we do not need to unsqueeze and expand input here.
    assert probs.size()[:-1] == sampled.size(), 'sampled should be 1 less dimension than probs.'
    zeros = Variable(torch.zeros(*probs.size()))
    return zeros.scatter(dim, sampled.unsqueeze(dim=-1).expand_as(probs), probs).sum(dim=-1)


# done: change type to Variable from Tensor
def one_hot(index: Variable, feat_n: int):
    """
    for one_hot masking of categorical distributions, use mask directly instead.
    :param index:
    :param feat_n:
    :return:
    """
    zeros = Variable(torch.FloatTensor(*index.size(), feat_n).zero_())
    ones = Variable(torch.ones(*zeros.size()))
    zeros.scatter(-1, index.unsqueeze(-1).expand_as(zeros), ones)
    return zeros
