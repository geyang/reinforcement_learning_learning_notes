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


def sample_probs(probs, index, dim=-1):
    """
    :param probs: Size(batch_n, feat_n)
    :param index: Size(batch_n, ), Variable(LongTensor)
    :param dim: integer from probs.size inclusive.
    :return: sampled_probability: Size(batch_n)

    NOTE: scatter does not propagate grad to source. Use mask_select instead.

    Is similar to `[var[row,col] for row, col in enumerate(index)]`.
    ```
    import torch
    import torch_extras
    setattr(torch, 'select_item', torch_extras.select_item)
    var = torch.range(0, 9).view(-1, 2)
    index = torch.LongTensor([0, 0, 0, 1, 1])
    torch.select_item(var, index)
    # [0, 2, 4, 7, 9]
    ```
    """

    assert probs.size()[:-1] == index.size(), 'index should be 1 less dimension than probs.'
    mask = one_hot(index, probs.size()[-1])
    mask = cast(mask, 'byte').detach()
    mask.requires_grad = False
    return torch.masked_select(probs, mask).view(*probs.size()[:-1])


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
    return zeros.scatter(-1, index.unsqueeze(-1).expand_as(zeros), ones)


TYPES = {
    "byte": "byte",
    torch.ByteTensor: "byte",
    "str": "char",
    torch.CharTensor: "char",
    "double": "double",
    torch.DoubleTensor: "double",
    "float": "float",
    torch.FloatTensor: "float",
    "int": "int",
    torch.IntTensor: "int",
    "long": "long",
    torch.LongTensor: "long",
    "short": "short",
    torch.ShortTensor: "short"
}


def cast(var, dtype):
    """ Cast a Tensor to the given type.
        ```
        import torch
        import torch_extras
        setattr(torch, 'cast', torch_extras.cast)
        input = torch.FloatTensor(1)
        target_type = type(torch.LongTensor(1))
        type(torch.cast(input, target_type))
        # <class 'torch.LongTensor'>
        ```
    """

    if dtype in TYPES:
        return getattr(var, TYPES[dtype])()
    else:
        raise ValueError("Not a Tensor type.")
