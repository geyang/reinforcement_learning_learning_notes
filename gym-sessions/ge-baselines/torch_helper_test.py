import torch_helpers as h
import numpy as np


def test_one_hot():
    acts = h.varify([1, 2], dtype='int')
    n = 3
    oh = h.one_hot(acts, n)
    assert (oh.data.numpy() == np.array([[0, 1, 0], [0, 0, 1]])).all(), "one_hot gives incorrect output {}".format(oh)



def test_mask():
    probs = h.varify([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], requires_grad=True)
    acts = h.varify([1, 2], dtype='int')
    sampled_probs = h.sample_probs(probs, acts)
    sampled_probs.sum().backward()
    dp = probs.grad.data.numpy()
    assert dp[0, 1] is not None and dp[1, 2] is not None, 'key entries of probs grad should be non-zero'
