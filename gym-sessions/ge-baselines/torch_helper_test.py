import torch_helpers as h


def test_one_hot():
    acts = h.varify([[1], [2]], dtype='int')
    n = 3
    oh = h.one_hot(acts, n)
    print(oh)


def test_mask():
    probs = h.varify([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], requires_grad=True)
    acts = h.varify([1, 2], dtype='int')
    oh = h.sample_probs(probs, acts)
    oh.sum().backward()
    print(oh)
