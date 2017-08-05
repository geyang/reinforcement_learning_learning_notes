import torch
import torch_helpers as h
from vpg_torch import VPG, ValueNetwork
from moleskin import Moleskin

m = Moleskin()


def test_act():
    obs = [[0.0, 0.0, 1.0, 1.0]]
    vpg = VPG(ob_size=4, ac_size=1, action_type='linear')
    acts, probs = vpg.act(obs)
    assert acts.data.numpy().shape == (1,)


def show_params(mo):
    for p in mo.parameters():
        print(p)


def test_value():
    print("""test the ValueNetwork""")
    value_fn = ValueNetwork(ob_size=4)
    value_fn.optimizer.param_groups[0]['lr'] = 5e-2
    # test against values larger than 1.
    target_val = h.varify([50.0])
    for i in range(1000):
        obs = h.varify([[0.0, 0.0, 1.0, 1.0]])
        value_fn.zero_grad()
        vals = value_fn(obs)
        loss = value_fn.criterion(vals, target_val)
        if i % 100 == 0:
            print(loss.data.numpy()[0])
        loss.backward()
        value_fn.optimizer.step()
    assert loss.data.numpy()[0] < 1e-1, 'loss should be very small (l < 0.1)'

# def test_learn_value():
#     vpg = VPG(ob_size=4, ac_size=1, ac_is_discrete=True)
#     obs = [[0, 0, 1, 1], [0.5, 0.5, 0.5, 1]]
#     rewards = [[1], [1]]
#     acts = [[1], [0]]
#     LR = 1e-3
#     vpg.learn_value(obs, rewards, acts, LR)

#
# def test_learn():
#     vpg = VPG(ob_size=4, ac_size=1, ac_is_discrete=True)
#     obs = [[0, 0, 1, 1]]
#     acs = vpg.learn(obs, acts, rs, lr)
#     print(acs)
