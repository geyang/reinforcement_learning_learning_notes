import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch_helpers as h
from moleskin import Moleskin

M = Moleskin()


class Action(nn.Module):
    def __init__(self, input_size, ac_size, action_type):
        super(Action, self).__init__()
        self.input_size = input_size
        self.action_type = action_type

        if action_type == 'gaussian':
            self.mu_fc1 = nn.Linear(input_size, ac_size)
            self.mu_fc1.weight.data.normal_(0.0, 0.3)
            self.stddev = Variable(torch.ones(1, ac_size) * 0.01, requires_grad=False)
            # self.stddev_fc1 = nn.Linear(input_size, ac_size)
        elif action_type == 'linear':
            # todo: make action configurable
            self.mu_fc = nn.Linear(input_size, ac_size)

        # define the optimizer locally ^_^
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def forward(self, x):
        mu, stddev = [x] * 2
        assert len(mu.size()) == 2, 'mu should be a 2D tensor with batch_index first.'
        if self.action_type == "gaussian":
            mu = self.mu_fc1(mu)
            return mu, self.stddev.expand_as(mu)
        elif self.action_type == "linear":
            mu = self.mu_fc(mu)
            return mu, None


class ValueNetwork(nn.Module):
    def __init__(self, ob_size):
        super(ValueNetwork, self).__init__()
        self.input_size = ob_size

        self.fc1 = nn.Linear(ob_size, 28)
        self.fc2 = nn.Linear(28, 1)
        # self.layers = nn.ParameterList(Parameter1, Parameter2, ...)

        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(dim=-1)
        return x


class VPG(nn.Module):
    def __init__(self, ob_size, ac_size, action_type, **kwargs):
        super(VPG, self).__init__()

        self.input_size = ob_size
        self.ac_size = ac_size
        self.action_type = action_type

        # value function shouldn't really be part of VPG module, because
        # the gradient is not automatically propagated.
        self.value_fn = ValueNetwork(ob_size)
        self.action = Action(ob_size, ac_size, action_type)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.surrogate_loss = 0

    @staticmethod
    def discrete_sampling(mu, sampled_acts=None):
        """Only supports bernoulli distribution. Does not support categorical distribution."""
        # Use an energy calculation for the probability
        probs = F.softmax(mu)
        if sampled_acts is not None:
            # enforce sampled_acts being a variable
            assert isinstance(sampled_acts, Variable), "acts should be a Variable"
            sampled_acts.requires_grad = False
            assert len(sampled_acts.size()) == 1, "acts should be a batch of scalars"
            assert len(probs.size()) == 2, "act_probs should be a batch of 1d tensor"
            sampled_log_probs = torch.log(h.sample_probs(probs, sampled_acts))
            return sampled_log_probs
        else:  # sample
            draws = torch.multinomial(probs, num_samples=1)
            sampled_acts = torch.squeeze(draws, dim=-1)
            return sampled_acts

    @staticmethod
    def gaussian_sampling(mu, stddev, sampled_acts: Variable = None):
        if sampled_acts is not None:
            # use externally fed samples, enables REINFORCE with samples generated from a different policy
            # enforce sampled_acts being a variable
            assert isinstance(sampled_acts, Variable), "acts should be a Variable"
            sampled_acts.requires_grad = False
            # compute sample probability
            sampled_log_probs = - float(np.log(2 * np.pi) / 2) \
                                - torch.log(stddev) \
                                - (sampled_acts - mu) ** 2 / (2 * stddev ** 2)
            return sampled_log_probs
        else:
            zeros = torch.zeros(mu.size())
            ones = torch.ones(mu.size())
            epsilon = Variable(torch.normal(zeros, ones))
            # sample
            sampled_action = torch.mul(epsilon, stddev) + mu
            return sampled_action

    def act(self, obs):
        obs = h.varify(obs, volatile=True)  # use as inference mode.
        mus, stddev = self.action(obs)
        if self.action_type == 'linear':
            acts = self.discrete_sampling(mus)
        elif self.action_type == 'gaussian':
            acts = self.gaussian_sampling(mus, stddev)
        else:
            raise Exception('action_type {} is not supported'.format(self.action_type))
        return acts

    def learn_value(self, obs, vals, lr):
        # b/c the value_fn is trained in a supervised fashion, we can do the forward/recompute each time.
        vals = h.varify(vals)
        obs = h.varify(obs)
        self.value_fn.set_lr(lr)
        self.value_fn.zero_grad()
        val_preds = self.value_fn(obs)
        loss = self.value_fn.criterion(val_preds, vals)
        loss.backward()
        self.value_fn.optimizer.step()

    def reinforce(self, obs, acts, vs):
        """
        :param obs: Size(batch_n, steps, ob_size)
        :param acts: Size(batch_n, steps, ac_size)
        :param vs: Size(batch_n, steps)
        :param normalize: bool
        :param use_baseline: bool
        :return: None
        """
        obs = h.varify(obs)  # .view(-1, self.input_size)
        # todo: support higher dimensional value functions?
        vs = h.varify(vs)  # .view(-1)  # self.value_fn(obs)
        mu, stddev = self.action(obs)
        if self.action_type == 'linear':
            acts = h.varify(acts, dtype='int')
            sampled_log_probs = self.discrete_sampling(mu, sampled_acts=acts)
        elif self.action_type == "gaussian":
            acts = h.varify(acts, dtype='float')
            sampled_log_probs = self.gaussian_sampling(mu, stddev, sampled_acts=acts)
        # eligibility is the derivative of log_probability
        self.surrogate_loss -= torch.sum(vs * sampled_log_probs)

    def step(self, lr):
        """
        :param lr:
        :return:
        """
        self.surrogate_loss.backward()
        self.action.set_lr(lr)
        self.action.optimizer.step()
        self.surrogate_loss = 0
        self.action.optimizer.zero_grad()

    def __enter__(self):
        return self

    def __exit__(self, *err):
        pass
