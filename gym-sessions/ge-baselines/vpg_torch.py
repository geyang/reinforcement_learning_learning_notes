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
            self.mu_fc = nn.Linear(input_size, ac_size)
            self.stddev_fc = nn.Linear(input_size, ac_size)
        elif action_type == 'linear':
            # todo: make action configurable
            self.mu_fc = nn.Linear(input_size, ac_size)

        # define the optimizer locally ^_^
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def forward(self, x):
        mu, stddev = [x] * 2
        assert len(mu.size()) == 2, 'mu should be a 2D tensor with batch_index first.'
        # mu = F.softmax(self.mu_fc(mu))
        mu = self.mu_fc(mu)
        if self.action_type == "gaussian":
            # todo: need tested
            stddev = F.softmax(self.stddev_fc(stddev))
            return mu, stddev
        elif self.action_type == "linear":
            return mu, None


class ValueNetwork(nn.Module):
    def __init__(self, ob_size):
        super(ValueNetwork, self).__init__()
        self.input_size = ob_size

        self.fc1 = nn.Linear(ob_size, 1)
        # self.layers = nn.ParameterList(Parameter1, Parameter2, ...)

        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def forward(self, x):
        x = self.fc1(x)
        x = x.squeeze(dim=-1)
        return x


class VPG(nn.Module):
    def __init__(self, ob_size, ac_size, action_type, **kwargs):
        super(VPG, self).__init__()

        self.ac_size = ac_size
        self.action_type = action_type

        # value function shouldn't really be part of VPG module, because
        # the gradient is not automatically propagated.
        self.value_fn = ValueNetwork(ob_size)
        self.action = Action(ob_size, ac_size, action_type)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    def discrete_sampling(mu, flag=None):
        """Only supports bernoulli distribution. Does not support categorical distribution."""
        # Use an energy calculation for the probability
        probs = F.softmax(mu)
        if flag == 'probs_only':
            return probs
        acts = torch.multinomial(probs, 1)
        return torch.squeeze(acts, dim=1), probs

    @staticmethod
    def gaussian_sampling(mu, log_stddev, flag=None):
        zeros = torch.zeros(mu.size())
        ones = torch.ones(mu.size())
        epsilon = Variable(torch.normal(zeros, ones))
        stddev = torch.exp(log_stddev)
        # sample
        action = torch.mul(epsilon, stddev) + mu
        # compute sample probability
        act_probs = epsilon / (2. * np.exp(1. * 2.))
        return action, act_probs

    def act(self, obs):
        obs = h.varify(obs, volatile=True)  # use as inference mode.
        mus, stddev = self.action(obs)
        if self.action_type == 'linear':
            acts, act_probs = self.discrete_sampling(mus)
        elif self.action_type == 'gaussian':
            acts, act_probs = self.gaussian_sampling(mus, stddev)
        else:
            raise Exception('action_type {} is not supported'.format(self.action_type))
        return acts, act_probs

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

    def learn(self, obs, acts, vals, lr, normalize=False, use_baseline=False):
        obs = h.varify(obs)
        if self.action_type == 'linear':
            acts = h.tensorify(acts, type='int')
        elif self.action_type == 'gaussian':
            acts = h.tensorify(acts, type='float')
        if use_baseline:
            vals = h.varify(vals) - self.value_fn(obs)
        else:
            vals = h.varify(vals)
        if normalize:
            vals = (vals - torch.mean(vals, dim=0).expand_as(vals)) / (torch.std(vals, dim=0) + 1e-8).expand_as(vals)
        self.action.set_lr(lr)
        self.action.optimizer.zero_grad()
        mu, stddev = self.action(obs)
        # todo: problem: different parts of the comp graph got complicated.
        if self.action_type == 'linear':
            act_probs = self.discrete_sampling(mu, 'probs_only')
        elif self.action_type == "gaussian":
            act_probs = self.gaussian_sampling(mu, stddev, 'probs_only')
        # discrete action space only
        assert len(acts.size()) == 1, "acts should be a batch of scalars"
        assert len(act_probs.size()) == 2, "act_probs should be a batch of 1d tensor"
        act_oh = h.one_hot(acts, n=self.ac_size)
        # eligibility is the derivative of log_probability
        normalized_act_prob = act_probs.mul(act_oh).sum(dim=-1)
        log_probability = torch.log(normalized_act_prob) - 1e-6
        surrogate_loss = - torch.sum(vals * log_probability)
        # M.red(surrogate_loss.data.numpy()[0])
        surrogate_loss.backward()
        self.action.optimizer.step()

    def __enter__(self):
        return self

    def __exit__(self, *err):
        pass

    @staticmethod
    def discrete_eligibility_fn(actions_ph, action_prob, scope_name="eligibility"):
        with tf.name_scope(scope_name):
            oh = tf.transpose(tf.one_hot(actions_ph, 2, axis=0))
            action_likelihood = tf.reduce_sum(action_prob * oh, axis=1)
            eligibility = tf.log(action_likelihood) - 1e-6
        return eligibility

    @staticmethod
    def eligibility_fn(actions_ph, mu, log_stddev=None, scope_name="eligibility"):
        with tf.name_scope(scope_name):
            elgb = - 0.5 * (np.log(2) + np.log(np.pi)) - log_stddev \
                   - (actions_ph - mu) ** 2 / (2 * tf.exp(log_stddev * 2)) \
                   - 1e-6
        return elgb

    @staticmethod
    def surrogate_fn(elgb, advantage, scope_name="surrogate_loss"):
        with tf.name_scope(scope_name):
            surrogate_loss = tf.multiply(elgb, advantage)
            loss = - tf.reduce_mean(surrogate_loss)
            return loss

    @staticmethod
    def value_function(state_ph, rewards_ph, value_lr, scope_name="value_function"):
        '''predicts the value of the current state, not sum of discounted future rewards.'''
        with tf.name_scope(scope_name):
            x = state_ph
            x = tf_helpers.dense(128, x, scope_name="layer_1", nonlin='relu')
            # x = tf_helpers.dense(128, x, scope_name="layer_2", nonlin='relu')
            # x = tf_helpers.dense(128, x, scope_name="layer_3", nonlin='relu')
            x = tf_helpers.dense(1, x, scope_name="layer_4")

            with tf.name_scope('optimizer'):
                error = rewards_ph - x
                loss = error ** 2
                optimizer = tf.train.AdamOptimizer(value_lr).minimize(loss)

            return x, optimizer
