import torch
import torch_helpers as h
import gym, os
import numpy as np
from moleskin import Moleskin

from vpg_torch import VPG


class GymSession:
    def __init__(self, environment, algorithm):
        self.env = environment
        self.algo = algorithm

    def run_episode(self):
        states = []
        actions = []
        rewards = []
        ob = self.env.reset()
        states.append(ob)
        for step in range(env._max_episode_steps):
            acs = self.algo.act([ob])  # use batch as default
            action = acs.data.numpy()[0]
            ob, r, done, _ = self.env.step(action)  #
            states.append(ob)
            rewards.append(r)
            actions.append(action)
            if done:
                break

        return states[:-1], actions, rewards

    def value_iteration(self, *args, **kwargs):
        self.algo.learn_value(*args, **kwargs)

    def policy_iteration(self, *args, **kwargs):
        self.algo.learn(*args, **kwargs)

    @staticmethod
    def reward_to_value(rewards: list,
                        gamma: float = 1.0) -> list:  # note: it is debatable if list is a good design here.
        """return value from rewards"""
        assert 0 < gamma <= 1.0, 'gamma has to be between 0 and 1 (inclusive)'
        assert type(rewards[0]) in [float, np.float16, np.float32, np.float64], 'reward should be a list of floats'
        n = len(rewards)
        R = 0
        vals = []
        for i in range(1, n + 1):
            R = gamma * R + rewards[-i]
            vals.insert(0, R)
        return vals


if __name__ == "__main__":
    M = Moleskin()
    LR = 1e-5
    MAX_STEPS = 2000
    TRAIN_BIAS = 4
    ENV = "CartPole-v0"
    # ENV = "Pendulum-v0"
    RUN_ID = os.environ['RUN_ID']

    env = gym.make(ENV)
    env._max_episode_steps = MAX_STEPS
    # action space is discrete if no `shape` attribute
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        ac_type = 'linear'
        ac_size = env.action_space.n  # Discrete
    elif type(env.action_space) is gym.spaces.box.Box:
        ac_type = 'gaussian'
        ac_size = env.action_space.shape[0]  # Box
    else:
        raise Exception('Enviroment {} is unsupported'.format(env))

    ob_size = env.observation_space.shape[0]
    with VPG(ob_size, ac_size, ac_type, run_id=RUN_ID, env=ENV) as algo:
        sess = GymSession(env, algo)
        for ind_epoch in range(5000):
            obs, acts, rs = sess.run_episode()
            vals = sess.reward_to_value(rs, 0.95)
            # if ind_epoch > 500:
            # if ind_epoch % 5 == 0:
            sess.value_iteration(obs, vals, 5e-3)
            sess.policy_iteration(obs, acts, vals, 5e-1,
                                  normalize=True,
                                  use_baseline=True)
            # sess.policy_iteration(obs, acts, np.array(rs) * len(rs), 2e-2, use_baseline=False)

            M.white("#", ind_epoch, end="\t")
            M.print("action: ", end="\t")
            M.red(acts[0], end="\t")
            M.print("steps: ", end="\t")
            M.green(np.sum(rs), end="\t")
            M.print("action mean: ", end="\t")
            M.red("{:.2f}".format(np.mean(acts)), end="\n")
