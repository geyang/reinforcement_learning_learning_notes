import gym, os
import numpy as np
from moleskin import Moleskin

import vpg

MAX_STEPS = 200
TRAIN_BIAS = 4
ENV = "CartPole-v0"
RUN_ID = os.environ['RUN_ID']


class GymSession:
    def __init__(self, environment, algorithm):
        self.env = environment
        self.algo = algorithm

    def run_episode(self):
        states = []
        actions = []
        rewards = []
        preds = []
        ob = self.env.reset()
        states.append(ob)
        for step in range(200):
            acs, vpreds = self.algo.act([ob])
            ob, r, done, _ = self.env.step(acs[0][0])
            states.append(ob)
            rewards.append([r])
            actions.append(acs[0])
            preds.append(vpreds[0])
            if done:
                break

        return states[:-1], actions, rewards

    def value_iteration(self, *args):
        self.algo.learn_value_function(*args)

    def policy_iteration(self, *args):
        self.algo.learn(*args)


m = Moleskin()
LR = 1e-5
if __name__ == "__main__":
    env = gym.make(ENV)
    # action space is discrete if no `shape` attribute
    if hasattr(env.action_space, 'shape'):
        ac_size = env.action_space.shape[0]
        is_discrete = False
    else:
        is_discrete = True
        ac_size = 1
    ob_size = env.observation_space.shape[0]
    with vpg.VPG(ob_size, ac_size, is_discrete, RUN_ID, ENV) as algo:
        sess = GymSession(env, algo)
        for ind_epoch in range(50):
            obs, acs, rs = sess.run_episode()
            algo.logs.scaler(np.sum(rs), ind_epoch)
            algo.logs.histogram(acs, ind_epoch, bins=10)
            rewards = np.expand_dims(list(range(len(rs)))[::-1], axis=1)
            for _ in range(1):
                sess.value_iteration(obs, rewards, acs, LR)
            for _ in range(1):
                sess.policy_iteration(obs, rewards, acs, LR)

            m.green(np.sum(rs), end="")
            # m.red('\tmean', np.mean(acs), end="")
            # m.yellow('\tvariance', np.mean((acs - np.mean(acs)) ** 2))
            print('\r', end='')
