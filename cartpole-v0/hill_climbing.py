import gym
import numpy as np
from scipy import ndimage
import time
import matplotlib.pyplot as plt

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


env = gym.envs.make('CartPole-v0')
DEBUG = True


def run_episode(env, parameters, sleep=0):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, _ = env.step(action)
        if DEBUG:
            env.render()
            if sleep:
                time.sleep(sleep)
        totalreward += reward
        if done:
            break
    return totalreward


DEBUG = False
rewards = []

ϵ = 3
bestparams = np.zeros(4)
bestreward = 0
for _ in range(10000):
    parameters = bestparams + (np.random.rand(4) - 0.5) * ϵ
    reward = run_episode(env, parameters, sleep=0.1)
    rewards.append(reward)
    bestparams = bestparams + 0.005 * (parameters - bestparams) * (reward - bestreward)
    print(_, rewards[-1])
    if len(rewards) > 20 and np.mean(rewards[-20:]) == 200:
        print('finished after #{} episodes'.format(_))
        break
