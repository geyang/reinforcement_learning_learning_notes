import gym
import numpy as np
import time


def run_episode(env, parameters, sleep=0):
    observation = env.reset()
    env._max_episode_steps = MAX_STEPS
    totalreward = 0
    for _ in range(MAX_STEPS):
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


MAX_STEPS = 200
DEBUG = False
rewards = []

ϵ = 1e-1

env = gym.envs.make('CartPole-v0')

good_params = []
for _ in range(10000):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env, parameters, 0.01)
    print(_, reward)
    if reward >= 200:
        good_params.append(parameters)

print('best_params: ', np.mean(good_params, axis=0))

DEBUG = True
MAX_STEPS = 2000
for i in range(10):
    reward = run_episode(env, np.mean(good_params, axis=0), 0.01)
    print("test of final parameter", i, reward)
    time.sleep(1)
