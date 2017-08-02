import gym
import numpy as np
import time


def run_episode(env, parameters, sleep=0):
    env._max_episode_steps = MAX_STEPS
    observation = env.reset()
    totalreward = 0
    for _ in range(MAX_STEPS):
        action = np.matmul(parameters, observation)
        observation, reward, done, _ = env.step([action])
        if DEBUG:
            env.render()
            if sleep:
                time.sleep(sleep)
        totalreward += reward
        if done:
            break
    return totalreward


ϵ = 20

env = gym.envs.make('Pendulum-v0')

# the interesting constraint of this environment is that the force limit is +-2.0,
# which is not enough to hold the pole without using some parametric amplification.
# As a result, a simple linear response solution wouldn't work.
best_param = np.array([-10, 0, -0.1]) * 2

DEBUG = True
rewards = []
for _ in range(20):
    MAX_STEPS = 2000
    reward = run_episode(env, best_param, sleep=0.16)
    rewards.append(reward)
    print(_, reward, best_param)
