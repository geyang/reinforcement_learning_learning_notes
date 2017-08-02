import gym
import numpy as np
import time


def run_episode(env, parameters, sleep=0):
    observation = env.reset()
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

ϵ = 3

env = gym.envs.make('CartPole-v0')
env._max_episode_steps = MAX_STEPS

best_param = np.zeros(4)
best_reward = 0
for _ in range(100000):
    parameters = best_param + (np.random.rand(4) - 0.5) * ϵ
    reward = run_episode(env, parameters, sleep=0.016)
    rewards.append(reward)
    best_param = best_param + 0.001 * (parameters - best_param) * (reward - best_reward)
    print(_, rewards[-1])
    if len(rewards) > 20 and np.mean(rewards[-20:]) >= (MAX_STEPS - 1):
        print('finished after #{} episodes'.format(_))
        break
