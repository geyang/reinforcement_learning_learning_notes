import gym
import numpy as np
import time


def run_episode(env, parameters, sleep=0):
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


MAX_STEPS = 200
DEBUG = False

ϵ = 20

env = gym.envs.make('Pendulum-v0')
env._max_episode_steps = MAX_STEPS

best_param = [0.18439764, - 3.66344595, - 4.933473]
best_reward = -1500
for _ in range(5000):
    parameters = (np.random.rand(3) - 0.5) * ϵ
    reward = run_episode(env, parameters, sleep=0.016)

    if reward > best_reward:
        best_param = parameters
        best_reward = reward
        print("best reward so far {reward}".format(reward=reward))

    if _ % 5000 == 0:
        print("{}% exploration finished".format(_ // 500))

print('random policy is finished.')

DEBUG = True
rewards = []
for _ in range(20):
    MAX_STEPS = 2000
    reward = run_episode(env, best_param, sleep=0.016)
    rewards.append(reward)
    print(_, reward, best_param)
