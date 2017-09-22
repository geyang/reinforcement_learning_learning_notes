# Continuous cart pole using policy gradients (PG)
# Running this script does the trick!

import gym
import numpy as np
from gym import wrappers

# env = gym.make('InvertedPendulum-v1')
env = gym.make('Pendulum-v0')
# env = wrappers.Monitor(env, '/home/sid/ccp_pg', force=True)


def simulate(policy, steps, graphics=False):
    observation = env.reset()
    R = 0
    for i in range(steps):
        if graphics: env.render()
        a = policy(observation)

        observation, reward, done, info = env.step(a)
        R += reward
        if done:
            break
    return R


def grad_gaussian_policy(w, sigma, gamma):
    '''
    Computes gradient for a one dimensional continuous action sampled from a gaussian
    :param w: parameters w of the mean of the gaussian policy a = w^Ts
    :param sigma: variance of the policy
    :param gamma: discount factor for variance reduction
    :return: reward and the gradient
    '''
    R_tau = 0
    observation = env.reset()
    grad = 0
    d = 1
    for i in range(1000):
        a = w.dot(observation) + sigma * np.random.randn()

        # log norm(w^Ts, sigma^2) = log(k/sigma) - (a-w^Ts)/sigma^2
        # grad_w log norm(w^Ts, sigma^2) = (a-w^Ts).s/sigma**2
        # grad_sigma log norm(w^Ts, sigma^2) = -1/sigma + (a-w^Ts)^2/sigma^3
        # append 4x1 grad_w with grad_sigma
        grad += np.append(1. / sigma ** 2 * (a - w.dot(observation)) * observation,  ### 4x1
                          -1. / sigma + 1. / sigma ** 3 * (a - w.dot(observation)) ** 2)  ### 1x1
        observation, reward, done, info = env.step([a])

        R_tau += d * (reward)
        d *= gamma
        if done:
            break
    return R_tau, grad


def pg():
    sigma = 0.3  # noise standard deviation
    alpha = 0.05  # learning rate
    alpha_sigma = 0.01
    n = env.observation_space.shape[0]
    w = np.random.randn(n)  # initial guess
    epochs = 50000
    n_samples = 20
    b = 0
    gamma = 1
    max_R = -float('inf')
    if gamma < 1:
        T = 1. / (1 - gamma)
    else:
        T = float('inf')
    for i in range(epochs):

        if b < 10:
            w = np.random.randn(n)  # restart
        R = np.zeros(n_samples)
        grad = np.zeros((n + 1, n_samples))
        for j in range(n_samples):
            R[j], grad[:, j] = grad_gaussian_policy(w, sigma, gamma)

        b = np.mean(R)
        s = np.std(R)

        if s != 0:
            A = (R - b) / s  # advantage
        else:
            A = R

        if b > min(T, 950):  # if done..., we use T because by discounting rewards we can only at most get T
            return w
        if b > max_R:
            max_R = b
            best_w = w.copy()

        # some hardcoding for decreasing stepsize when we get higher rewards
        check_step = min(1, (50.0 / b))
        dparam = check_step * alpha * 1. / n_samples * grad.dot(A)
        w += dparam[:-1]

        if sigma + alpha_sigma * dparam[-1] > 0:
            sigma += alpha_sigma * dparam[-1]
        if i % 100 == 0:
            print('iteration', i, 'mean return', b)
    print('bets R', max_R)
    return best_w


w = pg()
print('Training is Done, now run evaluations')
r = 0
for i in range(100):
    r += simulate(lambda s: (w.dot(s)), 1000, True)

print('average_return over 100 trials:', r / 100.0)
