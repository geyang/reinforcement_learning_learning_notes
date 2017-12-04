from copy import deepcopy

import numpy


def contextify(env):
    type(env).__enter__ = lambda s: s
    type(env).__exit__ = lambda s, *args: s.close()
    return env


def monitor(env):
    episode_rewards = []
    _step = env.step

    def step(action):
        s, rew, done, info = _step(action)
        episode_rewards.append(rew)

        if not done:
            return s, rew, done, info

        episode_info = dict(
            total_reward=sum(episode_rewards),
            average_reward=numpy.mean(episode_rewards),
            timesteps=len(episode_rewards)
        )
        episode_rewards.clear()

        if type(info) is list:
            info = deepcopy(info) + [episode_info]
        elif type(info) is tuple:
            info = tuple(*deepcopy(info), *episode_info)
        elif hasattr(info, 'update'):
            info = deepcopy(info)
            info.update(**episode_info)
        return s, rew, done, info

    env.step = step
    return env
