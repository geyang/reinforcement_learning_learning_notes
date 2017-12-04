from datetime import datetime
import tensorflow as tf

import config
from train import train


def train_mountaincar(seed=42, b=False):
    time = datetime.now()
    config.G.env_name = "MountainCar-v0"
    config.G.seed = seed
    config.G.value_params = 64,
    config.G.use_layer_norm = True
    config.G.buffer_size = 50000
    config.G.n_timesteps = 500000
    config.G.learning_start = 1000
    config.G.exploration_fraction = 0.1
    config.G.final_eps = 0.1
    config.G.gamma = 1.0
    config.G.learning_rate = 0.001
    config.G.prioritized_replay = False
    config.G.replay_batch_size = 32
    config.G.double_q = b
    config.G.use_dueling = False
    config.G.param_noise = False

    config.RUN.log_directory = config.DATA_ROOT + \
                               f"ge_dqn/{time:%Y-%m-%d}/{time:%H%M%S.%f}" \
                               f"-{config.G.env_name}-seed({config.G.seed})" \
                               f"-prioritized_replay({config.G.prioritized_replay})" \
                               f"-double_q({config.G.double_q})" \
                               f"-param_noise({config.G.param_noise})"

    config.Reporting.print_interval = 10
    train()
    tf.reset_default_graph()


def fn(ps):
    train_mountaincar(*ps)


if __name__ == "__main__":
    from multiprocessing import Pool

    pool = Pool(processes=4)  # start 4 worker processes
    params = zip(list(range(49, 53)) * 2, [False] * 4 + [True] * 4)

    pool.map(fn, params)
