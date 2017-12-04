from datetime import datetime
import tensorflow as tf

import config
from train import train


def train_breakout(seed=42):
    time = datetime.now()
    config.G.env_name = "BreakoutNoFrameskip-v4"
    config.G.seed = seed
    config.G.conv_params = (32, 8, 4), (64, 4, 2), (64, 3, 1)
    config.G.value_params = 256,
    config.G.use_dueling = True
    config.G.use_layer_norm = True
    config.G.buffer_size = 10000
    config.G.n_timesteps = int(20e6)
    config.G.learning_start = 10000
    config.G.exploration_fraction = 0.1
    config.G.final_eps = 0.01
    config.G.gamma = 0.99
    config.G.learning_rate = 1e-4
    config.G.learn_interval = 4
    config.G.prioritized_replay = True
    config.G.replay_batch_size = 32
    config.G.double_q = True
    config.G.param_noise = False
    config.G.target_network_update_interval = 1000

    config.RUN.log_directory = config.DATA_ROOT + \
                               f"{time:%Y-%m-%d}/{time:%H%M%S.%f}" \
                               f"-{config.G.env_name}-seed({config.G.seed})" \
                               f"-prioritized_replay({config.G.prioritized_replay})" \
                               f"-param_noise({config.G.param_noise})"

    config.Reporting.print_interval = 1
    config.RUN.num_cpu = 8
    train()
    tf.reset_default_graph()


if __name__ == "__main__":
    train_breakout(42)
    # from multiprocessing import Pool
    #
    # pool = Pool(processes=4)  # start 4 worker processes
    # pool.map(train_mountaincar, range(42, 46))
