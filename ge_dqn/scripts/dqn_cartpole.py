from datetime import datetime

import config
from train import train

if __name__ == "__main__":
    time = datetime.now()
    config.G.env_name = "CartPole-v0"
    value_params = 64,
    config.G.final_eps = 0.02
    config.G.prioritized_replay = False
    config.G.use_dueling = False
    config.G.double_q = True
    config.G.use_layer_norm = True
    config.G.n_timesteps = 10000
    config.RUN.log_directory = config.DATA_ROOT + f"ge_dqn/{time:%Y-%m-%d}/{time:%H%M%S.%f}-{config.G.env_name}"
    train()
