from datetime import datetime
from params_proto import cli_parse

DATA_ROOT = "/tmp/ge_dqn/"


@cli_parse
class G:
    env_name = "MountainCar-v0"
    seed = 100
    # network parameters
    stochastic_action = True
    conv_params = None
    value_params = 64,
    use_layer_norm = False
    buffer_size = 50000
    replay_batch_size = 32
    prioritized_replay = False
    param_noise: "NotImplemented yet" = False
    alpha = 0.6
    beta_start = 0.4
    beta_end = 1.0
    prioritized_replay_eps = 1e-6
    grad_norm_clipping = 10  # reference uses 10.
    double_q: "flag for using the double Q architecture" = False
    use_dueling = False
    exploration_fraction: "fraction after which exploration stays at final_eps" = 0.1
    final_eps = 0.1
    # training parameters
    n_timesteps: "number of epochs to run (not epochs since only train on interval)" = int(1e5)
    learning_rate: "for the optimizer" = 0.001
    gamma: "discount rate" = 1.00
    learning_start: "num of timesteps before learning starts" = 1000
    learn_interval: "Interval for learning from the replay buffer" = 1
    target_network_update_interval: "update interval for the target network" = 500


@cli_parse
class RUN:
    num_cpu = 16
    log_directory = DATA_ROOT + "{time:%Y-%m-%d}/{time:%H%M%S.%f}-{G.env_name}" \
                                "-prioritized({G.prioritized_replay})-duel({G.use_dueling})" \
        .format(time=datetime.now(), prefix="debug", G=G)
    checkpoint = "checkpoint.cp"
    log_file = "output.log"


@cli_parse
class Reporting:
    checkpoint_interval = 10000
    reward_average = 100
    print_interval = 1
