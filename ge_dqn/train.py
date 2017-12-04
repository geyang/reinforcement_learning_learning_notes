import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
import gym
from baselines.common.atari_wrappers_deprecated import EpisodicLifeEnv, MaxAndSkipEnv, NoopResetEnv, FireResetEnv, \
    ProcessFrame84, FrameStack, ClippedRewardsWrapper, ScaledFloatFrame
from moleskin import moleskin as M

from config import G, Reporting, RUN
from logger import Logger, green, default, percent, MovingAverage, yellow, ms, sec
from monitor import monitor, contextify
from q_network import q_policy
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer


class TrainInputs:
    def __init__(self, action_space, observation_space):
        obs_shape = (None, *observation_space.shape)
        with tf.variable_scope('inputs'):
            self.lr = tf.placeholder(tf.float32, shape=[], name='LR')
            self.eps = tf.placeholder(tf.float32, shape=[], name='epsilon')
            self.s0 = tf.placeholder(tf.float32, shape=obs_shape, name="x0")
            self.act = tf.placeholder(tf.int32, [None], name="action")
            self.rew = tf.placeholder(tf.float32, [None], name="reward")
            self.s1 = tf.placeholder(tf.float32, shape=obs_shape, name="x1")
            self.done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
            self.sample_weights = tf.placeholder(tf.float32, [None], name="sample_weights")  # for importance sampling


class QTrainer():
    def __init__(self, inputs: TrainInputs, action_space, observation_space):
        act_size = action_space.n
        optimizer = tf.train.AdamOptimizer(learning_rate=inputs.lr)

        with tf.variable_scope('q_func'):  # child scopes of reusable parent scope are reusable
            self.runner = q_policy(obs=inputs.s0, epsilon=inputs.eps, action_space=action_space)

        with tf.variable_scope('q_func', reuse=True):  # child scopes of reusable parent scope are reusable
            q_net = q_policy(obs=inputs.s0, epsilon=inputs.eps, action_space=action_space)

        with tf.variable_scope('target_q_func'):
            target_q_net = q_policy(obs=inputs.s1, epsilon=inputs.eps, action_space=action_space)

        update_target_op = tf.group(*[tf.assign(a, b) for a, b in zip(target_q_net.trainables, q_net.trainables)])

        if G.double_q:
            with tf.variable_scope('q_func', reuse=True):  # child scopes of reusable parent scope are reusable
                inner_q_net = q_policy(obs=inputs.s1, epsilon=inputs.eps, action_space=action_space)

        with tf.variable_scope('Q_training'):
            q_sampled = tf.reduce_sum(q_net.q_values * tf.one_hot(inputs.act, act_size), axis=1)

            if G.double_q:
                q_asterisk = tf.reduce_sum(target_q_net.q_values * tf.one_hot(inner_q_net.act_argmax, act_size), axis=1)
            else:
                q_asterisk = tf.reduce_max(target_q_net.q_values, axis=1)

            # compute RHS of bellman equation
            T_q = inputs.rew + (1.0 - inputs.done_mask_ph) * G.gamma * q_asterisk

            # compute the error (potentially clipped)
            td_error = q_sampled - tf.stop_gradient(T_q)
            _ = U.huber_loss(td_error)
            if G.prioritized_replay:
                loss = tf.reduce_mean(inputs.sample_weights * _)
            else:
                loss = tf.reduce_mean(_)

            # compute optimization op (potentially with gradient clipping)
            if G.grad_norm_clipping:
                optimize_op = U.minimize_and_clip(optimizer, loss, var_list=q_net.trainables,
                                                  clip_val=G.grad_norm_clipping)
            else:
                optimize_op = optimizer.minimize(loss, var_list=q_net.trainables)

        def train(*, s0s, actions, rewards, s1s, dones, sample_weights=None):  # read: SARSA
            feed_dict = {inputs.lr: G.learning_rate,
                         inputs.s0: s0s,
                         inputs.act: actions,
                         inputs.rew: rewards,
                         inputs.s1: s1s,
                         inputs.done_mask_ph: dones}
            if G.prioritized_replay:
                assert sample_weights is not None, "sample_weights is required when prioritized_replay is ON."
                feed_dict[inputs.sample_weights] = sample_weights
            td_error_val, loss_val, _ = U.get_session().run([td_error, loss, optimize_op], feed_dict)
            return td_error_val, loss_val

        def update_target():
            U.get_session().run(update_target_op)

        self.train = train
        self.update_target = update_target


def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = monitor(env)
    env = ClippedRewardsWrapper(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    return env


def train():
    from linear_schedule import Linear

    ledger = defaultdict(lambda: MovingAverage(Reporting.reward_average))

    M.config(file=os.path.join(RUN.log_directory, RUN.log_file))
    M.diff()

    with U.make_session(RUN.num_cpu), Logger(RUN.log_directory) as logger, contextify(gym.make(G.env_name)) as env:
        env = ScaledFloatFrame(wrap_dqn(env))

        if G.seed is not None:
            env.seed(G.seed)
        logger.log_params(G=vars(G), RUN=vars(RUN), Reporting=vars(Reporting))
        inputs = TrainInputs(action_space=env.action_space, observation_space=env.observation_space)
        trainer = QTrainer(inputs=inputs, action_space=env.action_space, observation_space=env.observation_space)
        if G.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(size=G.buffer_size, alpha=G.alpha)
        else:
            replay_buffer = ReplayBuffer(size=G.buffer_size)

        class schedules:
            # note: it is important to have this start from the begining.
            eps = Linear(G.n_timesteps * G.exploration_fraction, 1, G.final_eps)
            if G.prioritized_replay:
                beta = Linear(G.n_timesteps - G.learning_start, G.beta_start, G.beta_end)

        U.initialize()
        trainer.update_target()
        x = np.array(env.reset())
        ep_ind = 0
        M.tic('episode')
        for t_step in range(G.n_timesteps):
            # schedules
            eps = 0 if G.param_noise else schedules.eps[t_step]
            if G.prioritized_replay:
                beta = schedules.beta[t_step - G.learning_start]

            x0 = x
            M.tic('sample', silent=True)
            (action, *_), action_q, q = trainer.runner.act([x], eps)
            x, rew, done, info = env.step(action)
            ledger['action_q_value'].append(action_q.max())
            ledger['action_q_value/mean'].append(action_q.mean())
            ledger['action_q_value/var'].append(action_q.var())
            ledger['q_value'].append(q.max())
            ledger['q_value/mean'].append(q.mean())
            ledger['q_value/var'].append(q.var())
            ledger['timing/sample'].append(M.toc('sample', silent=True))
            # note: adding sample to the buffer is identical between the prioritized and the standard replay strategy.
            replay_buffer.add(s0=x0, action=action, reward=rew, s1=x, done=float(done))

            logger.log(t_step, {'q_value': ledger['q_value'].latest,
                                'q_value/mean': ledger['q_value/mean'].latest,
                                'q_value/var': ledger['q_value/var'].latest,
                                'q_value/action': ledger['action_q_value'].latest,
                                'q_value/action/mean': ledger['action_q_value/mean'].latest,
                                'q_value/action/var': ledger['action_q_value/var'].latest},
                       action=action, eps=eps, silent=True)

            if G.prioritized_replay:
                logger.log(t_step, beta=beta, silent=True)

            if done:
                ledger['timing/episode'].append(M.split('episode', silent=True))
                ep_ind += 1
                x = np.array(env.reset())
                ledger['rewards'].append(info['total_reward'])

                silent = (ep_ind % Reporting.print_interval != 0)
                logger.log(t_step, timestep=t_step, episode=green(ep_ind), total_reward=ledger['rewards'].latest,
                           episode_length=info['timesteps'], silent=silent)
                logger.log(t_step, {'total_reward/mean': yellow(ledger['rewards'].mean, lambda v: f"{v:.1f}"),
                                    'total_reward/max': yellow(ledger['rewards'].max, lambda v: f"{v:.1f}"),
                                    "time_spent_exploring": default(eps, percent),
                                    "timing/episode": green(ledger['timing/episode'].latest, sec),
                                    "timing/episode/mean": green(ledger['timing/episode'].mean, sec),
                                    }, silent=silent)
                try:
                    logger.log(t_step,
                               {"timing/sample": default(ledger['timing/sample'].latest, sec),
                                "timing/sample/mean": default(ledger['timing/sample'].mean, sec),
                                "timing/train": default(ledger['timing/train'].latest, sec),
                                "timing/train/mean": green(ledger['timing/train'].mean, sec),
                                "timing/log_histogram": default(ledger['timing/log_histogram'].latest, sec),
                                "timing/log_histogram/mean": default(ledger['timing/log_histogram'].mean, sec)
                                }, silent=silent)
                    if G.prioritized_replay:
                        logger.log(t_step, {
                            "timing/update_priorities": default(ledger['timing/update_priorities'].latest, sec),
                            "timing/update_priorities/mean": default(ledger['timing/update_priorities'].mean, sec)
                        }, silent=silent)
                except Exception as e:
                    pass
                if G.prioritized_replay:
                    logger.log(t_step, {"replay_beta": default(beta, lambda v: f"{v:.2f}")}, silent=silent)

            # note: learn here.
            if t_step >= G.learning_start and t_step % G.learn_interval == 0:
                if G.prioritized_replay:
                    experiences, weights, indices = replay_buffer.sample(G.replay_batch_size, beta)
                    logger.log_histogram(t_step, weights=weights)
                else:
                    experiences, weights = replay_buffer.sample(G.replay_batch_size), None
                M.tic('train', silent=True)
                x0s, actions, rewards, x1s, dones = zip(*experiences)
                td_error_val, loss_val = trainer.train(s0s=x0s, actions=actions, rewards=rewards, s1s=x1s, dones=dones,
                                                       sample_weights=weights)
                ledger['timing/train'].append(M.toc('train', silent=True))
                M.tic('log_histogram', silent=True)
                logger.log_histogram(t_step, td_error=td_error_val)
                ledger['timing/log_histogram'].append(M.toc('log_histogram', silent=True))
                if G.prioritized_replay:
                    M.tic('update_priorities', silent=True)
                    new_priorities = np.abs(td_error_val) + eps
                    replay_buffer.update_priorities(indices, new_priorities)
                    ledger['timing/update_priorities'].append(M.toc('update_priorities', silent=True))

            if t_step % G.target_network_update_interval == 0:
                trainer.update_target()

            if t_step % Reporting.checkpoint_interval == 0:
                U.save_state(os.path.join(RUN.log_directory, RUN.checkpoint))


if __name__ == "__main__":
    train()
