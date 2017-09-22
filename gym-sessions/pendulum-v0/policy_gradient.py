import tensorflow as tf
import numpy as np
import time
import gym
from moleskin import Moleskin

import os

RUN_ID = os.environ['RUN_ID']

run_writer = tf.summary.FileWriter('/tmp/tensorflow/pendulum-v0/advantage/{RUN_ID}/run'.format(RUN_ID=RUN_ID))
learn_writer = tf.summary.FileWriter('/tmp/tensorflow/pendulum-v0/advantage/{RUN_ID}/learn'.format(RUN_ID=RUN_ID))

m = Moleskin()


def variable_summaries(var, scope_name='Summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):
        with tf.name_scope("mean"):
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('average', mean)
        tf.summary.scalar('variance', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def dense(out_n, x, scope_name='dense', nonlin=False):
    with tf.name_scope(scope_name):
        shape = x.get_shape().as_list()[-1:] + [out_n]
        params = tf.Variable(tf.random_normal_initializer(mean=0, stddev=1)(shape), name='Weights')
        out = tf.matmul(x, params)
        if nonlin:
            out = getattr(tf.nn, nonlin)(out)
        return out


def sample_action(mu, log_stddev, state_placeholder, scope_name="sample_action"):
    with tf.name_scope(scope_name):
        BATCH_N = tf.shape(state_placeholder)[0]
        action = (tf.random_normal([BATCH_N]) + mu) / (2 * tf.exp(log_stddev * 2))
    return action


def eligibility(actions_ph, mu, log_stddev, scope_name="eligibility"):
    with tf.name_scope(scope_name):
        elgb = - 0.5 * (np.log(2) + np.log(np.pi)) - log_stddev \
               - (actions_ph - mu) ** 2 / (2 * tf.exp(log_stddev * 2)) \
               - 1e-6
    return elgb


def scalar_summary(name, shape):
    with tf.name_scope(name):
        assert len(shape) <= 2, 'only scalar or 1-D vectors are allowed.'
        ph = tf.placeholder('float', shape, 'data')
        with tf.name_scope("mean"):
            mean = tf.reduce_mean(ph)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(ph - mean)))
        sums = [tf.summary.histogram(name, ph),
                tf.summary.scalar("average", mean),
                tf.summary.scalar("variance", stddev)]
        return ph


def placeholder_summary(name, ph):
    with tf.name_scope(name):
        assert len(ph.shape) <= 2, 'only scalar or 1-D vectors are allowed.'
        with tf.name_scope("mean"):
            mean = tf.reduce_mean(ph)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(ph - mean)))
        sums = [tf.summary.histogram(name, ph),
                tf.summary.scalar("average", mean),
                tf.summary.scalar("variance", stddev)]
    return sums


def surrogate(elgb, advantage, scope_name="surrogate_loss"):
    with tf.name_scope(scope_name):
        surrogate_loss = tf.multiply(elgb, advantage)
        loss = - tf.reduce_mean(surrogate_loss)
        # scalar_summary(loss)
        return loss


def value_function(state_ph, rewards_ph, value_lr, scope_name="value_function"):
    '''predicts the value of the current state, not sum of discounted future rewards.'''
    with tf.name_scope(scope_name):
        x = state_ph
        x = dense(128, x, scope_name="layer_1", nonlin='relu')
        x = dense(128, x, scope_name="layer_2", nonlin='relu')
        x = dense(128, x, scope_name="layer_3", nonlin='relu')
        x = dense(1, x, scope_name="layer_4")

        with tf.name_scope('optimizer'):
            error = rewards_ph - x
            loss = error ** 2
            optimizer = tf.train.AdamOptimizer(value_lr).minimize(loss)

        variable_summaries(rewards_ph, 'sampled_reward_value')
        variable_summaries(x, 'value_prediction')
        variable_summaries(error, 'error')
        variable_summaries(loss, 'loss')

        return x, optimizer


def policy_gradient():
    """usage:

    when running episode: (inside run_episode)
        sess_2.run(action_p, feed_dict={state: [s]})

    when running optimization:
        sess_2.run(optimizer, feed_dict={state: states, actions: [actions]})
    """
    with tf.name_scope("vanilla_policy_gradient"):
        # Placeholders
        lr_placeholder = tf.placeholder('float', [], name='Policy_Learning_Rate')
        value_lr_placeholder = tf.placeholder('float', [], name='Value_Learning_Rate')
        state_placeholder = tf.placeholder("float", [None, 3], name='IN_State')
        actions_placeholder = tf.placeholder("float", [None, 1], name='IN_Action')
        reward_placeholder = tf.placeholder("float", [None, 1], name='IN_Reward')

        with tf.name_scope('mu_NET'):
            x = state_placeholder
            x = dense(128, x, scope_name='layer_1', nonlin='relu')
            x = dense(128, x, scope_name='layer_2', nonlin='relu')
            x = dense(128, x, scope_name='layer_3', nonlin='relu')
            mu = dense(1, x, scope_name='layer_4', nonlin='tanh')

        with tf.name_scope('log_stddev_NET'):
            x = state_placeholder
            x = dense(128, x, scope_name='layer_1', nonlin='relu')
            x = dense(128, x, scope_name='layer_2', nonlin='relu')
            x = dense(128, x, scope_name='layer_3', nonlin='relu')
            log_stddev = dense(1, x, scope_name='layer_4', nonlin='tanh')

        action = sample_action(mu, log_stddev, state_placeholder)

        value, value_optimizer = value_function(state_placeholder, reward_placeholder, value_lr_placeholder)

        # calculate eligibility
        elgb = eligibility(actions_placeholder, mu, log_stddev)

        with tf.name_scope("advantage"):
            advantage = reward_placeholder - value

        surrogate_loss = surrogate(elgb, advantage)
        # variable_summaries(surrogate_loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(lr_placeholder).minimize(surrogate_loss)

        return action, \
               [state_placeholder, actions_placeholder, reward_placeholder], \
               [value, value_lr_placeholder, value_optimizer], \
               [lr_placeholder, optimizer], \
               [surrogate_loss]


def run_episode(run_ind, env, sess, merged_summary, state_ph, torque, sleep=0):
    env._max_episode_steps = MAX_STEPS
    s, r = env.reset(), -5000
    states = [s]
    actions = []
    rewards = []

    for step_ind in range(MAX_STEPS - 1):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        run_tag = 'run{}-step{}'.format(run_ind, step_ind)
        run_writer.add_run_metadata(run_metadata, run_tag)

        a, *_ = sess.run(
            torque,
            feed_dict={state_ph: [s]},
            options=run_options,
            run_metadata=run_metadata
        )

        s, r, done, _ = env.step(a)

        if DEBUG:
            env.render()
            if sleep:
                time.sleep(sleep)
            if done:
                time.sleep(0.1)
        if done:
            break
        states.append(s)
        actions.append(a)
        rewards.append(r)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    return states[:-1], actions, rewards


def summarize_run():
    with tf.name_scope('summary'):
        return [
            scalar_summary('cos', [MAX_STEPS - 1]),
            scalar_summary('sin', [MAX_STEPS - 1]),
            scalar_summary('angular-velocity', [MAX_STEPS - 1]),
            scalar_summary('rewards', [MAX_STEPS - 1]),
            scalar_summary('actions', [MAX_STEPS - 1])
        ]


def to_values(rewards, values):
    discounted_total_reward = 0
    total_values = []
    for r, adv in zip(rewards[::-1], values[::-1]):
        discounted_total_reward = (r - adv) + discounted_total_reward * gamma
        total_values.insert(0, discounted_total_reward)
    return total_values


MAX_STEPS = 200
TRAIN_BIAS = 4
DEBUG = False
env = gym.make('Pendulum-v0')

gamma = 0.99

ctheta, stheta, thetadot, r_ph, a_ph = summarize_run()
torque, placeholders, value, policy, etc = policy_gradient()
state_ph, actions_ph, rewards_ph = placeholders
value, value_learning_rate, value_optim = value
learning_rate, adam = policy

# run_config = tf.ConfigProto(log_device_placement=True)

episode_rewards = []
with tf.Session(config=None) as sess:
    run_writer.add_graph(sess.graph)
    learn_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()

    lr = 1e-3
    for ep_ind in range(100000):
        states, actions, rewards = run_episode(ep_ind, env, sess, merged_summary, state_ph, torque, sleep=0.05)

        run_sum = tf.summary.merge_all()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        VALUE_LOOPS = 10
        for op_ind in range(VALUE_LOOPS):
            values, _, run_sum_result = sess.run(
                [value, value_optim, run_sum],
                feed_dict={
                    state_ph: states, rewards_ph: np.expand_dims(rewards, axis=1),
                    value_learning_rate: 1e-3,
                    ctheta: states[:, 0], stheta: states[:, 1], thetadot: states[:, 2],
                    r_ph: rewards,
                    a_ph: actions[:, 0],
                },
                options=run_options,
                run_metadata=run_metadata
            )
            run_writer.add_summary(run_sum_result, ep_ind * VALUE_LOOPS + op_ind)

        avg_reward = sum(rewards) / len(rewards)
        episode_rewards.append(avg_reward)
        m.green(ep_ind, '\t', avg_reward, '\t{:1.4f}'.format(lr))

        POLICY_LOOPS = 3
        # if ep_ind % TRAIN_BIAS == TRAIN_BIAS - 1:
        if ep_ind > 25:
            MAX_STEPS = 200

            for op_ind in range(POLICY_LOOPS):
                advantage = to_values(rewards, values)

                *_, loss_val = \
                    sess.run([torque, adam, *etc],
                             feed_dict={
                                 learning_rate: lr,
                                 state_ph: states,
                                 actions_ph: actions,
                                 rewards_ph: advantage
                             },
                             options=run_options,
                             run_metadata=run_metadata
                             )

            if ep_ind % 50 == 10 and lr >= 1e-5:
                lr /= 1.02

        if len(episode_rewards) > 20 and np.mean(episode_rewards[-20:]) >= (MAX_STEPS - 1):
            print("finished after {} steps".format(ep_ind))
            break
    run_writer.close()
    learn_writer.close()
    m.green('graph is saved!!')
