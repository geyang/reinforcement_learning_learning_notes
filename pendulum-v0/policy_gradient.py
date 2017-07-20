import tensorflow as tf
import numpy as np
import time
import gym
from moleskin import Moleskin

run_writer = tf.summary.FileWriter('/tmp/tensorflow/pendulum-v0/advantage/run')
learn_writer = tf.summary.FileWriter('/tmp/tensorflow/pendulum-v0/advantage/learn')
m = Moleskin()


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope("mean"):
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def dense(out_n, x, scope_name='dense', nonlin=False):
    with tf.name_scope(scope_name):
        shape = x.get_shape().as_list()[-1:] + [out_n]
        params = tf.Variable(tf.random_normal_initializer(mean=0, stddev=1)(shape), name='Weights')
        out = tf.matmul(x, params)
        if nonlin:
            out = tf.nn.relu(out)
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
                tf.summary.scalar("mean", mean),
                tf.summary.scalar("stddev", stddev)]
        return ph


def surrogate(elgb, reward_ph, scope_name="surrogate_loss"):
    with tf.name_scope(scope_name):
        surrogate_loss = tf.multiply(elgb, reward_ph)
        loss = - tf.reduce_mean(surrogate_loss)
        # scalar_summary(loss)
        return loss


def value_function(state_ph, scope_name="Value_Network"):
    with tf.name_scope(scope_name):
        x = state_ph
        x = dense(40, x, scope_name="layer_1", nonlin=True)
        x = dense(20, x, scope_name="layer_2", nonlin=True)
        x = dense(1, x, scope_name="layer_3", nonlin=True)
        return x


def policy_gradient():
    """usage:

    when running episode: (inside run_episode)
        sess.run(action_p, feed_dict={state: [s]})

    when running optimization:
        sess.run(optimizer, feed_dict={state: states, actions: [actions]})
    """
    with tf.name_scope("vanilla_policy_gradient"):
        # Placeholders
        lr_placeholder = tf.placeholder('float', [], name='Learning_Rate')
        state_placeholder = tf.placeholder("float", [None, 3], name='IN_State')
        actions_placeholder = tf.placeholder("float", [None, 1], name='IN_Action')
        reward_placeholder = tf.placeholder("float", [None, 1], name='IN_Reward')

        with tf.name_scope('mu_NET'):
            x = state_placeholder
            x = dense(40, x, scope_name='layer_1', nonlin=True)
            x = dense(40, x, scope_name='layer_2', nonlin=True)
            x = dense(20, x, scope_name='layer_3', nonlin=True)
            mu = dense(1, x, scope_name='layer_4')

        with tf.name_scope('log_stddev_NET'):
            x = state_placeholder
            x = dense(30, state_placeholder, scope_name='layer_1', nonlin=True)
            x = dense(20, x, scope_name='layer_2', nonlin=True)
            x = dense(20, x, scope_name='layer_3', nonlin=True)
            x = dense(20, x, scope_name='layer_4', nonlin=True)
            log_stddev = dense(1, x, scope_name='layer_5', nonlin=True)

        action = sample_action(mu, log_stddev, state_placeholder)

        value = value_function(state_placeholder)

        # calculate eligibility
        elgb = eligibility(actions_placeholder, mu, log_stddev)

        surrogate_loss = surrogate(elgb, reward_placeholder)
        # variable_summaries(surrogate_loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(lr_placeholder).minimize(surrogate_loss)

        return action, \
               [state_placeholder, actions_placeholder, reward_placeholder], \
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

    global_index = run_ind

    # scalar_summary(states, 'states') # state too high dimension
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


def to_values(rewards):
    discounted_total_reward = 0
    values = []
    for r in rewards[::-1]:
        discounted_total_reward = r + discounted_total_reward * gamma
        values.insert(0, discounted_total_reward)
    return values


MAX_STEPS = 2000
DEBUG = False
env = gym.make('Pendulum-v0')

gamma = 0.9

ctheta, stheta, thetadot, r_ph, a_ph = summarize_run()
torque, placeholders, optim, etc = policy_gradient()
state_ph, actions_ph, rewards_ph = placeholders
learning_rate, adam = optim

# run_config = tf.ConfigProto(log_device_placement=True)

episode_rewards = []
with tf.Session(config=None) as sess:
    run_writer.add_graph(sess.graph)
    learn_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()

    lr = 5e-2
    for ep_ind in range(100000):
        states, actions, rewards = run_episode(ep_ind, env, sess, merged_summary, state_ph, torque, sleep=0.05)

        run_sum = tf.summary.merge_all()
        run_sum_result = sess.run(run_sum, feed_dict={
            ctheta: states[:, 0], stheta: states[:, 1], thetadot: states[:, 2],
            r_ph: rewards, a_ph: actions[:, 0]
        })
        run_writer.add_summary(run_sum_result, ep_ind)

        avg_reward = sum(rewards) / len(rewards)
        episode_rewards.append(avg_reward)
        # Now run optimizer
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        rewards_vector = to_values(rewards)

        *_, loss_val = \
            sess.run([torque, adam, *etc],
                     feed_dict={
                         learning_rate: lr,
                         state_ph: states,
                         actions_ph: actions,
                         # rewards_ph: np.array([rewards]).T})
                         rewards_ph: np.expand_dims(rewards_vector, axis=1)},
                     options=run_options,
                     run_metadata=run_metadata
                     )
        # learn_writer.add_summary(summary)
        if ep_ind % 50 == 10:
            lr /= 1.5
        if ep_ind > 100:
            DEBUG = True
        if episode_rewards[-1] >= -3:
            DEBUG = True
        print(ep_ind, '\t', avg_reward, '\t{:1.4f}'.format(lr))
        if len(episode_rewards) > 20 and np.mean(episode_rewards[-20:]) >= (MAX_STEPS - 1):
            print("finished after {} steps".format(ep_ind))
            break
    run_writer.close()
    learn_writer.close()
    m.green('graph is saved!!')
